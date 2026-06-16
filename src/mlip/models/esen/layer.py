# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Literal, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from mlip.graph import Graph
from mlip.models.esen.coefficient_mapping import CoefficientMapping
from mlip.models.esen.esen_helpers import (
    NODE_OFFSET,
    GateActivation,
    SO2Convolution,
    SO3Linear,
)
from mlip.models.esen.moe import (
    expand_graph_coeffs_to_edges,
    get_graph_moe_coefficients,
)
from mlip.models.esen.normalisation import get_normalization_layer
from mlip.utils.jax_utils import segment_sum


class ESENLayer(nn.Module):
    sphere_channels: int
    hidden_channels: int
    l_max: int
    m_max: int
    mapping_reduced: CoefficientMapping
    edge_channels_list: Sequence[int]
    graph_cutoff_angstrom: float
    norm_type: Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"]
    act_type: Literal["gate"]
    num_experts: int | None = None
    deterministic_scatter_ops: bool = False
    use_remat_edgewise: bool = False

    def setup(self) -> None:
        """Initializes the Esen layer."""
        # norms
        self.norm_1 = get_normalization_layer(
            self.norm_type, l_max=self.l_max, num_channels=self.sphere_channels
        )
        self.norm_2 = get_normalization_layer(
            self.norm_type, l_max=self.l_max, num_channels=self.sphere_channels
        )

        # edgewise
        EdgewiseCls = nn.remat(Edgewise) if self.use_remat_edgewise else Edgewise
        self.edge_wise = EdgewiseCls(
            sphere_channels=self.sphere_channels,
            hidden_channels=self.hidden_channels,
            l_max=self.l_max,
            m_max=self.m_max,
            edge_channels_list=self.edge_channels_list,
            mapping_reduced=self.mapping_reduced,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            act_type=self.act_type,
            num_experts=self.num_experts,
            deterministic_scatter_ops=self.deterministic_scatter_ops,
        )

        # atomwise (feed-forward head) spectral is default in UMA
        self.atom_wise = SpectralAtomwise(
            sphere_channels=self.sphere_channels,
            hidden_channels=self.hidden_channels,
            l_max=self.l_max,
            m_max=self.m_max,
        )

    def _input_shape_assertions(self, graph: Graph) -> None:
        node_feats = graph.nodes.features.get("latent")
        if node_feats is None:
            node_feats = graph.nodes.features["embedding"]
        assert node_feats.ndim == 3
        assert node_feats.shape[1] == ((self.l_max + 1) ** 2)
        assert node_feats.shape[2] == self.sphere_channels

    def __call__(self, graph: Graph) -> Graph:
        """Applies the Esen layer to update node features using neighboring nodes'
        species and edge features.

        Updated features in this function:
        - node-wise features: graph.nodes.features["latent"]
        - edge-wise features: graph.edges.features["latent"]

        Args:
            graph: Graph containing node features with "latent" or "embedding" if this
                   is a first layer. Same for edge features.

        Returns:
            Updated Graph with updated node features.
        """
        self._input_shape_assertions(graph)

        node_feats = graph.nodes.features.get("latent")
        if node_feats is None:
            node_feats = graph.nodes.features["embedding"]
            graph = graph.update_node_features(latent=node_feats)

        edge_feats = graph.edges.features.get("latent")
        if edge_feats is None:
            edge_feats = graph.edges.features["embedding"]
            graph = graph.update_edge_features(latent=edge_feats)

        # First norm
        node_feats_res = node_feats
        node_feats = self.norm_1(node_feats)
        graph = graph.update_node_features(latent=node_feats)

        # Edge-wise + residual
        graph = self.edge_wise(graph=graph)
        graph = self._residual_connection(graph=graph, node_feats_res=node_feats_res)

        # Second norm
        node_feats_res = graph.nodes.features["latent"]
        node_feats = self.norm_2(graph.nodes.features["latent"])
        graph = graph.update_node_features(latent=node_feats)

        # Atom-wise (feed-forward) + residual
        graph = self.atom_wise(graph)
        graph = self._residual_connection(graph=graph, node_feats_res=node_feats_res)

        return graph

    def _residual_connection(self, graph: Graph, node_feats_res: jax.Array) -> Graph:
        return graph.update_node_features(
            latent=graph.nodes.features["latent"] + node_feats_res
        )


class Edgewise(nn.Module):
    """Applies the edge-wise convolution to update node features using neighboring
    nodes' species and edge features.

    Attributes:
        sphere_channels: The number of channels for the node embedding.
        hidden_channels: The number of channels for the hidden layer.
        l_max: Highest harmonic order included in the Spherical Harmonics series.
        m_max: Maximum order of the spherical harmonics to include in the embedding.
        edge_channels_list: The list of channels for the edge embedding.
        mapping_reduced: The mapping of the spherical harmonics to the reduced set of
                        coefficients.
        graph_cutoff_angstrom: The cutoff radius for the graph.
        act_type: The type of activation function to apply.
    """

    sphere_channels: int
    hidden_channels: int
    l_max: int
    m_max: int
    edge_channels_list: Sequence[int]
    mapping_reduced: object
    graph_cutoff_angstrom: float
    act_type: Literal["gate"] = "gate"
    num_experts: int | None = None
    deterministic_scatter_ops: bool = False

    def setup(self) -> None:
        """Initializes the edge-wise convolution layers."""
        if self.act_type == "gate":
            self.act = GateActivation(
                l_max=self.l_max,
                m_max=self.m_max,
                num_channels=self.hidden_channels,
                m_prime=True,
            )
            extra_m0_output_channels = self.l_max * self.hidden_channels
        else:
            raise ValueError(f"Unknown activation type {self.act_type}")

        self.so2_conv_1 = SO2Convolution(
            sphere_channels=2 * self.sphere_channels,
            m_output_channels=self.hidden_channels,
            l_max=self.l_max,
            m_max=self.m_max,
            mapping_reduced=self.mapping_reduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list,
            extra_m0_output_channels=extra_m0_output_channels,
            num_experts=self.num_experts,
        )

        self.so2_conv_2 = SO2Convolution(
            sphere_channels=self.hidden_channels,
            m_output_channels=self.sphere_channels,
            l_max=self.l_max,
            m_max=self.m_max,
            mapping_reduced=self.mapping_reduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
            num_experts=self.num_experts,
        )

    def _input_shape_assertions(self, graph: Graph) -> None:
        assert graph.nodes.features["latent"].ndim == 3
        assert graph.nodes.features["latent"].shape[1] == ((self.l_max + 1) ** 2)
        assert graph.nodes.features["latent"].shape[2] == self.sphere_channels
        assert graph.edges.features["latent"].ndim == 2
        assert graph.edges.features["envelope"].ndim == 3
        assert (
            graph.edges.features["envelope"].shape[0]
            == graph.edges.features["latent"].shape[0]
        )
        assert graph.edges.features["envelope"].shape[1] == 1
        assert graph.edges.features["envelope"].shape[2] == 1
        assert (
            graph.edges.features["latent"].shape[0]
            == graph.senders.shape[0]
            == graph.receivers.shape[0]
        )

    def __call__(self, graph: Graph) -> Graph:
        """Applies the edge-wise convolution to update node features using neighboring
        nodes' species and edge features.

        Steps:
          1) gather source/target node features -> concat along channels
          2) rotate (align with edge)
          3) SO2 conv 1 -> activation (gate / s2)
          4) SO2 conv 2
          5) apply envelope
          6) rotate back
          7) scatter-add to destination nodes

        Updated features in this function:
        - node-wise features: graph.nodes.features["latent"]

        Args:
            graph: Graph containing node features with "latent" and edge features
            with "latent".

        Returns:
            Updated Graph with updated node features.
        """
        self._input_shape_assertions(graph)

        node_feats = graph.nodes.features["latent"]
        edge_feats = graph.edges.features["latent"]
        senders, receivers = graph.senders, graph.receivers
        wigner_and_m_mapping = graph.edges.features["wigner_and_m_mapping"]
        edge_envelope = graph.edges.features["envelope"]

        num_nodes = node_feats.shape[0]

        source_node_feats = node_feats[senders]
        target_node_feats = node_feats[receivers]

        # Concatenate along channels
        node_feats = jnp.concatenate([source_node_feats, target_node_feats], axis=-1)

        # Rotate to edge frame: bmm -> einsum
        node_feats = jnp.einsum("eij,ejc->eic", wigner_and_m_mapping, node_feats)

        edge_moe_coeffs = None
        if self.num_experts is not None:
            graph_moe_coeffs = get_graph_moe_coefficients(
                graph,
                self.num_experts,
            )
            edge_moe_coeffs = expand_graph_coeffs_to_edges(
                graph_moe_coeffs,
                graph.n_edge,
                total_edges=graph.senders.shape[0],
            )

        # SO2 conv 1 (may use x_edge and/or edge_distance internally)
        edge_messages, gating_scalars = self.so2_conv_1(
            node_feats,
            edge_feats,
            edge_moe_coeffs,
        )

        # Activation in M' space
        edge_messages = self.act(gating_scalars, edge_messages)

        # SO2 conv 2
        edge_messages = self.so2_conv_2(edge_messages, edge_feats, edge_moe_coeffs)

        # Envelope per-edge
        edge_messages = edge_messages * edge_envelope

        # Rotate back
        wigner_and_m_mapping_inv = jnp.swapaxes(wigner_and_m_mapping, 1, 2)
        edge_messages = jnp.einsum(
            "eij,ejc->eic", wigner_and_m_mapping_inv, edge_messages
        )

        # Scatter-add onto destination nodes
        dst_adj = (receivers - NODE_OFFSET).astype(jnp.int32)
        node_feats = segment_sum(
            edge_messages,
            dst_adj,
            num_segments=num_nodes,
            deterministic=self.deterministic_scatter_ops,
        )

        graph = graph.update_node_features(latent=node_feats)

        return graph


class SpectralAtomwise(nn.Module):
    """Applies the spectral atom-wise convolution to update node features.

    Attributes:
        sphere_channels: The number of channels for the node embedding.
        hidden_channels: The number of channels for the hidden layer.
        l_max: Highest harmonic order included in the Spherical Harmonics series.
        m_max: Maximum order of the spherical harmonics to include in the embedding.
    """

    sphere_channels: int
    hidden_channels: int
    l_max: int
    m_max: int

    def setup(self) -> None:
        """Initializes the spectral atom-wise layers."""
        self.scalar_dense = nn.Dense(self.l_max * self.hidden_channels, use_bias=True)

        self.so3_linear_1 = SO3Linear(
            in_channels=self.sphere_channels,
            out_channels=self.hidden_channels,
            l_max=self.l_max,
        )
        self.act = GateActivation(
            l_max=self.l_max,
            m_max=self.l_max,
            num_channels=self.hidden_channels,
        )
        self.so3_linear_2 = SO3Linear(
            in_channels=self.hidden_channels,
            out_channels=self.sphere_channels,
            l_max=self.l_max,
        )

    def _input_shape_assertions(self, graph: Graph) -> None:
        assert graph.nodes.features["latent"].ndim == 3
        assert graph.nodes.features["latent"].shape[1] == ((self.l_max + 1) ** 2)
        assert graph.nodes.features["latent"].shape[2] == self.sphere_channels

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        """Applies the spectral atom-wise convolution to update node features.

        Updated features in this function:
        - node-wise features: graph.nodes.features["latent"]

        Args:
            graph: Graph containing node features with "latent".

        Returns:
            Updated Graph with updated node features.
        """
        self._input_shape_assertions(graph)

        node_feats = graph.nodes.features["latent"]
        num_nodes = node_feats.shape[0]

        scalars = node_feats[:, :1, :]
        gating_scalars = self.scalar_dense(scalars)
        gating_scalars = nn.silu(gating_scalars)
        gating_scalars = gating_scalars.reshape(num_nodes, -1)

        node_feats = self.so3_linear_1(node_feats)
        node_feats = self.act(gating_scalars, node_feats)
        node_feats = self.so3_linear_2(node_feats)

        graph = graph.update_node_features(latent=node_feats)

        return graph
