# MIT License
# Copyright (c) 2022 mace-jax
# See https://github.com/ACEsuit/mace-jax/blob/main/MIT.md
#
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

from typing import Callable, Literal

import e3nn_jax as e3nn
import flax.linen as nn
from e3j.linen import LinearIndexwise
from e3j.utils.options import Layout
from jax import Array

from mlip.graph import Graph
from mlip.models import options
from mlip.models.gaunt_tensor_product import GauntMessagePassingBlock
from mlip.models.mace.blocks import EquivariantProductBasisBlock
from mlip.models.mace.config import MaceConfig  # noqa
from mlip.models.message_passing import O3MessagePassingBlock
from mlip.models.options import (
    Activation,
)
from mlip.models.readout import MultiHeadReadoutBlock


class MaceLayer(nn.Module):
    """A single MACE layer.

    This module assumes that the input graph has been transformed by
    :class:`mlip.models.blocks.MaceEmbeddingBlock` upstream.
    """

    use_residuals: bool
    last_layer: bool
    num_channels: int
    source_irreps: e3nn.Irreps
    interaction_irreps: e3nn.Irreps
    node_irreps: e3nn.Irreps
    activation: Activation | Callable[[Array], Array]
    num_species: int
    # InteractionBlock:
    l_max: int
    num_rbf: int
    radial_mlp_hidden: list[int]
    radial_mlp_activation: Activation | Callable[[Array], Array]
    avg_num_neighbors: float
    # EquivariantProductBasisBlock:
    correlation: int
    soft_normalization: float | None
    # ReadoutBlock:
    output_irreps: e3nn.Irreps
    readout_mlp_irreps: e3nn.Irreps
    num_readout_heads: int = 1
    gate_nodes: bool = False
    deterministic_scatter_ops: bool = False
    symmetric_contraction_backend: Literal[
        "e3j", "e3nn", "e3nn_symmetric", "gaunt_tp"
    ] = "e3j"
    use_gaunt_tp_message_passing: bool = False

    def setup(self):
        source_irreps = e3nn.Irreps(self.source_irreps)
        interaction_irreps = e3nn.Irreps(self.interaction_irreps)
        node_irreps = e3nn.Irreps(self.node_irreps)
        output_irreps = e3nn.Irreps(self.output_irreps)
        readout_mlp_irreps = e3nn.Irreps(self.readout_mlp_irreps)

        # Interaction block:
        # wraps message-passing convolution between Linear layers.
        if self.use_gaunt_tp_message_passing:
            self.interaction_block = GauntMessagePassingBlock(
                source_irreps=self.num_channels * source_irreps,
                target_irreps=self.num_channels * interaction_irreps,
                l_max=self.l_max,
                num_rbf=self.num_rbf,
                radial_mlp_hidden=self.radial_mlp_hidden,
                radial_mlp_activation=self.radial_mlp_activation,
                avg_num_neighbors=self.avg_num_neighbors,
                deterministic_scatter_ops=self.deterministic_scatter_ops,
                radial_mlp_variance_scale=None,
            )
        else:
            self.interaction_block = O3MessagePassingBlock(
                source_irreps=self.num_channels * source_irreps,
                target_irreps=self.num_channels * interaction_irreps,
                l_max=self.l_max,
                num_rbf=self.num_rbf,
                radial_mlp_hidden=self.radial_mlp_hidden,
                radial_mlp_activation=self.radial_mlp_activation,
                avg_num_neighbors=self.avg_num_neighbors,
                deterministic_scatter_ops=self.deterministic_scatter_ops,
                radial_mlp_variance_scale=None,
            )

        # Expansion block:
        # exponentiates node features, keeping degrees < node_symmetry only.
        self.expansion_block = EquivariantProductBasisBlock(
            source_irreps=self.num_channels * interaction_irreps,
            target_irreps=self.num_channels
            * (e3nn.Irreps("0e") if self.last_layer else node_irreps),
            correlation=self.correlation,
            num_species=self.num_species,
            num_channels=self.num_channels,
            gate_nodes=self.gate_nodes,
            symmetric_contraction_backend=self.symmetric_contraction_backend,
        )

        # Skip-connection:
        # LinearIndexwise applies specie-dependent linear matrices.
        # Equivalent to the FullyConnectedTensorProduct with one-hot species in v1.
        if self.use_residuals:
            target_irreps = e3nn.Irreps("0e") if self.last_layer else node_irreps
            self.residual_block = LinearIndexwise(
                self.num_channels * source_irreps,
                self.num_channels * target_irreps,
                num_indices=self.num_species,
                num_channels=None,
                kernel_init="FAN_IN",
                rescale_gradients=True,
                layout=Layout.E3NN,
            )

        # A self-interaction block applied right after the InteractionBlock
        # when no residual connection.
        else:
            self.linear_by_species_block = LinearIndexwise(
                source_irreps=self.num_channels * interaction_irreps,
                target_irreps=self.num_channels * interaction_irreps,
                num_indices=self.num_species,
                num_channels=None,
                kernel_init="FAN_IN",
                rescale_gradients=True,
                layout=Layout.E3NN,
            )

        # Readout Blocks:
        # - one readout block for each readout head
        # - each readout block is linear before the last layer, MLP then.
        if not self.last_layer:
            self.readout_block = MultiHeadReadoutBlock(
                num_heads=self.num_readout_heads,
                features=(output_irreps,),
                activation=None,
                use_equiv=True,
            )
        else:
            self.readout_block = MultiHeadReadoutBlock(
                num_heads=self.num_readout_heads,
                features=self._get_feature_sizes_for_nonlinear_readout(
                    readout_mlp_irreps, output_irreps
                ),
                activation=options.parse_activation(self.activation),
                use_equiv=True,
            )

    def _get_feature_sizes_for_nonlinear_readout(
        self, hidden_irreps: e3nn.Irreps, output_irreps: e3nn.Irreps
    ) -> tuple[e3nn.Irreps, e3nn.Irreps]:
        num_vectors = hidden_irreps.filter(
            drop=["0e", "0o"]
        ).num_irreps  # Multiplicity of (l > 0) irreps
        return (
            (hidden_irreps + e3nn.Irreps(f"{num_vectors}x0e")).simplify(),
            output_irreps,
        )

    def normalize_features(self, node_feats: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        """Apply a soft saturation function on the norm of node features."""

        def phi(n):
            n = n / self.soft_normalization
            return 1.0 / (1.0 + n * e3nn.sus(n))

        node_feats = e3nn.norm_activation(node_feats, [phi] * len(node_feats.irreps))

        return node_feats

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        """Apply the `MaceLayer` on a featured graph.

        Args:
            graph: an input graph which should have the following features assigned:
                * `edges.features['spherical_embedding']`
                * `edges.features['radial_embedding']`

        Returns:
            a graph with the following features updated:
                * `nodes.features['latent']`
                * `nodes.features['outputs']`
        """
        # residual connection:
        residuals = None

        node_species = graph.nodes.features["species"]

        node_feats = graph.nodes.features.get("latent")
        if node_feats is None:
            node_feats = graph.nodes.features["embedding"]
            graph = graph.update_node_features(latent=node_feats)

        if self.use_residuals:
            assert node_feats.irreps == self.residual_block.source
            residuals = e3nn.IrrepsArray(
                self.residual_block.target,
                self.residual_block(node_feats.array, node_species),
            )

        # Interaction block
        graph = self.interaction_block(graph)

        # selector tensor product (first layer only)
        node_feats = graph.nodes.features["latent"]
        if not self.use_residuals:
            node_feats = e3nn.IrrepsArray(
                node_feats.irreps,
                self.linear_by_species_block(node_feats.array, node_species),
            )
            graph = graph.update_node_features(latent=node_feats)

        # Exponentiate node features, keep degrees < node_symmetry only
        graph = self.expansion_block(graph)

        # Apply residuals (and optional soft normalization)
        node_feats = graph.nodes.features["latent"]

        if self.soft_normalization is not None:
            node_feats = self.normalize_features(node_feats)

        if residuals is not None:
            node_feats += residuals

        graph = graph.update_node_features(latent=node_feats)

        # Apply readout block: populates features["outputs"]
        return self.readout_block(graph)
