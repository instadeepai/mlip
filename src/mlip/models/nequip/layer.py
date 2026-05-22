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
import os
from typing import Literal

import e3nn_jax as e3nn
import flax.linen as nn
from e3j.core import Permutation
from e3j.linen import LinearIndexwise
from e3j.utils.options import Layout

from mlip.graph import Graph
from mlip.models.message_passing import O3MessagePassingBlock
from mlip.models.nequip.nequip_helpers import split_target_node_irreps
from mlip.models.options import Activation, parse_activation


class NequipLayer(nn.Module):
    """NequIP Layer, consisting of a convolution block and a gate nonlinearity.

    Adapted from Google DeepMind materials discovery:
    https://github.com/google-deepmind/materials_discovery/blob/main/model/nequip.py

    Implementation follows the original paper by Batzner et al. (2022):
    nature.com/articles/s41467-022-29939-5 and partially
    https://github.com/mir-group/nequip.

    Args:
        target_irreps: Target irreps for the output node features. Acts as an
            upper bound — only paths reachable via the tensor product are included,
            so earlier layers may not fully achieve these irreps.
        source_node_irreps: Expected irreps of the input node features. Used to
            validate the graph at call time.
        l_max: Maximum degree of spherical harmonics. Used to validate the edge
            spherical embedding shape.
        num_rbf: Number of radial basis functions. Used to validate the radial
            embedding shape in the convolution block.
        use_residual_connection: If use residual connection in network (recommended).
        nonlinearities: Nonlinearities to use for even/odd irreps.
        radial_mlp_activation: Activation for the radial MLP.
        radial_mlp_hidden: Dimensions of hidden layers of the radial MLP.
        radial_mlp_variance_scale: Variance scaling for all-but-last layers of the
            radial MLP.
        avg_num_neighbors: Constant number of per-atom neighbors, used for internal
            normalization.
        deterministic_scatter_ops: Whether to use deterministic scatter operations.

    Returns:
        Graph containing updated node features after the convolution and gating.
    """

    target_irreps: e3nn.Irreps
    source_node_irreps: e3nn.Irreps
    l_max: int
    num_species: int
    num_rbf: int
    use_residual_connection: bool
    nonlinearities: dict[str, Activation]
    radial_mlp_activation: Activation | Literal["beta_swish"]
    radial_mlp_hidden: list[int]
    radial_mlp_variance_scale: float
    avg_num_neighbors: float
    deterministic_scatter_ops: bool = False

    def _input_shape_assertions(self, graph: Graph) -> None:
        node_feats = graph.nodes.features.get("latent")
        if node_feats is None:
            node_feats = graph.nodes.features["embedding"]
        species_one_hot = graph.nodes.features["species_one_hot"]
        spherical_embedding = graph.edges.features["spherical_embedding"]

        assert node_feats.ndim == 2
        assert node_feats.irreps == e3nn.Irreps(self.source_node_irreps)

        assert species_one_hot.ndim == 2
        assert species_one_hot.shape[1] == self.num_species

        assert spherical_embedding.ndim == 2
        assert spherical_embedding.irreps == e3nn.Irreps.spherical_harmonics(self.l_max)

    def setup(self):
        # Convolution block:
        # Note gate_irreps need to be regrouped, which permutes
        # scalars with gates when there are odd scalars.
        layout = (
            # Required to pass v1 consistency tests for now.
            # The exact ordering of isomorphic irreps differs between layouts.
            Layout.TRAILING_CHANNELS
            if os.environ.get("MLIP_TESTING") is None
            else Layout.E3NN
        )
        self.convolution_block = O3MessagePassingBlock(
            source_irreps=self.source_node_irreps,
            target_irreps=self.gate_irreps.regroup(),
            l_max=self.l_max,
            num_rbf=self.num_rbf,
            radial_mlp_activation=self.radial_mlp_activation,
            radial_mlp_hidden=self.radial_mlp_hidden,
            avg_num_neighbors=self.avg_num_neighbors,
            radial_mlp_variance_scale=self.radial_mlp_variance_scale,
            deterministic_scatter_ops=self.deterministic_scatter_ops,
            layout=layout,
            normalize_activation=False,
        )

        # Skip-connection:
        # Applies learnable species-dependent matrices on each degree l.
        # Equivalent to FullyConnectedTensorProduct with 1-hot species in v1.
        self.residual_block = LinearIndexwise(
            source_irreps=e3nn.Irreps(self.source_node_irreps),
            target_irreps=self.gate_irreps.regroup(),
            num_channels=None,
            num_indices=self.num_species,
            kernel_init="FAN_IN",
            rescale_gradients=True,
            layout="E3NN",
        )

        # Permutation from regrouped gate_irreps to gate_irreps.
        # Consistent ordering by (l, p) between source and target is needed for
        # Linear blocks, but gate irreps interleave 0e with 0o on target only.
        perm_inv = Permutation.sort(self.gate_irreps)
        self.permutation_block = Permutation(perm_inv.sigma_1)

    @property
    def gate_triplet(self) -> e3nn.Irreps:
        scalars, gates, nonscalars = split_target_node_irreps(
            self.source_node_irreps,
            self.spherical_irreps,
            self.target_irreps,
        )
        return scalars, gates, nonscalars

    @property
    def gate_irreps(self) -> e3nn.Irreps:
        scalars, gates, nonscalars = self.gate_triplet
        return scalars + gates + nonscalars

    @property
    def spherical_irreps(self) -> e3nn.Irreps:
        return e3nn.Irreps.spherical_harmonics(self.l_max)

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        """Apply the Nequip layer to update the node features of the input graph.

        Args:
            graph: Graph containing precomputed node and edge embeddings,
                   and current node features `graph.nodes.features["latent"]`, or
                   if this is the first layer `graph.nodes.features["embedding"]`.

        Returns:
            Graph with updated node features `graph.nodes.features["latent"]`.
        """
        self._input_shape_assertions(graph)

        node_feats = graph.nodes.features.get("latent")
        if node_feats is None:
            node_feats = graph.nodes.features["embedding"]
            graph = graph.update_node_features(latent=node_feats)

        node_species = graph.nodes.features["species"]

        if self.use_residual_connection:
            residuals = self.residual_block(node_feats.array, node_species)

        # Message-passing step
        graph = self.convolution_block(graph)

        node_feats = graph.nodes.features["latent"]
        node_dtype = node_feats.dtype  # capture after convolution; gate may upcast

        # Add residuals before gating
        if self.use_residual_connection:
            node_feats = node_feats + e3nn.IrrepsArray(node_feats.irreps, residuals)

        # Gate nonlinearity:
        # The e3nn API requires us to sort regrouped node features to interleave
        # 0e/0o gating scalars in between the scalar and non-scalar features.
        node_feats = e3nn.IrrepsArray(
            self.gate_irreps,
            self.permutation_block(node_feats.array),
        )
        node_feats = e3nn.gate(
            node_feats,
            even_act=parse_activation(self.nonlinearities["e"]),
            odd_act=parse_activation(self.nonlinearities["o"]),
            even_gate_act=parse_activation(self.nonlinearities["e"]),
            odd_gate_act=parse_activation(self.nonlinearities["o"]),
        )
        node_feats = node_feats.astype(node_dtype)

        return graph.update_node_features(latent=node_feats)
