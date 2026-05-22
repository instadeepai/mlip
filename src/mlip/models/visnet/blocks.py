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

import functools
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers

from mlip.graph import Graph
from mlip.models.blocks import (
    MLP,
    JointNodeEmbeddingBlock,
    NodeEmbeddingBlock,
    RadialEmbeddingBlock,
    SphericalHarmonicsBlock,
)
from mlip.models.options import RadialBasis, RadialEnvelope, parse_activation
from mlip.models.radial_embedding import cosine_cutoff
from mlip.models.visnet.visnet_helpers import (
    LAYER_NORM_EPSILON,
    VEC_LAYER_NORM_EPSILON,
    VecLayerNorm,
)
from mlip.utils.jax_utils import segment_sum
from mlip.utils.safe_norm import safe_norm


class VisnetEmbeddingBlock(nn.Module):
    """Embeds input node and edge features for the ViSNet model.

    Initializes and applies the embedding layers for node species, radial
    functions, and spherical harmonics. Updates the input Graph with the embedded
    features and pre-processes edges and neighbors for subsequent network layers.

    Attributes:
        l_max: Highest harmonic order included in the Spherical Harmonics series.
        num_channels: The number of channels.
        num_rbf: Number of basis functions used in the embedding block.
        trainable_rbf: Whether to add learnable weights to each of the radial embedding
                       basis functions.
        graph_cutoff_angstrom: The cutoff radius for the graph.
        num_species: The number of elements (atomic species descriptors) allowed.
        deterministic_scatter_ops: Whether to use deterministic scatter operations.
    """

    l_max: int
    num_channels: int
    num_rbf: int
    trainable_rbf: bool
    graph_cutoff_angstrom: float
    num_species: int
    num_charges: int | None
    radial_basis: str | RadialBasis
    activation_fn: Callable | None
    deterministic_scatter_ops: bool = False

    def setup(self) -> None:
        """Initializes the embedding layers for node species, radial functions,
        and spherical harmonics, neighbor and edge embedding blocks.
        """
        if self.num_charges is not None:
            self.node_embedding = JointNodeEmbeddingBlock(
                num_species=self.num_species,
                num_charge=self.num_charges,
                num_channels=self.num_channels,
                activation_fn=self.activation_fn,
            )
        else:
            self.node_embedding = NodeEmbeddingBlock(
                num_species=self.num_species,
                num_channels=self.num_channels,
            )

        _radial_envelope = None
        _radial_basis = RadialBasis(self.radial_basis)
        if _radial_basis == RadialBasis.EXPNORM:
            _radial_envelope = RadialEnvelope.COSINE_CUTOFF
        elif _radial_basis == RadialBasis.BESSEL:
            _radial_envelope = RadialEnvelope.POLYNOMIAL
        self.radial_embedding = RadialEmbeddingBlock(
            radial_basis=_radial_basis,
            radial_envelope=_radial_envelope,
            num_rbf=self.num_rbf,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            learnable=self.trainable_rbf,
        )

        self.spherical_embedding = SphericalHarmonicsBlock(
            l_max=self.l_max,
            normalize=True,
            normalization="norm",
        )

        self.neighbor_embedding = VisnetNeighborEmbeddingBlock(
            num_channels=self.num_channels,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            num_species=self.num_species,
            num_rbf=self.num_rbf,
            deterministic_scatter_ops=self.deterministic_scatter_ops,
        )

        self.edge_embedding = VisnetEdgeEmbeddingBlock(
            num_channels=self.num_channels,
            num_rbf=self.num_rbf,
        )

    def _input_shape_assertions(self, graph: Graph) -> None:
        edge_vectors = graph.edges.features["vectors"]
        node_species = graph.nodes.features["species"]

        assert edge_vectors.ndim == 2 and edge_vectors.shape[1] == 3
        assert node_species.ndim == 1
        assert graph.senders.ndim == 1 and graph.receivers.ndim == 1
        assert (
            edge_vectors.shape[0] == graph.senders.shape[0] == graph.receivers.shape[0]
        )

    def __call__(self, graph: Graph) -> Graph:
        """Applies embedding transformations to the input graph.

        Updates the graph with node, edge, and spherical harmonic features, processes
        neighbor and edge information for ViSNet layers, and initializes vector
        features. Returns the updated graph.

        Embedded features in the final graph can be accessed as:
        - scalar node features: graph.nodes.features["embedding_scalars"]
        - vector features: graph.nodes.features["embedding_vectors"]
        - edge features: graph.edges.features["embedding"]
        - edge distances: graph.edges.features["distances"]
        - spherical harmonic features: graph.edges.features["spherical_embedding"]

        Args:
            graph: Input Graph object with atomic positions, node species, and
                edge vectors.

        Returns:
            Updated Graph with embedded node and edge features ready for ViSNet
            processing.
        """
        self._input_shape_assertions(graph)

        edge_vectors = graph.edges.features["vectors"]
        node_species = graph.nodes.features["species"]

        # Calculate distances
        distances = safe_norm(edge_vectors, axis=-1)

        # Embedding Layers
        if self.num_charges is not None:
            charge_indices = graph.nodes.features["charge_indices"]
            node_feats = self.node_embedding(node_species, charge_indices)
        else:
            node_feats = self.node_embedding(node_species)

        # Seems like doubled from within the neighbor embedding module
        edge_feats = self.radial_embedding(distances)

        spherical_feats = self.spherical_embedding(
            edge_vectors / (distances[:, None] + 1e-8)
        ).array[:, 1:]

        graph = graph.update_node_features(embedding_scalars=node_feats)
        graph = graph.update_edge_features(
            embedding=edge_feats,
            spherical_embedding=spherical_feats,
            distances=distances,
        )

        graph = self.neighbor_embedding(graph)

        graph = self.edge_embedding(graph)

        vec_shape = (
            node_feats.shape[0],
            ((self.l_max + 1) ** 2) - 1,
            node_feats.shape[1],
        )
        vector_feats = jnp.zeros(vec_shape, dtype=node_feats.dtype)
        graph = graph.update_node_features(embedding_vectors=vector_feats)

        return graph


class VisnetNeighborEmbeddingBlock(nn.Module):
    """Applies the neighbor embedding to update node features using neighboring
    nodes' species and edge features.

    Attributes:
        num_channels: The number of channels.
        graph_cutoff_angstrom: The cutoff radius for the graph.
        num_species: The number of elements (atomic species descriptors) allowed.
        num_rbf: Number of basis functions used in the embedding block.
        deterministic_scatter_ops: Whether to use deterministic scatter operations.
    """

    num_channels: int
    graph_cutoff_angstrom: float
    num_species: int
    num_rbf: int  # Required for input shape assertions.
    deterministic_scatter_ops: bool = False

    def setup(self) -> None:
        """Initializes the neighbor embedding layers."""
        self.embedding = nn.Embed(
            num_embeddings=self.num_species,
            features=self.num_channels,
        )
        self.distance_proj = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.combine = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.cutoff_fn = functools.partial(
            cosine_cutoff, graph_cutoff_angstrom=self.graph_cutoff_angstrom
        )

    def _input_shape_assertions(self, graph: Graph) -> None:
        assert graph.nodes.features["embedding_scalars"].ndim == 2
        assert graph.nodes.features["embedding_scalars"].shape[1] == self.num_channels
        assert graph.edges.features["embedding"].ndim == 2
        assert graph.edges.features["embedding"].shape[1] == self.num_rbf
        assert graph.senders.ndim == 1 and graph.receivers.ndim == 1
        assert (
            graph.edges.features["embedding"].shape[0]
            == graph.senders.shape[0]
            == graph.receivers.shape[0]
        )

    def _message_fn(self, x_j: jax.Array, w: jax.Array) -> jax.Array:
        return x_j * w

    def __call__(self, graph: Graph) -> Graph:
        """Applies the neighbor embedding to update node features using neighboring
        nodes' species and edge features.

        Args:
            graph: Input Graph containing node features (species, node_feats), edge
                features (edge_feats, distances), and graph topology (senders,
                receivers).

        Returns:
            Updated Graph object with new node features
            (field name: "embedding_scalars").
        """
        self._input_shape_assertions(graph)

        node_feats = graph.nodes.features["embedding_scalars"]

        cutoffs = self.cutoff_fn(graph.edges.features["distances"])

        edge_feats = graph.edges.features["embedding"]
        weights = self.distance_proj(edge_feats) * cutoffs[:, jnp.newaxis]

        embbedded_nodes = self.embedding(graph.nodes.features["species"])
        embedded_senders = embbedded_nodes[graph.senders]

        node_msgs = self._message_fn(embedded_senders, weights)
        aggregated_msgs = segment_sum(
            node_msgs,
            graph.receivers,
            num_segments=node_feats.shape[0],
            deterministic=self.deterministic_scatter_ops,
        )

        # Update between x and aggregated_messages over neighbors
        node_feats = self.combine(
            jnp.concatenate([node_feats, aggregated_msgs], axis=1)
        )

        return graph.update_node_features(embedding_scalars=node_feats)


class VisnetEdgeEmbeddingBlock(nn.Module):
    """Applies the edge embedding to update edge features using node features.

    Attributes:
        num_channels: The number of channels.
        num_rbf: Number of basis functions used in the embedding block.
    """

    num_channels: int
    num_rbf: int  # Required for input shape assertions.

    def setup(self) -> None:
        """Initializes the edge embedding layers."""
        self.edge_proj = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )

    def _input_shape_assertions(self, graph: Graph) -> None:
        assert graph.nodes.features["embedding_scalars"].ndim == 2
        assert graph.nodes.features["embedding_scalars"].shape[1] == self.num_channels
        assert graph.edges.features["embedding"].ndim == 2
        assert graph.edges.features["embedding"].shape[1] == self.num_rbf
        assert graph.senders.ndim == 1 and graph.receivers.ndim == 1
        assert (
            graph.edges.features["embedding"].shape[0]
            == graph.senders.shape[0]
            == graph.receivers.shape[0]
        )

    def _message_fn(
        self, x_i: jax.Array, x_j: jax.Array, edge_feats: jax.Array
    ) -> jax.Array:
        return (x_i + x_j) * self.edge_proj(edge_feats)

    def __call__(self, graph: Graph) -> Graph:
        """Applies the edge embedding to update edge features using node features.

        Args:
            graph: Input Graph containing node features, edge
                   features, and graph topology (senders, receivers).

        Returns:
            Updated Graph object with new edge features (field name: "embedding").
        """
        self._input_shape_assertions(graph)

        node_feats = graph.nodes.features["embedding_scalars"]
        senders_feats = node_feats[graph.senders]
        receivers_feats = node_feats[graph.receivers]

        edge_messages = self._message_fn(
            senders_feats, receivers_feats, graph.edges.features["embedding"]
        )

        return graph.update_edge_features(embedding=edge_messages)


class VisnetMultiHeadReadoutBlock(nn.Module):
    """Applies the final readout processing network to node and edge features.

    Attributes:
        num_heads: The number of readout heads.
        num_channels: The number of channels.
        activation: The activation function.
        vecnorm_type: The type of vector normalization to apply.
        l_max: Highest harmonic order included in the Spherical Harmonics series.
        predict_partial_charges: Whether to predict partial charges.
    """

    num_heads: int
    num_channels: int
    activation: str
    vecnorm_type: str
    l_max: int  # Required for input shape assertions.
    predict_partial_charges: bool

    def setup(self) -> None:
        """Initializes the output processing network."""
        num_output_features = 1 + self.predict_partial_charges
        self.output_networks = [
            [
                GatedEquivariantBlock(
                    num_channels=self.num_channels,
                    out_channels=self.num_channels // 2,
                    intermediate_channels=None,
                    activation=self.activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    num_channels=self.num_channels // 2,
                    out_channels=num_output_features,
                    intermediate_channels=None,
                    activation=self.activation,
                    scalar_activation=False,
                ),
            ]
            for _ in range(self.num_heads)
        ]
        self.layernorm = nn.LayerNorm(epsilon=LAYER_NORM_EPSILON)
        self.vec_layernorm = VecLayerNorm(
            num_channels=self.num_channels,
            norm_type=self.vecnorm_type,
            eps=VEC_LAYER_NORM_EPSILON,
        )

    def _input_shape_assertions(self, graph: Graph) -> None:
        assert graph.nodes.features["latent_scalars"].ndim == 2
        assert graph.nodes.features["latent_scalars"].shape[1] == self.num_channels
        assert graph.edges.features["latent"].ndim == 2
        assert graph.edges.features["latent"].shape[1] == self.num_channels
        assert graph.nodes.features["latent_vectors"].ndim == 3
        assert (
            graph.nodes.features["latent_vectors"].shape[1]
            == ((self.l_max + 1) ** 2) - 1
        )
        assert graph.nodes.features["latent_vectors"].shape[2] == self.num_channels

    def __call__(self, graph: Graph) -> Graph:
        """Applies the final output processing network to node and edge features.

        Passes the node features and edge vector features through a stack of
        GatedEquivariantBlock layers to produce final per-node outputs.

        Args:
            graph: Input Graph object containing scalar node features
                   ("latent_scalars") and vector node features ("latent_vectors").

        Returns:
            Updated Graph object with processed node features ("latent_scalars")
            of shape [num_nodes, num_heads, Nx0e].
        """
        self._input_shape_assertions(graph)

        graph = graph.update_node_features(
            latent_scalars=self.layernorm(graph.nodes.features["latent_scalars"]),
            latent_vectors=self.vec_layernorm(graph.nodes.features["latent_vectors"]),
        )

        node_outputs = []
        for head_idx in range(self.num_heads):
            node_outputs.append(
                self._compute_node_outputs_for_one_head(graph, head_idx)
            )

        node_outputs = jnp.stack(node_outputs, axis=1)  # [num_nodes, num_heads, Nx0e]
        return graph.update_node_features(outputs=node_outputs)

    def _compute_node_outputs_for_one_head(
        self, graph: Graph, head_idx: int
    ) -> jax.Array:
        node_feats = graph.nodes.features["latent_scalars"]
        vector_feats = graph.nodes.features["latent_vectors"]

        for layer in self.output_networks[head_idx]:
            node_feats, vector_feats = layer(node_feats, vector_feats)

        # vector_feats is discarded here; the last GatedEquivariantBlock's
        # vec2_proj therefore receive no gradient. The PyTorch reference
        # patched this with `+ jnp.sum(vector_feats) * 0` so every parameter
        # would "have a gradient".
        # In JAX that trick is a no-op — backward gives ∂L/∂v = ∂L/∂node_feats * 0 = 0
        # for the same params — and risks NaN if vector_feats ever contains
        # inf, so it is intentionally omitted.
        return node_feats


class GatedEquivariantBlock(nn.Module):
    """Applies the gated equivariant block to update node features using vector
    features.

    Attributes:
        num_channels: The number of channels.
        out_channels: The number of output channels.
        intermediate_channels: The number of intermediate channels.
        activation: The activation function.
        scalar_activation: Whether to apply the activation function to the scalar
                           output.
    """

    num_channels: int
    out_channels: int
    intermediate_channels: int
    activation: str
    scalar_activation: bool

    def setup(self) -> None:
        """Initializes the gated equivariant block."""
        if self.intermediate_channels is None:
            intermediate_channels = self.num_channels
        else:
            intermediate_channels = self.intermediate_channels

        self.vec1_proj = nn.Dense(
            self.num_channels,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
        )
        self.vec2_proj = nn.Dense(
            self.out_channels, use_bias=False, kernel_init=initializers.xavier_uniform()
        )

        self.update_net = MLP(
            layer_sizes=(
                2 * self.num_channels,
                intermediate_channels,
                self.out_channels * 2,
            ),
            activation=self.activation,
            kernel_init=initializers.xavier_uniform(),
            use_bias=True,
        )
        if self.scalar_activation:
            self.act = parse_activation(
                self.activation
            )  # Assuming direct call is intended, otherwise needs adjustment

    def __call__(self, x: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Applies the gated equivariant block to update node features using vector
        features.

        Runs the forward pass of the GatedEquivariantBlock module on an input node
        features and vector features and returns an updated node features and vector
        features.

        Args:
            x: The node features.
            v: The vector features.

        Returns:
            The updated node features and vector features.
        """
        vec1 = safe_norm(self.vec1_proj(v), axis=-2)
        vec2 = self.vec2_proj(v)
        x = jnp.concatenate([x, vec1], axis=-1)
        x = self.update_net(x)
        x, v = jnp.split(x, 2, axis=-1)
        v = jnp.expand_dims(v, axis=1) * vec2

        if self.scalar_activation:
            x = self.act(x)
        return x, v
