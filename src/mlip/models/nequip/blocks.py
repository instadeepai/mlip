# Copyright (c) 2021 The President and Fellows of Harvard College
# Copyright (c) 2025 The NequIP Developers
#
# Licensed under the MIT License (https://opensource.org/licenses.MIT)
#
# Copyright (c) 2026 InstaDeep Ltd
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

from typing import Callable

import e3nn_jax as e3nn
import flax.linen as nn
import jax.numpy as jnp

from mlip.graph import Graph
from mlip.models.blocks import (
    JointNodeEmbeddingBlock,
    NodeEmbeddingBlock,
    RadialEmbeddingBlock,
    SphericalHarmonicsBlock,
)
from mlip.models.options import GradientScaledKernelInit, RadialBasis, RadialEnvelope
from mlip.models.readout import MultiHeadReadoutBlock
from mlip.utils.safe_norm import safe_norm


class NequipEmbeddingBlock(nn.Module):
    """Initial embedding block for the NequIP model.

    Encodes atomic species as one-hot vectors and projects them to the hidden
    node feature space. Encodes edge geometry into radial basis functions and
    spherical harmonics, which together form the edge embedding used in the
    interaction layers.

    Attributes:
        num_species: Number of distinct atomic species in the dataset.
        target_irreps: Target irreps for the initial node feature projection.
        l_max: Maximum degree of spherical harmonics for edge angular encoding.
        num_rbf: Number of Bessel radial basis functions.
        r_max: Cutoff distance in Angstroms.
        avg_r_min: Average minimum interatomic distance, used to shift the
            radial basis. If None, no shift is applied.
        radial_envelope: Envelope function to smoothly decay the basis at
            `r_max` (e.g. polynomial or soft envelope).
    """

    num_species: int
    num_charges: int | None
    target_irreps: str
    l_max: int
    num_rbf: int
    r_max: float
    avg_r_min: float | None
    radial_envelope: RadialEnvelope
    activation_fn: Callable | None

    def _input_shape_assertions(self, graph: Graph) -> None:
        node_species = graph.nodes.features["species"]
        edge_vectors = graph.edge_vectors()

        assert node_species.ndim == 1

        assert edge_vectors.ndim == 2
        assert edge_vectors.shape[1] == 3

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        """Apply the NequIP embedding block to compute initial node and edge embeddings.

        Args:
            graph: Graph containing node features with "species" and edge vectors.

        Returns:
            Updated graph containing node features ("species_one_hot", "embedding"),
            and edge features ("radial_embedding", "spherical_embedding").
        """
        self._input_shape_assertions(graph)

        edge_vectors = graph.edge_vectors()
        node_species = graph.nodes.features["species"]

        # One-hot encode species
        species_one_hot = e3nn.IrrepsArray(
            f"{self.num_species}x0e", jnp.eye(self.num_species)[node_species]
        )

        # Scalar species embedding only (0e).
        scalar_dim = e3nn.Irreps(self.target_irreps).filter("0e").dim
        if self.num_charges is not None:
            charge_indices = graph.nodes.features["charge_indices"]
            embed = JointNodeEmbeddingBlock(
                num_species=self.num_species,
                num_charge=self.num_charges,
                num_channels=scalar_dim,
                kernel_init=GradientScaledKernelInit.FAN_IN_NORMAL,
                activation_fn=self.activation_fn,
            )(node_species, charge_indices)
        else:
            embed = NodeEmbeddingBlock(
                num_species=self.num_species,
                num_channels=scalar_dim,
                kernel_init=GradientScaledKernelInit.FAN_IN_NORMAL,
            )(node_species)
        node_feats = e3nn.IrrepsArray(f"{scalar_dim}x0e", embed)

        graph = graph.update_node_features(
            species_one_hot=species_one_hot, embedding=node_feats
        )

        # graph.edge_vectors() may return an IrrepsArray; unwrap to plain array.
        if hasattr(edge_vectors, "irreps"):
            edge_vectors = edge_vectors.array
        edge_lengths = safe_norm(edge_vectors, axis=-1)

        radial_embedding = RadialEmbeddingBlock(
            radial_basis=RadialBasis.BESSEL,
            num_rbf=self.num_rbf,
            graph_cutoff_angstrom=self.r_max,
            learnable=False,
            radial_envelope=self.radial_envelope,
            avg_r_min=self.avg_r_min,
            return_as_irreps=True,
        )(edge_lengths)

        spherical_embedding = SphericalHarmonicsBlock(
            l_max=self.l_max,
            normalize=True,
        )(edge_vectors)

        return graph.update_edge_features(
            radial_embedding=radial_embedding,
            spherical_embedding=spherical_embedding,
        )


class NequipMultiHeadReadoutBlock(nn.Module):
    """Readout block that maps final node features to per-atom energies.

    Does input shape assertions, then prepares the correct feature sizes
    for the linear layers and then calls the `ReadoutBlock` with it.

    Extracts the scalar (0e) channels from the final node feature irreps and
    applies two successive linear projections: first halving the scalar
    multiplicity, then reducing to a single scalar per atom. The result is
    stored as "outputs" in the node features.

    Attributes:
        source_node_irreps: Expected irreps of the input node features, used to
            validate the graph at call time.
    """

    source_node_irreps: e3nn.Irreps  # Required for _input_shape_assertions.
    num_heads: int
    predict_partial_charges: bool

    def _input_shape_assertions(self, graph: Graph) -> None:
        node_feats = graph.nodes.features["latent"]

        assert node_feats.ndim == 2
        assert node_feats.irreps == e3nn.Irreps(self.source_node_irreps)

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        self._input_shape_assertions(graph)

        node_feats = graph.nodes.features["latent"]
        scalar_mul = next(
            (mul for mul, ir in node_feats.irreps if ir == e3nn.Irrep("0e")),
            None,
        )
        if scalar_mul is None:
            raise ValueError(
                "NequipMultiHeadReadoutBlock requires at least one 0e irrep in "
                f"node_feats, but got: {node_feats.irreps}"
            )

        second_to_final_irreps = e3nn.Irreps(f"{scalar_mul // 2}x0e")
        final_irreps = e3nn.Irreps("1x0e")
        if self.predict_partial_charges:
            final_irreps = (final_irreps + e3nn.Irreps("1x0e")).simplify()

        graph = MultiHeadReadoutBlock(
            num_heads=self.num_heads,
            features=(second_to_final_irreps, final_irreps),
            activation=None,
            use_equiv=True,
        )(graph)

        return graph
