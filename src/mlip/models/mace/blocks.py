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

import functools
from typing import Callable, Literal

import e3j
import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
from e3j.utils.options import Layout

from mlip.graph import Graph
from mlip.models import options
from mlip.models.blocks import (
    JointNodeEmbeddingBlock,
    NodeEmbeddingBlock,
    RadialEmbeddingBlock,
    SphericalHarmonicsBlock,
)
from mlip.models.gaunt_tensor_product import GauntSymmetricContraction
from mlip.models.mace.symmetric_contraction import (
    SymmetricContraction,
    SymmetricContractionE3NN,
)
from mlip.models.options import GradientScaledKernelInit, RadialBasis
from mlip.utils.safe_norm import safe_norm


class MaceEmbeddingBlock(nn.Module):
    """Initial embedding block for MACE.

    This module will assign the following graph features:

    * `nodes.features['embedding']`: initial embedding of atomic numbers
    * `edges.features['radial_embedding']`: RBF-encoding of interatomic distances.
    * `edges.features['spherical_embedding']`: harmonic embedding of unit edge
      vectors.
    """

    num_species: int
    num_charges: int | None
    num_channels: int
    l_max: int
    r_max: float
    num_rbf: int
    radial_envelope: options.RadialEnvelope
    avg_r_min: float | None = None
    activation_fn: Callable = jax.nn.silu

    def setup(self):
        if self.num_charges is not None:
            self.node_embedding_block = JointNodeEmbeddingBlock(
                num_species=self.num_species,
                num_charge=self.num_charges,
                num_channels=self.num_channels,
                kernel_init=GradientScaledKernelInit.FAN_IN_NORMAL,
                activation_fn=self.activation_fn,
            )
        else:
            self.node_embedding_block = NodeEmbeddingBlock(
                num_species=self.num_species,
                num_channels=self.num_channels,
                kernel_init=GradientScaledKernelInit.FAN_IN_NORMAL,
            )
        self.radial_embedding_block = RadialEmbeddingBlock(
            radial_basis=RadialBasis.BESSEL,
            num_rbf=self.num_rbf,
            graph_cutoff_angstrom=self.r_max,
            learnable=False,
            radial_envelope=self.radial_envelope,
            avg_r_min=self.avg_r_min,
            return_as_irreps=True,
        )
        self.spherical_embedding_block = SphericalHarmonicsBlock(
            l_max=self.l_max,
            normalize=True,
        )

    def __call__(self, graph: Graph) -> Graph:
        node_species = graph.nodes.features.get("species")
        if node_species is None:
            raise KeyError("Node feature 'species' has to be assigned upstream.")

        # Node features
        if self.num_charges is not None:
            charge_indices = graph.nodes.features["charge_indices"]
            node_embedding = self.node_embedding_block(node_species, charge_indices)
        else:
            node_embedding = self.node_embedding_block(node_species)
        node_embedding = e3nn.IrrepsArray(f"{self.num_channels}x0e", node_embedding)
        species_one_hot = jnp.eye(self.num_species)[node_species]

        # Geometric edge features
        edge_vectors = graph.edge_vectors()
        radial_embedding = self.radial_embedding_block(safe_norm(edge_vectors, axis=-1))
        spherical_embedding = self.spherical_embedding_block(edge_vectors)

        # Assign graph features
        graph = graph.update_node_features(
            embedding=node_embedding,
            species_one_hot=species_one_hot,
        )
        graph = graph.update_edge_features(
            radial_embedding=radial_embedding,
            spherical_embedding=spherical_embedding,
        )
        return graph


class EquivariantProductBasisBlock(nn.Module):
    source_irreps: e3nn.Irreps
    target_irreps: e3nn.Irreps
    correlation: int
    num_species: int
    num_channels: int
    gate_nodes: bool = False
    layout: Layout | str = Layout.TRAILING_CHANNELS
    symmetric_contraction_backend: Literal[
        "e3j", "e3nn", "e3nn_symmetric", "gaunt_tp"
    ] = "e3j"

    def node_gating(
        self, node_feats: e3nn.IrrepsArray, node_species: jnp.ndarray
    ) -> e3nn.IrrepsArray:
        node_scalars = node_feats.filter(e3nn.Irreps("0e")).array
        w = self.param(
            "species_wise_gate_weights",
            nn.initializers.normal(stddev=1 / jnp.sqrt(node_scalars.shape[-1])),
            (
                self.num_species,
                node_scalars.shape[-1],
                node_feats.irreps.num_irreps,
            ),
        )[node_species]
        b = self.param(
            "species_wise_gate_bias",
            nn.initializers.normal(),
            (self.num_species, node_feats.irreps.num_irreps),
        )[node_species]
        node_feats = node_feats * (jax.vmap(jnp.matmul)(node_scalars, w) + b)
        return node_feats

    def setup(self):
        target_irreps = e3nn.Irreps(self.target_irreps)
        source_irreps = e3nn.Irreps([(1, ir) for _, ir in self.source_irreps])
        filter_irreps = e3nn.Irreps([(1, ir) for _, ir in target_irreps])

        if self.symmetric_contraction_backend in ["e3nn", "e3nn_symmetric"]:
            # E3NN fallback with optional `symmetric_tensor_product_basis` argument:
            # affects numerical outputs and runtime. Consider for removal.
            symmetric_basis = self.symmetric_contraction_backend == "e3nn_symmetric"
            SymContraction = functools.partial(
                SymmetricContractionE3NN,
                symmetric_tensor_product_basis=symmetric_basis,
            )
        elif self.symmetric_contraction_backend == "gaunt_tp":
            SymContraction = GauntSymmetricContraction
        else:
            # Use e3j-based power expansion and linear projection.
            SymContraction = functools.partial(
                SymmetricContraction,
                layout=Layout.parse(self.layout),
            )

        self.symmetric_contraction = SymContraction(
            source_irreps=str(source_irreps),
            keep_irrep_out=str(filter_irreps),
            correlation=self.correlation,
            num_species=self.num_species,
            num_channels=self.num_channels,
        )

        self.linear_block = e3j.linen.Linear(
            source_irreps=str(self.num_channels * filter_irreps),
            target_irreps=str(target_irreps),
            layout="E3NN",
            kernel_init="FAN_IN",
            rescale_gradients=True,
        )

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        """Compute higher-order features by equivariant power expansion.

        Args:
            graph: Input graph with node features "latent" and "species".

        Raises:
            KeyError if "latent" or "species" are not found in node features.
        """
        node_feats = graph.nodes.features["latent"]
        assert node_feats.irreps == e3nn.Irreps(self.source_irreps)
        node_feats = node_feats.mul_to_axis().remove_zero_chunks()

        node_species = graph.nodes.features["species"]

        node_feats = self.symmetric_contraction(node_feats, node_species)
        node_feats = node_feats.axis_to_mul()

        if self.gate_nodes:
            node_feats = self.node_gating(node_feats, node_species)

        node_feats = e3nn.IrrepsArray(
            self.target_irreps,
            self.linear_block(node_feats.array),
        )

        return graph.update_node_features(latent=node_feats)
