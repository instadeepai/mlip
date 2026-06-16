# Code converted from https://github.com/facebookresearch/fairchem
# Some parts of the code may remain identical, distributed under:
#
#     MIT License- Copyright (c) Meta Platforms, Inc. and affiliates.
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

from pathlib import Path
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from mlip.graph import Graph
from mlip.models.blocks import (
    JointNodeEmbeddingBlock,
    NodeEmbeddingBlock,
    RadialEmbeddingBlock,
)
from mlip.models.esen.coefficient_mapping import CoefficientMapping
from mlip.models.esen.esen_helpers import NODE_OFFSET, EdgeDegreeEmbedding
from mlip.models.esen.eulers import eulers_to_wigner, init_edge_rot_euler_angles
from mlip.models.esen.quaternion.wigner_hybrid import axis_angle_wigner_hybrid
from mlip.models.options import RadialBasis, RadialEnvelope, parse_radial_envelope
from mlip.utils.safe_norm import safe_norm

###############################
# Edgewise and Edgewise utils #
###############################

RESCALE_FACTOR = 5.0
BASE_DIR = Path(__file__).parent


class EsenEmbeddingBlock(nn.Module):
    """Embeds input node and edge features for the Esen model.

    Initializes and applies the embedding layers for node species, radial
    functions, and spherical harmonics. Updates the input Graph with the embedded
    features and pre-processes edges and neighbors for subsequent network layers.

    Attributes:
        graph_cutoff_angstrom: The cutoff radius for the graph.
        l_max: Highest harmonic order included in the Spherical Harmonics series.
        num_species: The number of elements (atomic species descriptors) allowed.
        sphere_channels: The number of channels for the node embedding.
        radial_envelope: The radial envelope function.
        radial_basis: The type of radial basis function used.
        num_rbf: Number of radial basis functions used in the embedding block.
        trainable_rbf: Whether to add learnable weights to each of the radial embedding
                       basis functions.
        edge_channels: The number of channels for the edge embedding.
        m_max: The maximum order of the spherical harmonics to include in the embedding.
        mapping_reduced: The mapping of the spherical harmonics to the reduced set of
                        coefficients.
        edge_channels_list: The list of channels for the edge embedding.
    """

    graph_cutoff_angstrom: float
    l_max: int
    num_species: int
    num_charges: int | None
    sphere_channels: int
    radial_envelope: RadialEnvelope | str
    radial_basis: RadialBasis | str
    num_rbf: int
    basis_width_scalar: float
    cosine_cutoff: bool
    trainable_rbf: bool
    edge_channels: int
    m_max: int
    mapping_reduced: CoefficientMapping
    edge_channels_list: list[int]
    activation_fn: Callable = jax.nn.silu
    deterministic_scatter_ops: bool = False
    use_quaternion: bool = True

    def setup(self) -> None:
        """Initializes the embedding layers for node species, radial functions,"""

        # NOTE: In original torch code they use uniform distribution for these inits.
        self.sph_feature_size = int((self.l_max + 1) ** 2)
        if self.num_charges is not None:
            self.node_embedding = JointNodeEmbeddingBlock(
                num_species=self.num_species,
                num_charge=self.num_charges,
                num_channels=self.sphere_channels,
                activation_fn=self.activation_fn,
            )
        else:
            self.node_embedding = NodeEmbeddingBlock(
                num_species=self.num_species,
                num_channels=self.sphere_channels,
            )

        self.envelope = parse_radial_envelope(self.radial_envelope)

        self.radial_embedding = RadialEmbeddingBlock(
            radial_basis=RadialBasis(self.radial_basis),
            num_rbf=self.num_rbf,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            learnable=self.trainable_rbf,
            radial_envelope=RadialEnvelope.COSINE_CUTOFF
            if self.cosine_cutoff
            else None,
            basis_width_scalar=self.basis_width_scalar,
        )

        self.source_embedding = NodeEmbeddingBlock(
            num_species=self.num_species,
            num_channels=self.edge_channels,
            kernel_init=nn.initializers.lecun_normal(),
        )
        self.target_embedding = NodeEmbeddingBlock(
            num_species=self.num_species,
            num_channels=self.edge_channels,
            kernel_init=nn.initializers.lecun_normal(),
        )

        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            l_max=self.l_max,
            m_max=self.m_max,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=RESCALE_FACTOR,
            mapping_reduced=self.mapping_reduced,
            deterministic_scatter_ops=self.deterministic_scatter_ops,
        )

        self.coefficient_index = self.mapping_reduced.coefficient_idx(
            self.l_max, self.m_max
        )

    def _input_shape_assertions(self, graph: Graph) -> None:
        assert graph.nodes.features["species"].ndim == 1
        assert graph.edges.features["vectors"].ndim == 2
        assert graph.edges.features["vectors"].shape[1] == 3
        assert (
            graph.edges.features["vectors"].shape[0]
            == graph.receivers.shape[0]
            == graph.senders.shape[0]
        )

    def __call__(self, graph: Graph) -> Graph:
        """Applies the Esen embedding block to compute initial node and edge embeddings.

        Embedded features in the final graph can be accessed as:
        - node features: graph.nodes.features["embedding"]
        - edge features: graph.edges.features["embedding"]
        - edge envelope: graph.edges.features["envelope"]
        - wigner and m mapping: graph.nodes.features["wigner_and_m_mapping"]

        Args:
            graph: Graph containing node features with "species" and edge vectors.

        Returns:
            Updated Graph with embedded node and edge features ready for Esen
            processing.
        """
        self._input_shape_assertions(graph)

        senders, receivers = graph.senders, graph.receivers
        node_species = graph.nodes.features["species"]
        edge_vectors = graph.edges.features["vectors"]

        edge_index = (senders, receivers)
        node_feats = jnp.zeros(
            (node_species.shape[0], self.sph_feature_size, self.sphere_channels),
        )

        if self.num_charges is not None:
            charge_indices = graph.nodes.features["charge_indices"]
            scalar_node_feats = self.node_embedding(node_species, charge_indices)
        else:
            scalar_node_feats = self.node_embedding(node_species)
        node_feats = node_feats.at[:, 0, :].set(scalar_node_feats)

        # Edge embedding
        distances = safe_norm(edge_vectors, axis=-1)
        edge_envelope = self.envelope(distances, self.graph_cutoff_angstrom).reshape(
            -1, 1, 1
        )

        wigner_and_m_mapping = self._get_rotmat_and_wigner(
            edge_vectors,  # In Torch: graph_dict["edge_distance_vec"]
        )

        edge_distance_embedding = self.radial_embedding(distances)
        source_node_embedding = self.source_embedding(node_species[senders])
        target_node_embedding = self.target_embedding(node_species[receivers])

        edge_feats = jnp.concatenate(
            (edge_distance_embedding, source_node_embedding, target_node_embedding),
            axis=1,
        )

        # Edge degree embedding
        node_feats = self.edge_degree_embedding(
            node_feats,
            edge_feats,
            edge_index,
            wigner_and_m_mapping,
            edge_envelope,
            NODE_OFFSET,
        )

        graph = graph.update_node_features(
            embedding=node_feats,  # [num_nodes, (l_max + 1) ** 2), sphere_channels]
        )
        graph = graph.update_edge_features(
            embedding=edge_feats,
            envelope=edge_envelope,
            wigner_and_m_mapping=wigner_and_m_mapping,
        )

        return graph

    def _get_rotmat_and_wigner(
        self, edge_distance_vecs: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """
        Args:
          edge_distance_vecs: [E, 3] (or compatible) edge direction vectors

        Returns:
          wigner_and_m_mapping:      [E, M, K']
                                     where M = rows of to_m, K' == K or reduced
        """
        jd_file_path = BASE_DIR / "Jd.npz"
        jd_data = np.load(jd_file_path, allow_pickle=True)
        jd_list = list(jd_data["Jd"])
        with jax.ensure_compile_time_eval():
            jd_buffers = [
                jnp.asarray(jd_list[l_number], dtype=edge_distance_vecs.dtype)
                for l_number in range(self.l_max + 1)
            ]

        if self.use_quaternion:
            wigner = axis_angle_wigner_hybrid(
                edge_distance_vecs,
                self.l_max,
                jd_buffers,
                key=jax.random.PRNGKey(42),
            )
        else:
            # Euler angles -> Wigner-D
            # (full basis, block-diagonal in l, but stored dense)
            euler_angles = init_edge_rot_euler_angles(
                edge_distance_vecs, key=jax.random.PRNGKey(42)
            )
            wigner = eulers_to_wigner(
                eulers=euler_angles, start_l_max=0, end_l_max=self.l_max, jd=jd_buffers
            )

        # select the subset of coefficients used when m_max < l_max
        if self.m_max != self.l_max:
            wigner = wigner[:, self.coefficient_index, :]

        # M-mapping reindex matrix (to_m) on the m/“row” side.
        to_m = jnp.asarray(self.mapping_reduced.to_m, dtype=wigner.dtype)
        wigner_and_m_mapping = jnp.einsum("mk,nkj->nmj", to_m, wigner)

        return wigner_and_m_mapping
