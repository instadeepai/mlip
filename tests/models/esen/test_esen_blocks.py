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

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp

from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.models.esen.blocks import EsenEmbeddingBlock
from mlip.models.esen.coefficient_mapping import CoefficientMapping


class TestEsenBlocks:
    sphere_channels = 4
    hidden_channels = 4
    l_max = 2
    m_max = 2
    num_species = 10
    num_charges = None
    num_rbf = 4
    edge_channels = 4
    radial_envelope = "polynomial_envelope"
    radial_basis = "gauss"
    trainable_rbf = False
    basis_width_scalar = 2.0
    cosine_cutoff = False
    mapping_reduced = CoefficientMapping(l_max=l_max, m_max=m_max)
    graph_cutoff_angstrom = 5.0
    norm_type = "rms_norm_sh"

    key = jax.random.PRNGKey(0)

    n_nodes = 10
    n_edges = 68

    @property
    def edge_channels_list(self) -> list[int]:
        return [
            self.num_rbf + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]

    def create_graph_from_input(
        cls,
        senders: jax.Array,
        receivers: jax.Array,
        edge_features_dict: dict[str, jax.Array] = {},
        node_features_dict: dict[str, jax.Array] = {},
    ) -> Graph:
        graph = Graph(
            nodes=GraphNodes(
                positions=None,
                features=node_features_dict,
            ),
            edges=GraphEdges(
                features=edge_features_dict,
            ),
            globals=GraphGlobals(
                cell=None,
                weight=None,
            ),
            senders=senders,
            receivers=receivers,
            n_node=None,
            n_edge=None,
        )
        return graph

    def common_input(self):
        graph_definition_kwargs = {}
        graph_definition_kwargs.update(
            minval=0, maxval=self.n_nodes, shape=(self.n_edges,), key=self.key
        )
        senders = jax.random.randint(**graph_definition_kwargs)
        receivers = jax.random.randint(**graph_definition_kwargs)
        return senders, receivers

    def embedding_block_input(self):
        node_features_dict = {}
        edge_features_dict = {}
        graph_definition_kwargs = {}
        graph_definition_kwargs.update(
            minval=0, maxval=self.n_nodes, shape=(self.n_edges,), key=self.key
        )
        senders = jax.random.randint(**graph_definition_kwargs)
        receivers = jax.random.randint(**graph_definition_kwargs)
        node_features_dict.update(species=jnp.zeros(self.n_nodes).astype(jnp.int32))
        edge_features_dict.update(vectors=jnp.ones((self.n_edges, 3)))
        graph = self.create_graph_from_input(
            senders, receivers, edge_features_dict, node_features_dict
        )
        return graph

    def test_esen_embedding_block(self):
        graph_in = self.embedding_block_input()
        block = EsenEmbeddingBlock(
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            l_max=self.l_max,
            num_species=self.num_species,
            num_charges=self.num_charges,
            sphere_channels=self.sphere_channels,
            radial_envelope=self.radial_envelope,
            radial_basis=self.radial_basis,
            basis_width_scalar=self.basis_width_scalar,
            cosine_cutoff=self.cosine_cutoff,
            trainable_rbf=self.trainable_rbf,
            edge_channels=self.edge_channels,
            m_max=self.m_max,
            mapping_reduced=self.mapping_reduced,
            edge_channels_list=self.edge_channels_list,
            num_rbf=self.num_rbf,
        )
        params = block.init(self.key, graph_in)
        graph = jax.jit(block.apply)(params, graph_in)
        assert graph.nodes.features["embedding"].shape == (
            self.n_nodes,
            ((self.l_max + 1) ** 2),
            self.sphere_channels,
        )
        assert graph.edges.features["embedding"].shape == (
            self.n_edges,
            self.edge_channels + 2 * self.sphere_channels,
        )
        assert graph.edges.features["envelope"].shape == (self.n_edges, 1, 1)
        assert graph.edges.features["wigner_and_m_mapping"].shape == (
            self.n_edges,
            ((self.l_max + 1) ** 2),
            ((self.l_max + 1) ** 2),
        )
        assert graph.edges.features["wigner_and_m_mapping_inv"].shape == (
            self.n_edges,
            ((self.l_max + 1) ** 2),
            ((self.l_max + 1) ** 2),
        )

    def test_esen_embedding_block_rot_equivariance(self):
        graph_in = self.embedding_block_input()
        block = EsenEmbeddingBlock(
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            l_max=self.l_max,
            num_species=self.num_species,
            num_charges=self.num_charges,
            sphere_channels=self.sphere_channels,
            radial_envelope=self.radial_envelope,
            radial_basis=self.radial_basis,
            basis_width_scalar=self.basis_width_scalar,
            cosine_cutoff=self.cosine_cutoff,
            trainable_rbf=self.trainable_rbf,
            edge_channels=self.edge_channels,
            m_max=self.m_max,
            mapping_reduced=self.mapping_reduced,
            edge_channels_list=self.edge_channels_list,
            num_rbf=self.num_rbf,
        )
        params = block.init(self.key, graph_in)
        apply = jax.jit(block.apply)
        graph_out = apply(params, graph_in)

        # apply random rotation to the vectors features
        rotation_matrix = e3nn.rand_matrix(self.key)
        vector_features_rot = graph_in.edges.features["vectors"] @ rotation_matrix
        graph_in_rot = graph_in.update_edge_features(vectors=vector_features_rot)
        graph_out_rot = apply(params, graph_in_rot)

        # Compare embedding features (invariant to rotation)
        assert jnp.allclose(
            graph_out.edges.features["embedding"],
            graph_out_rot.edges.features["embedding"],
            atol=1e-6,
        )
        assert jnp.allclose(
            graph_out.edges.features["envelope"],
            graph_out_rot.edges.features["envelope"],
            atol=1e-6,
        )
        # rotate outputted node features :
        rot_irreps = e3nn.Irreps.spherical_harmonics(self.l_max)
        rot_matrix_irreps = rot_irreps.D_from_matrix(rotation_matrix)
        node_feats_rot = jnp.einsum(
            "ijk, jl -> ilk", graph_out.nodes.features["embedding"], rot_matrix_irreps
        )
        # Ensure node feats are equivariant to rotation
        assert jnp.allclose(
            node_feats_rot,
            graph_out_rot.nodes.features["embedding"],
            atol=1e-6,
        )
