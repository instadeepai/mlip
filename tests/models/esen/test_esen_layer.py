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
from mlip.models.esen.layer import ESENLayer


class TestESENLayer:
    sphere_channels = 4
    hidden_channels = 4
    l_max = 2
    m_max = 2
    mapping_reduced = CoefficientMapping(l_max=l_max, m_max=m_max)
    graph_cutoff_angstrom = 5.0
    norm_type = "rms_norm_sh"
    act_type = "gate"
    ff_type = "spectral"

    # embedding block exclusive args:
    num_species = 10
    num_charges = None
    num_rbf = 4
    edge_channels = 4
    radial_envelope = "polynomial"
    radial_basis = "gauss"
    trainable_rbf = False
    basis_width_scalar = 2.0
    cosine_cutoff = False

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

    def module_v2(self) -> ESENLayer:
        return ESENLayer(
            sphere_channels=self.sphere_channels,
            hidden_channels=self.hidden_channels,
            l_max=self.l_max,
            m_max=self.m_max,
            mapping_reduced=self.mapping_reduced,
            edge_channels_list=self.edge_channels_list,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            norm_type=self.norm_type,
            act_type=self.act_type,
        )

    def embedding_block(self) -> EsenEmbeddingBlock:
        # required to create proper wigner matrices as layer input
        return EsenEmbeddingBlock(
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            l_max=self.l_max,
            num_species=self.num_species,
            num_charges=self.num_charges,
            sphere_channels=self.sphere_channels,
            radial_envelope=self.radial_envelope,
            radial_basis=self.radial_basis,
            num_rbf=self.num_rbf,
            basis_width_scalar=self.basis_width_scalar,
            cosine_cutoff=self.cosine_cutoff,
            trainable_rbf=self.trainable_rbf,
            edge_channels=self.edge_channels,
            m_max=self.m_max,
            mapping_reduced=self.mapping_reduced,
            edge_channels_list=self.edge_channels_list,
        )

    def wigner_and_m_mapping(
        self, edge_vectors: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        wigner_and_m_mapping = self.embedding_block()._get_rotmat_and_wigner(
            edge_vectors
        )
        return wigner_and_m_mapping

    def input_dict(self) -> dict[str, jax.Array]:
        n_nodes = 10
        n_edges = 68
        node_feats = jnp.ones(
            shape=(n_nodes, ((self.l_max + 1) ** 2), self.sphere_channels)
        )
        edge_feats = jnp.ones(shape=(n_edges, self.sphere_channels))
        edge_envelope = jnp.ones(shape=(n_edges, 1, 1))
        edge_vectors = jnp.ones(shape=(n_edges, 3))
        wigner_and_m_mapping = self.wigner_and_m_mapping(edge_vectors)
        graph_definition_kwargs = {}
        graph_definition_kwargs.update(
            minval=0, maxval=n_nodes, shape=(n_edges,), key=self.key
        )
        senders = jax.random.randint(**graph_definition_kwargs)
        receivers = jax.random.randint(**graph_definition_kwargs)

        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "latent_node": node_feats,
            "latent_edge": edge_feats,
            "vectors": edge_vectors,
            "wigner_and_m_mapping": wigner_and_m_mapping,
            "envelope": edge_envelope,
            "senders": senders,
            "receivers": receivers,
        }

    def input_v2(self) -> tuple[Graph, ...]:
        full_input_dict = self.input_dict().copy()
        graph = Graph(
            nodes=GraphNodes(positions=None, features={}),
            edges=GraphEdges(features={}),
            globals=GraphGlobals(
                cell=None,
                weight=None,
            ),
            senders=full_input_dict.pop("senders"),
            receivers=full_input_dict.pop("receivers"),
            n_node=full_input_dict.pop("n_nodes"),
            n_edge=full_input_dict.pop("n_edges"),
        )
        node_inputs = {
            "latent": full_input_dict.pop("latent_node"),
        }
        graph = graph.update_node_features(**node_inputs)
        full_input_dict["latent"] = full_input_dict["latent_edge"]
        graph = graph.update_edge_features(**full_input_dict)
        return (graph,)

    def test_esen_layer_rot_equivariance(self):
        (graph,) = self.input_v2()
        module = self.module_v2()
        params = module.init(self.key, graph)
        apply = jax.jit(module.apply)

        rotation = e3nn.rand_matrix(self.key)

        # g(f(x)): run on original input, then rotate output
        out_feats = apply(params, graph).nodes.features["latent"]

        irreps_rot = e3nn.Irreps.spherical_harmonics(self.l_max)
        rot_matrix_irreps = irreps_rot.D_from_matrix(rotation)
        out_node_feats_rot = jnp.einsum("ijk, jl -> ilk", out_feats, rot_matrix_irreps)

        # f(g(x)): rotate equivariant inputs, then run
        egde_vectors_rot = graph.edges.features["vectors"] @ rotation
        wigner_and_m_mapping_rot = self.wigner_and_m_mapping(egde_vectors_rot)
        node_feats_rot = jnp.einsum(
            "ijk, jl -> ilk", graph.nodes.features["latent"], rot_matrix_irreps
        )
        rotated_graph = graph.update_node_features(
            latent=node_feats_rot,
        )
        rotated_graph = rotated_graph.update_edge_features(
            wigner_and_m_mapping=wigner_and_m_mapping_rot,
        )

        # f(g(x)): rotate equivariant inputs, then run
        rotated_graph_out = apply(params, rotated_graph)

        assert jnp.allclose(
            out_node_feats_rot,
            rotated_graph_out.nodes.features["latent"],
            atol=1e-6,
        )
