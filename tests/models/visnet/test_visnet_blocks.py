import e3nn_jax as e3nn
import jax
import jax.numpy as jnp

from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.models.blocks import SphericalHarmonicsBlock
from mlip.models.visnet.blocks import (
    VisnetEdgeEmbeddingBlock,
    VisnetEmbeddingBlock,
    VisnetMultiHeadReadoutBlock,
    VisnetNeighborEmbeddingBlock,
)
from mlip.models_v1.visnet.blocks import Sphere


class TestVisnetBlocks:
    n_nodes = 10
    n_edges = 68
    n_channels = 8
    n_rbf = 4
    l_max = 2
    vecnorm_type = "none"
    trainable_rbf = False
    radial_basis = "expnorm"
    graph_cutoff_angstrom = 5.0
    num_species = 10
    num_charges = None
    key = jax.random.PRNGKey(0)
    activation = "silu"

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
        senders, receivers = self.common_input()
        node_features_dict.update(species=jnp.zeros(self.n_nodes).astype(jnp.int32))
        edge_features_dict.update(vectors=jnp.ones((self.n_edges, 3)))
        graph = self.create_graph_from_input(
            senders, receivers, edge_features_dict, node_features_dict
        )
        return graph

    def neighbor_embedding_block_input(self):
        node_features_dict = {}
        edge_features_dict = {}
        senders, receivers = self.common_input()
        node_features_dict.update(
            embedding_scalars=jnp.ones((self.n_nodes, self.n_channels)),
            species=jnp.zeros(self.n_nodes).astype(jnp.int32),
        )
        edge_features_dict.update(
            embedding=jnp.ones((self.n_edges, self.n_rbf)),
            distances=jnp.ones((self.n_edges,)),
        )
        graph = self.create_graph_from_input(
            senders, receivers, edge_features_dict, node_features_dict
        )
        return graph

    def edge_embedding_block_input(self):
        node_features_dict = {}
        edge_features_dict = {}
        senders, receivers = self.common_input()
        node_features_dict.update(
            embedding_scalars=jnp.ones((self.n_nodes, self.n_channels)),
        )
        edge_features_dict.update(
            embedding=jnp.ones((self.n_edges, self.n_rbf)),
        )
        graph = self.create_graph_from_input(
            senders, receivers, edge_features_dict, node_features_dict
        )
        return graph

    def readout_block_input(self):
        node_features_dict = {}
        edge_features_dict = {}
        senders, receivers = self.common_input()
        node_features_dict.update(
            latent_scalars=jnp.ones((self.n_nodes, self.n_channels)),
            latent_vectors=jnp.ones((
                self.n_nodes,
                (self.l_max + 1) ** 2 - 1,
                self.n_channels,
            )),
        )
        edge_features_dict.update(
            latent=jnp.ones((self.n_edges, self.n_channels)),
        )
        graph = self.create_graph_from_input(
            senders, receivers, edge_features_dict, node_features_dict
        )
        return graph

    def test_visnet_embedding_block(self):
        graph_in = self.embedding_block_input()
        block = VisnetEmbeddingBlock(
            l_max=self.l_max,
            num_channels=self.n_channels,
            num_rbf=self.n_rbf,
            radial_basis=self.radial_basis,
            trainable_rbf=self.trainable_rbf,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            num_species=self.num_species,
            num_charges=self.num_charges,
            activation_fn=self.activation,
        )
        params = block.init(self.key, graph_in)
        graph = block.apply(params, graph_in)

        # Output shape assertions
        assert graph.nodes.features["embedding_scalars"].shape == (
            self.n_nodes,
            self.n_channels,
        )
        assert graph.edges.features["embedding"].shape == (
            self.n_edges,
            self.n_channels,
        )
        assert graph.nodes.features["embedding_vectors"].shape == (
            self.n_nodes,
            (self.l_max + 1) ** 2 - 1,
            self.n_channels,
        )
        assert graph.edges.features["distances"].shape == (self.n_edges,)
        assert graph.edges.features["spherical_embedding"].shape == (
            self.n_edges,
            self.n_channels,
        )

    def test_visnet_embedding_block_rot_equivariance(self):
        graph_in = self.embedding_block_input()
        vector_features = graph_in.edges.features["vectors"]
        block = VisnetEmbeddingBlock(
            l_max=self.l_max,
            num_channels=self.n_channels,
            num_rbf=self.n_rbf,
            radial_basis=self.radial_basis,
            trainable_rbf=self.trainable_rbf,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            num_species=self.num_species,
            num_charges=self.num_charges,
            activation_fn=self.activation,
        )
        params = block.init(self.key, graph_in)
        apply = jax.jit(block.apply)
        graph_out = apply(params, graph_in)

        # apply random rotation to the vectors features
        rotation_matrix = e3nn.rand_matrix(self.key)
        vector_features_rot = vector_features @ rotation_matrix
        graph_in_rot = graph_in.update_edge_features(vectors=vector_features_rot)
        graph_out_rot = apply(params, graph_in_rot)

        # Compare embedding features (invariant to rotation)
        assert jnp.allclose(
            graph_out.edges.features["embedding"],
            graph_out_rot.edges.features["embedding"],
            atol=1e-6,
        )
        assert jnp.allclose(
            graph_out.nodes.features["embedding_scalars"],
            graph_out_rot.nodes.features["embedding_scalars"],
            atol=1e-6,
        )
        assert jnp.allclose(
            graph_out.edges.features["distances"],
            graph_out_rot.edges.features["distances"],
            atol=1e-6,
        )
        assert jnp.allclose(
            graph_out.nodes.features["embedding_vectors"],
            graph_out_rot.nodes.features["embedding_vectors"],
            atol=1e-6,
        )

        # Compare spherical features (equivariant to rotation)
        spherical_features = graph_out.edges.features["spherical_embedding"]
        # Convert rotation matrix to irreps representation:
        irreps = e3nn.Irreps.spherical_harmonics(self.l_max)
        rotation_matrix_irreps = irreps[1:].D_from_matrix(rotation_matrix)
        spherical_features_irreps_rot = spherical_features @ rotation_matrix_irreps
        assert jnp.allclose(
            spherical_features_irreps_rot,
            graph_out_rot.edges.features["spherical_embedding"],
            atol=1e-6,
        )

    def test_visnet_neighbor_embedding_block(self):
        graph_in = self.neighbor_embedding_block_input()
        block = VisnetNeighborEmbeddingBlock(
            num_channels=self.n_channels,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            num_species=self.num_species,
            num_rbf=self.n_rbf,
        )
        params = block.init(self.key, graph_in)
        graph = block.apply(params, graph_in)

        # Output shape assertions
        assert graph.nodes.features["embedding_scalars"].shape == (
            self.n_nodes,
            self.n_channels,
        )

    def test_visnet_edge_embedding_block(self):
        graph_in = self.edge_embedding_block_input()
        block = VisnetEdgeEmbeddingBlock(
            num_channels=self.n_channels,
            num_rbf=self.n_rbf,
        )
        params = block.init(self.key, graph_in)
        graph = block.apply(params, graph_in)

        # Output shape assertions
        assert graph.edges.features["embedding"].shape == (
            self.n_edges,
            self.n_channels,
        )

    def test_visnet_readout_block(self):
        graph_in = self.readout_block_input()
        block = VisnetMultiHeadReadoutBlock(
            num_heads=2,
            num_channels=self.n_channels,
            activation=self.activation,
            vecnorm_type=self.vecnorm_type,
            l_max=self.l_max,
            predict_partial_charges=False,
        )
        params = block.init(self.key, graph_in)
        graph = block.apply(params, graph_in)

        # Output shape assertions
        assert graph.nodes.features["outputs"].shape == (self.n_nodes, 2, 1)


def test_visnet_spherical_harmonics_v1_against_v2():
    l_max = 2
    key = jax.random.PRNGKey(42)
    edge_vec = jax.random.normal(key, (10, 3))
    edge_vec_normalized = edge_vec / jnp.linalg.norm(edge_vec, axis=-1, keepdims=True)

    v2_sphere = SphericalHarmonicsBlock(
        l_max=l_max,
        normalize=True,
        normalization="norm",
    )
    v2_sphere_output = v2_sphere(edge_vec_normalized).array[:, 1:]

    v1_sphere = Sphere(l_max)
    v1_sphere_output = v1_sphere(edge_vec_normalized)

    assert v1_sphere_output.shape == v2_sphere_output.shape
    assert jnp.allclose(v1_sphere_output, v2_sphere_output, atol=1e-10)
