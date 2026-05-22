from typing import Callable

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from e3j.linen import LinearIndexwise
from jax import Array
from numpy.testing import assert_allclose

from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.models.mace.layer import MaceLayer
from mlip.models_v1.blocks import FullyConnectedTensorProduct
from mlip.models_v1.mace.models import MaceLayer as MaceLayerV1

# MaceLayerV1 expects callables, in V2 options can be parsed.
ACTIVATIONS = {
    "swish": jax.nn.swish,
    "relu": jax.nn.relu,
}


class _TestMaceLayer:
    use_residuals: bool
    last_layer: bool
    num_channels: int
    source_irreps: e3nn.Irreps
    node_irreps: e3nn.Irreps
    interaction_irreps: e3nn.Irreps
    activation: Callable | str
    num_species: int = 3
    # InteractionBlock:
    l_max: int
    avg_num_neighbors: float = 2.0
    # EquivariantProductBasisBlock:
    correlation: int
    soft_normalization: float | None = 2.0
    # ReadoutBlock:
    output_irreps: e3nn.Irreps = e3nn.Irreps("0e")
    readout_mlp_irreps: e3nn.Irreps = e3nn.Irreps("16x0e")
    num_readout_heads: int = 1
    gate_nodes: bool = False

    num_rbf: int = 8

    symmetric_contraction_backend: str = "e3nn"
    use_gaunt_tp_message_passing: bool = False

    key = jax.random.key(123)
    n_node: int = 10
    n_edge: int = 33
    n_graph: int = 4
    allowed_atomic_numbers: list[int] = [1, 6, 8]

    @pytest.fixture(scope="class")
    def module_v1(self) -> nn.Module:
        activation = (
            self.activation
            if callable(self.activation)
            else ACTIVATIONS[self.activation]
        )
        return MaceLayerV1(
            selector_tp=not self.use_residuals,
            last_layer=self.last_layer,
            num_channels=self.num_channels,
            node_irreps=self.node_irreps,
            interaction_irreps=self.interaction_irreps,
            activation=activation,
            num_species=self.num_species,
            l_max=self.l_max,
            avg_num_neighbors=self.avg_num_neighbors,
            correlation=self.correlation,
            symmetric_tensor_product_basis=False,
            soft_normalization=self.soft_normalization,
            output_irreps=self.output_irreps,
            readout_mlp_irreps=self.readout_mlp_irreps,
            num_readout_heads=self.num_readout_heads,
            gate_nodes=self.gate_nodes,
            off_diagonal=False,  # dropped in V2, was dead or broken
        )

    @pytest.fixture(scope="class")
    def module_v2(self) -> nn.Module:
        return MaceLayer(
            use_residuals=self.use_residuals,
            last_layer=self.last_layer,
            num_channels=self.num_channels,
            source_irreps=self.source_irreps,
            interaction_irreps=self.interaction_irreps,
            node_irreps=self.node_irreps,
            activation=self.activation,
            num_species=self.num_species,
            l_max=self.l_max,
            num_rbf=self.num_rbf,
            # radial_mlp_hidden must be [64]*3 for v1 (hardcoded in MACEv1).
            radial_mlp_hidden=[64, 64, 64],
            radial_mlp_activation=self.activation,
            avg_num_neighbors=self.avg_num_neighbors,
            correlation=self.correlation,
            soft_normalization=self.soft_normalization,
            output_irreps=self.output_irreps,
            readout_mlp_irreps=self.readout_mlp_irreps,
            num_readout_heads=self.num_readout_heads,
            gate_nodes=self.gate_nodes,
            symmetric_contraction_backend=self.symmetric_contraction_backend,
            use_gaunt_tp_message_passing=self.use_gaunt_tp_message_passing,
        )

    @pytest.fixture(scope="class")
    def features(self) -> dict[str, Array | e3nn.IrrepsArray]:
        # Generate random species and atomic numbers
        num_species = len(self.allowed_atomic_numbers)
        node_species = jax.random.randint(self.key, (self.n_node,), 0, num_species)
        atomic_numbers = jnp.array([
            self.allowed_atomic_numbers[s] for s in node_species
        ])
        species_one_hot = jnp.eye(num_species)[node_species]
        # Generate random node features (channel-expanded, as after embedding)
        channel_irreps = self.num_channels * e3nn.Irreps(self.source_irreps)
        node_dim = channel_irreps.dim
        node_feats = jax.random.normal(self.key, (self.n_node, node_dim))
        node_feats = e3nn.IrrepsArray(
            channel_irreps,
            node_feats,
        )
        # Generate random edge embeddings
        radial_embedding = jax.random.normal(self.key, (self.n_edge, self.num_rbf))
        radial_embedding = e3nn.IrrepsArray(
            f"{self.num_rbf}x0e",
            radial_embedding,
        )
        vectors = jax.random.normal(self.key, (self.n_edge, 3))
        spherical_embedding = e3nn.spherical_harmonics(
            e3nn.Irreps.spherical_harmonics(self.l_max),
            vectors,
            normalize=True,
        )
        # Generate senders and receivers
        senders, receivers = jax.random.randint(
            self.key, (2, self.n_edge), 0, self.n_node
        )
        return dict(
            latent_node=node_feats,
            species_one_hot=species_one_hot,
            atomic_numbers=atomic_numbers,
            node_species=node_species,
            vectors=vectors,
            radial_embedding=radial_embedding,
            spherical_embedding=spherical_embedding,
            senders=senders,
            receivers=receivers,
        )

    @pytest.fixture(scope="class")
    def inputs_v1(self, features) -> tuple[Array | e3nn.IrrepsArray, ...]:
        # Note: edge vectors passed to MaceLayer, and Harmonics recomputed
        #       in each convolution block.
        return (
            e3nn.IrrepsArray("1o", features["vectors"]),
            features["latent_node"],
            features["node_species"],
            features["radial_embedding"],
            features["senders"],
            features["receivers"],
            None,  # node_mask: dead parameter
        )

    @pytest.fixture(scope="class")
    def inputs_v2(self, features) -> tuple[Graph]:
        # Note: spherical_embedding replaces edge vectors, they are
        #       assigned upstream in MaceEmbeddingBlock.
        graph = Graph(
            nodes=GraphNodes(
                positions=jnp.zeros((self.n_node, 3)),
                atomic_numbers=None,
                features=dict(
                    embedding=features["latent_node"],
                    species=features["node_species"],
                    species_one_hot=features["species_one_hot"],
                ),
            ),
            edges=GraphEdges(
                features=dict(
                    spherical_embedding=features["spherical_embedding"],
                    radial_embedding=features["radial_embedding"],
                ),
            ),
            globals=GraphGlobals(cell=None, weight=1),
            senders=features["senders"],
            receivers=features["receivers"],
            n_node=self.n_node,
            n_edge=self.n_edge,
        )
        return (graph,)

    def test_self_interaction_v1_against_v2(self, features, standardize_params):
        """Check FCTP(version=2) matches LinearIndexwise.

        For use_residuals=False (first layer): tests linear_by_species_block.
        For use_residuals=True (later layers): tests residual skip-connection.
        """
        species = features["node_species"]
        species_one_hot = features["species_one_hot"]
        num_species = len(self.allowed_atomic_numbers)

        if not self.use_residuals:
            # First layer: linear_by_species_block operates on
            # interaction_irreps (output of InteractionBlock).
            source = self.num_channels * self.interaction_irreps
            target = source
        else:
            # Later layers: residual skip-connection from
            # source_irreps to node_irreps.
            source = self.num_channels * e3nn.Irreps(self.source_irreps)
            target = self.num_channels * e3nn.Irreps(self.node_irreps).regroup()

        node_dim = source.dim
        node_feats = jax.random.normal(self.key, (self.n_node, node_dim))
        node_feats = e3nn.IrrepsArray(source, node_feats)

        # V1: FCTP with version=2
        fctp = FullyConnectedTensorProduct(
            irreps_in1=source,
            irreps_in2=num_species * e3nn.Irreps("0e"),
            irreps_out=target,
            version=2,
        )
        params_v1 = standardize_params(fctp.init(self.key, node_feats, species_one_hot))
        out_v1 = fctp.apply(params_v1, node_feats, species_one_hot).array

        # V2: LinearIndexwise
        liw = LinearIndexwise(
            source_irreps=source,
            target_irreps=e3nn.Irreps(target).regroup(),
            num_channels=None,
            num_indices=num_species,
            layout="E3NN",
        )
        params_v2 = standardize_params(liw.init(self.key, node_feats.array, species))
        out_v2 = liw.apply(params_v2, node_feats.array, species)

        assert_allclose(out_v1, out_v2, atol=1e-5, rtol=1e-5)

    def test_v2_matches_v1(
        self, module_v1, module_v2, inputs_v1, inputs_v2, standardize_params
    ):
        if (
            self.symmetric_contraction_backend != "e3nn"
            or self.use_gaunt_tp_message_passing
        ):
            pytest.skip(
                "V1 parity is only checked for e3nn symmetric contraction "
                "without Gaunt message passing."
            )
        params_v1 = standardize_params(module_v1.init(self.key, *inputs_v1))
        params_v2 = standardize_params(module_v2.init(self.key, *inputs_v2))

        output_v1 = jax.jit(module_v1.apply)(params_v1, *inputs_v1)
        output_v2 = jax.jit(module_v2.apply)(params_v2, *inputs_v2)

        feats_v1 = output_v1[1].array
        feats_v2 = output_v2.nodes.features["latent"].array
        assert_allclose(feats_v1, feats_v2, atol=1e-4, rtol=1e-4)

        readout_v1 = output_v1[0].array
        readout_v2 = output_v2.nodes.features["outputs"].array
        assert_allclose(readout_v1, readout_v2, atol=1e-4, rtol=1e-4)

    def test_equivariance_v2(self, module_v2, inputs_v2, features):
        """Check that MaceLayer output node features are equivariant under SO(3).

        Rotates node_feats and spherical_embedding (both equivariant) by the same
        rotation R, and verifies that the output transforms accordingly.
        species_one_hot and radial_embedding are l=0 scalars and are left unchanged.
        """
        (graph,) = inputs_v2
        params = module_v2.init(self.key, graph)
        apply = jax.jit(module_v2.apply)

        rotation = e3nn.rand_matrix(self.key)

        # g(f(x)): run on original input, then rotate output
        out_feats = apply(params, graph).nodes.features["latent"]
        rotation_out = out_feats.irreps.D_from_matrix(rotation)

        gfx = out_feats.array @ rotation_out

        # f(g(x)): rotate equivariant inputs, then run
        node_feats = features["latent_node"]
        edge_sh = features["spherical_embedding"]
        rotation_in_node = node_feats.irreps.D_from_matrix(rotation)
        rotation_in_sh = edge_sh.irreps.D_from_matrix(rotation)

        rotated_graph = graph.update_node_features(
            latent=e3nn.IrrepsArray(
                node_feats.irreps, node_feats.array @ rotation_in_node
            )
        ).update_edge_features(
            spherical_embedding=e3nn.IrrepsArray(
                edge_sh.irreps, edge_sh.array @ rotation_in_sh
            )
        )

        fgx = apply(params, rotated_graph).nodes.features["latent"].array

        # Assert equivalent
        norm = float(jnp.sqrt(jnp.sum((gfx - fgx) ** 2))) / gfx.size
        assert norm < 1e-5


class TestMaceLayerCorrelation3(_TestMaceLayer):
    # use_residuals=True: non-first layer, has skip-connection.
    # source_irreps must match node_irreps so the LinearIndexwise
    # residual block has matching (l, p) blocks in source and target.
    use_residuals = True
    last_layer = False
    num_channels = 16
    source_irreps = e3nn.Irreps("0e + 1o")
    interaction_irreps = e3nn.Irreps("0e + 1o + 2e")
    node_irreps = e3nn.Irreps("0e + 1o")
    activation = staticmethod(jax.nn.tanh)
    correlation = 3
    l_max = 2


class TestMaceLayerCorrelation2(_TestMaceLayer):
    # use_residuals=True: non-first layer, has skip-connection.
    # source_irreps must match node_irreps.
    use_residuals = True
    last_layer = False
    num_channels = 16
    source_irreps = e3nn.Irreps("0e + 1o")
    interaction_irreps = e3nn.Irreps("0e + 1o + 2e + 3o")
    node_irreps = e3nn.Irreps("0e + 1o")
    activation = "swish"
    correlation = 2
    l_max = 3


class TestMaceLayerFirstLayer(_TestMaceLayer):
    # use_residuals=False: first layer, no skip-connection.
    # source_irreps is scalar embedding (only 0e).
    use_residuals = False
    last_layer = False
    num_channels = 16
    source_irreps = e3nn.Irreps("0e")
    interaction_irreps = e3nn.Irreps("0e + 1o + 2e")
    node_irreps = e3nn.Irreps("0e + 1o")
    activation = "swish"
    correlation = 3
    l_max = 2


class TestMaceLayerGauntTpEquivariance(_TestMaceLayer):
    """SO(3) equivariance with Gaunt symmetric contraction and Gaunt message passing."""

    symmetric_contraction_backend = "gaunt_tp"
    use_gaunt_tp_message_passing = True
    # Same irreps setup as TestMaceLayerCorrelation2 (middle layer).
    use_residuals = True
    last_layer = False
    num_channels = 16
    source_irreps = e3nn.Irreps("0e + 1o")
    interaction_irreps = e3nn.Irreps("0e + 1o + 2e + 3o")
    node_irreps = e3nn.Irreps("0e + 1o")
    activation = "swish"
    correlation = 2
    l_max = 3
