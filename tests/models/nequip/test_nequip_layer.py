import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from e3j.linen import LinearIndexwise
from jax import Array
from numpy.testing import assert_allclose

from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.models.nequip.layer import NequipLayer
from mlip.models.nequip.nequip_helpers import split_target_node_irreps
from mlip.models_v1.blocks import FullyConnectedTensorProduct
from mlip.models_v1.nequip.models import NequipLayer as NequipLayerV1


class _TestNequipLayer:
    node_irreps: str = "0e + 1o + 2e"
    l_max: int = 3
    num_rbf: int = 8
    use_residual_connection: bool = True
    # v1 conversion only works when radial_mlp_hidden is constant (all layers equal).
    radial_mlp_activation: str = "swish"
    radial_mlp_hidden: list[int] = [64, 64]
    avg_num_neighbors: float = 10.0
    radial_mlp_variance_scale: float = 4.0
    nonlinearities: dict[str, str] = {"e": "swish", "o": "tanh"}

    key = jax.random.key(123)
    n_node: int = 10
    n_edge: int = 33
    n_graph: int = 4
    allowed_atomic_numbers: list[int] = [1, 6, 8]

    @pytest.fixture(scope="class")
    def module_v1(self) -> nn.Module:
        return NequipLayerV1(
            node_irreps=e3nn.Irreps(self.node_irreps),
            use_residual_connection=self.use_residual_connection,
            nonlinearities=self.nonlinearities,
            radial_net_nonlinearity=self.radial_mlp_activation,
            radial_net_n_hidden=self.radial_mlp_hidden[0],
            radial_net_n_layers=len(self.radial_mlp_hidden),
            avg_num_neighbors=self.avg_num_neighbors,
            scalar_mlp_std=self.radial_mlp_variance_scale,
            num_bessel=2048,  # NOTE: dead parameter
        )

    @pytest.fixture(scope="class")
    def module_v2(self) -> nn.Module:
        return NequipLayer(
            target_irreps=e3nn.Irreps(self.node_irreps),  # FIXME: str unsupported
            source_node_irreps=e3nn.Irreps(self.node_irreps),  # Matches node_feats
            l_max=self.l_max,
            num_species=len(self.allowed_atomic_numbers),
            num_rbf=self.num_rbf,
            use_residual_connection=self.use_residual_connection,
            nonlinearities=self.nonlinearities,
            radial_mlp_activation=self.radial_mlp_activation,
            radial_mlp_hidden=self.radial_mlp_hidden,
            avg_num_neighbors=self.avg_num_neighbors,
            radial_mlp_variance_scale=self.radial_mlp_variance_scale,
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
        dim_ylm = (self.l_max + 1) ** 2
        # Generate random node features
        node_dim = e3nn.Irreps(self.node_irreps).dim
        node_feats = jax.random.normal(self.key, (self.n_node, node_dim))
        node_feats = e3nn.IrrepsArray(
            self.node_irreps,
            node_feats,
        )
        # Generate random edge embeddings
        radial_embeddings = jax.random.normal(self.key, (self.n_edge, self.num_rbf))
        radial_embeddings = e3nn.IrrepsArray(
            f"{self.num_rbf}x0e",
            radial_embeddings,
        )
        spherical_embeddings = jax.random.normal(self.key, (self.n_edge, dim_ylm))
        spherical_embeddings = e3nn.IrrepsArray(
            e3nn.Irreps.spherical_harmonics(self.l_max),
            spherical_embeddings,
        )
        # Generate senders and receivers
        senders, receivers = jax.random.randint(
            self.key, (2, self.n_edge), 0, self.n_node
        )
        return dict(
            latent=node_feats,
            species=node_species,
            species_one_hot=species_one_hot,
            atomic_numbers=atomic_numbers,
            node_species=node_species,
            radial_embedding=radial_embeddings,
            spherical_embedding=spherical_embeddings,
            senders=senders,
            receivers=receivers,
        )

    @pytest.fixture(scope="class")
    def inputs_v1(self, features) -> tuple[Array | e3nn.IrrepsArray, ...]:
        return (
            features["latent"],
            features["species_one_hot"],
            features["spherical_embedding"],
            features["senders"],
            features["receivers"],
            features["radial_embedding"].array,
        )

    @pytest.fixture(scope="class")
    def inputs_v2(self, features) -> tuple[Graph]:
        graph = Graph(
            nodes=GraphNodes(
                positions=jnp.zeros((self.n_node, 3)),
                atomic_numbers=None,
                features=dict(
                    latent=features["latent"],
                    species=features["species"],
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

    def test_residual_block_v1_against_v2(self, features, standardize_params):
        """Check that FCTP(version=2) residual matches LinearIndexwise."""
        node_feats = features["latent"]
        species = features["species"]
        species_one_hot = features["species_one_hot"]
        num_species = len(self.allowed_atomic_numbers)

        # gate_irreps for node_irreps = "0e + 1o + 2e" with l_max = 3
        source = e3nn.Irreps(self.node_irreps)
        sh_irreps = e3nn.Irreps.spherical_harmonics(self.l_max)
        scalars, gates, nonscalars = split_target_node_irreps(
            source,
            sh_irreps,
            e3nn.Irreps(self.node_irreps),
        )
        gate_irreps = scalars + gates + nonscalars

        # V1: FCTP with version=2 (path_weight-rescaled params)
        fctp = FullyConnectedTensorProduct(
            irreps_out=gate_irreps,
            version=2,
        )
        params_v1 = standardize_params(fctp.init(self.key, node_feats, species_one_hot))
        out_v1 = fctp.apply(params_v1, node_feats, species_one_hot).array

        # V2: LinearIndexwise
        liw = LinearIndexwise(
            source_irreps=source,
            target_irreps=e3nn.Irreps(gate_irreps).regroup(),
            num_channels=None,
            num_indices=num_species,
        )
        params_v2 = standardize_params(liw.init(self.key, node_feats.array, species))
        out_v2 = liw.apply(params_v2, node_feats.array, species)

        assert_allclose(out_v1, out_v2, atol=1e-5, rtol=1e-5)

    def test_v1_against_v2(
        self, module_v1, module_v2, inputs_v1, inputs_v2, standardize_params
    ):
        params_v1 = standardize_params(module_v1.init(self.key, *inputs_v1))
        params_v2 = standardize_params(module_v2.init(self.key, *inputs_v2))

        output_v1 = jax.jit(module_v1.apply)(params_v1, *inputs_v1)
        output_v2 = jax.jit(module_v2.apply)(params_v2, *inputs_v2)

        feats_out_v1 = output_v1.array
        feats_out_v2 = output_v2.nodes.features["latent"].array
        assert_allclose(feats_out_v1, feats_out_v2, atol=1e-4, rtol=1e-4)

    def test_equivariance_v2(self, module_v2, inputs_v2, features):
        """Check that NequipLayer output node features are equivariant under SO(3).

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
        node_feats = features["latent"]
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


class TestNequipLayerN2ConstantMul(_TestNequipLayer):
    node_irreps = "8x0e + 8x1o + 8x2e"


class TestNequipLayerN3ConstantMul(_TestNequipLayer):
    node_irreps = "4x0e + 4x1o + 4x2e + 4x3o"


class TestNequipLayerN2(_TestNequipLayer):
    node_irreps = "8x0e + 8x1o + 4x2e"
