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
from functools import partial
from pathlib import Path
from typing import Callable

import e3j
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read_atoms
from e3j.utils.options import Layout

import mlip.models_v1 as models_v1
from mlip.data import ChemicalSystem, DatasetInfo
from mlip.data.helpers.dummy_init_graph import (
    get_dummy_graph_for_model_init as get_dummy_graph,  # noqa
)
from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.models import Mace, Nequip, Visnet
from mlip.models.blocks import SpeciesAssignmentBlock
from mlip.models.config import MLIPNetworkConfig
from mlip.models.esen.config import EsenConfig
from mlip.models.esen.network import Esen
from mlip.models.force_field import ForceField
from mlip.models.inference_context import InferenceContext
from mlip.models.mlip_network import MLIPNetwork
from mlip.typing.properties import Properties

GRAPH_CUTOFF_ANGSTROM = 3.0
SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"
XYZ_FILE_PATH = SAMPLE_DATA_DIR / "Dimethyl_sulfoxide.xyz"
KEY = jax.random.key(123)

# Cache compilations
jax.config.update(
    "jax_compilation_cache_dir",
    os.getenv(
        "JAX_COMPILATION_CACHE_DIR",
        str(Path.home() / ".cache" / "jax_compilation_cache"),
    ),
)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

e3j.config(
    layout=Layout.E3NN,
    tensor_product="SPARSE",
)


def _make_customizable_graph(n_nodes: int, n_edges: int) -> Graph:
    """Create a minimal Graph with the given number of nodes and edges."""
    positions = np.random.default_rng(0).standard_normal((n_nodes, 3))
    senders = np.zeros(n_edges, dtype=np.int32)
    receivers = np.minimum(np.arange(n_edges, dtype=np.int32), np.int32(n_nodes - 1))
    return Graph(
        nodes=GraphNodes(
            positions=positions,
            forces=np.zeros((n_nodes, 3)),
            atomic_numbers=np.ones(n_nodes, dtype=np.int32),
        ),
        edges=GraphEdges(shifts=np.zeros((n_edges, 3)), displ_fun=None),
        globals=jax.tree.map(
            lambda x: x[None, ...],
            GraphGlobals(
                cell=np.eye(3) * 10.0,
                energy=np.array(0.0),
                stress=np.zeros((3, 3)),
                weight=np.asarray(1.0),
            ),
        ),
        senders=senders,
        receivers=receivers,
        n_node=np.array([n_nodes]),
        n_edge=np.array([n_edges]),
    )


@pytest.fixture
def make_customizable_graph():
    """Fixture that returns a factory function for creating minimal Graphs."""
    return _make_customizable_graph


@pytest.fixture(scope="session")
def dataset_info() -> DatasetInfo:
    """Returns a dataset info for H, C, O, Na, S, Cl atoms and 3 Ångström cutoff."""
    allowed_z_numbers = {1, 6, 8, 11, 16, 17}
    available_total_charges = {1, 0, -1}
    return DatasetInfo(
        atomic_energies_map={k: float(-k) for k in allowed_z_numbers},
        total_charge_set=available_total_charges,
        avg_num_neighbors=1.0,
        avg_r_min_angstrom=0.1,
        graph_cutoff_angstrom=GRAPH_CUTOFF_ANGSTROM,
        scaling_mean=0.0,
        scaling_stdev=1.0,
    )


@pytest.fixture(scope="session")
def multi_head_dataset_info() -> DatasetInfo:
    """Returns a multi-dataset DatasetInfo with 2 datasets.

    Index 0 values match the single-head `dataset_info` fixture.
    """
    allowed_z_numbers = {1, 6, 8, 11, 16, 17}
    e0_map_0 = {k: float(-k) for k in allowed_z_numbers}
    e0_map_1 = {k: float(-k + 1) for k in allowed_z_numbers}
    return DatasetInfo(
        dataset_name=["dataset_0", "dataset_1"],
        atomic_energies_map=[e0_map_0, e0_map_1],
        avg_num_neighbors=1.0,
        avg_r_min_angstrom=0.1,
        graph_cutoff_angstrom=GRAPH_CUTOFF_ANGSTROM,
        scaling_mean=0.0,
        scaling_stdev=1.0,
        atomic_energies_removed=False,
    )


@pytest.fixture(scope="session")
def setup_system(dataset_info) -> tuple[Atoms, Graph]:
    """Returns an `(atoms, graph)` triple for dimethyl sulfoxide.

    Allowed atomic numbers are [1,6,8,16] <=> ["H","C","O","S"].
    """
    atoms = ase_read_atoms(XYZ_FILE_PATH)
    graph_cutoff_angstrom = dataset_info.graph_cutoff_angstrom

    chemical_system = ChemicalSystem.from_ase_atoms(atoms)
    graph = Graph.from_chemical_system(chemical_system, graph_cutoff_angstrom)

    assert all(z in dataset_info.atomic_energies_map for z in atoms.numbers)

    return atoms, graph


@pytest.fixture(scope="session")
def salt_graph(dataset_info) -> Graph:
    """Returns an NaCl CFC lattice, with artifactual ~2A lattice parameter.

    The lattice parameter is chosen so that only Na-Cl bonds are within 3A cutoff.
    """
    # NaCl CFC lattice (artifactual lattice parameter)
    salt = ChemicalSystem(
        atomic_numbers=np.array([11, 17]),
        positions=np.array([[0.0, 0.0, 0.0], [1.5, 1.6, 1.6]]),
        cell=np.array([[3.2, 0.1, 0.0], [0.0, 3.2, 0.0], [0.0, 0.0, 3.1]]),
        pbc=(True, True, True),
    )
    # slightly smaller than lattice width:
    # only Na-Cl bond should be within cutoff.
    cutoff = GRAPH_CUTOFF_ANGSTROM

    graph = Graph.from_chemical_system(salt, cutoff)
    return graph


def _replace_v1_linear_weights(params: dict) -> dict:
    """Re-generate v1 Linear weights to match `e3j.linen.Linear` convention.

    Replaces the weights created for v1 linear layers by `standardize_parameters`.

    Uses of `e3nn_jax.FunctionalLinear` and `e3nn.flax.Linear` in v1 are replaced by
    `e3j.linen.Linear` in v2. The v1 modules both store weights with key `w[...` as
    `(m_in, m_out)`, while v2 stores weights with key `weight...` as `(m_out, m_in)`.
    """
    key = KEY

    def fix(path, p):
        leaf_key = path[-1].key if hasattr(path[-1], "key") else ""
        is_v1_linear_weights = (
            len(p.shape) == 2
            and isinstance(leaf_key, str)
            and leaf_key.startswith("w[")
        )
        if is_v1_linear_weights:
            return jax.random.normal(key, (p.shape[1], p.shape[0])).T
        return p

    return jax.tree_util.tree_map_with_path(fix, params)


def standardize_parameters(params: dict) -> dict:
    """Overwrite parameters with shape-determined random parameters.

    This step is useful to compare equivalent yet differently structured
    dictionary of parameters, e.g. to check V1 against V2 parameters.

    Args:
        params: Parameters to standardize.
        ref_v1_params: Optional v1 reference parameters. Only provided for
            NequIP v2, and used to identify merged paths in `Linear` that need fixing.
    """
    key = KEY
    standardized_params = jax.tree.map(
        lambda p: jax.random.normal(key, p.shape), params
    )
    standardized_params = _replace_v1_linear_weights(standardized_params)
    return standardized_params


@pytest.fixture(scope="session")
def standardize_params() -> Callable[[dict], dict]:
    """Overwrite parameters with shape-determined random parameters.

    This step is useful to compare equivalent yet differently structured
    dictionary of parameters, e.g. to check V1 against V2 parameters.
    """
    return standardize_parameters


@pytest.fixture(scope="session")
def mace_config():
    # radial_mlp_hidden must be [64, 64, 64] for v1 conversion (hardcoded in MACEv1).
    return Mace.Config(
        num_layers=2,
        num_rbf=8,
        radial_envelope="polynomial_envelope",
        activation="silu",
        num_channels=4,
        readout_irreps=("4x0e", "0e"),
        correlation=2,
        node_symmetry=2,
        l_max=2,
        gate_nodes=True,
        symmetric_contraction_backend="e3nn",  # needed for v1 matching
    )


@pytest.fixture(scope="session")
def visnet_config():
    return Visnet.Config(
        num_layers=2,
        num_channels=6,
        l_max=2,
        num_heads=2,
        num_rbf=4,
        activation="silu",
        attn_activation="silu",
        vecnorm_type="max_min",
    )


@pytest.fixture(scope="session")
def nequip_config():
    # v1 conversion only works when radial_mlp_hidden is constant (all layers equal).
    return Nequip.Config(
        num_layers=2,
        target_irreps="4x0e + 4x0o + 4x1o + 4x1e + 4x2e + 4x2o",
        l_max=2,
        num_rbf=8,
        radial_mlp_activation="swish",
        radial_mlp_hidden=[8, 8],
        radial_envelope="polynomial_envelope",
        radial_mlp_variance_scale=4.0,
    )


@pytest.fixture(scope="session")
def esen_config():
    return EsenConfig()


@pytest.fixture(scope="session")
def mace_force_field(mace_config, dataset_info):
    mace_model = Mace(mace_config, dataset_info)
    mace_ff = ForceField.from_mlip_network(
        mace_model,
        seed=42,
        required_properties=Properties(stress=True),
    )
    return ForceField(
        mace_ff.predictor,
        standardize_parameters(mace_ff.params),
        inference_context=mace_ff.inference_context,
    )


@pytest.fixture(scope="session")
def multi_head_mace_force_field(mace_config, multi_head_dataset_info):
    multi_head_config = mace_config.model_copy(update={"num_readout_heads": 2})
    mace_model = Mace(multi_head_config, multi_head_dataset_info)
    mace_ff = ForceField.from_mlip_network(
        mace_model,
        seed=42,
        required_properties=Properties(stress=True),
        inference_context=InferenceContext(
            dataset_name=multi_head_dataset_info.dataset_name[0]
        ),
    )
    return ForceField(
        mace_ff.predictor,
        standardize_parameters(mace_ff.params),
        inference_context=mace_ff.inference_context,
    )


@pytest.fixture(scope="session")
def partial_charges_mace_force_field(mace_config, dataset_info):
    partial_charges_config = mace_config.model_copy(
        update={"predict_partial_charges": True}
    )
    mace_model = Mace(partial_charges_config, dataset_info)
    mace_ff = ForceField.from_mlip_network(
        mace_model,
        seed=42,
        required_properties=Properties(stress=True, partial_charges=True),
    )
    return ForceField(
        mace_ff.predictor,
        mace_ff.params,
    )


@pytest.fixture(scope="session")
def lri_mace_force_field(mace_config, dataset_info):
    lri_config = mace_config.model_copy(
        update={"predict_partial_charges": True, "use_coulomb_term": True}
    )
    lri_dataset_info = dataset_info.model_copy(
        update={"long_range_cutoff_angstrom": 5.0}
    )
    mace_model = Mace(lri_config, lri_dataset_info)
    mace_ff = ForceField.from_mlip_network(
        mace_model,
        seed=42,
        required_properties=Properties(stress=True, partial_charges=True),
    )
    return ForceField(
        mace_ff.predictor,
        mace_ff.params,
    )


@pytest.fixture(scope="session")
def visnet_force_field(visnet_config, dataset_info):
    visnet_model = Visnet(visnet_config, dataset_info)
    visnet_ff = ForceField.from_mlip_network(
        visnet_model,
        seed=42,
        required_properties=Properties(stress=True),
    )
    return ForceField(
        visnet_ff.predictor,
        standardize_parameters(visnet_ff.params),
    )


@pytest.fixture(scope="session")
def legacy_visnet_force_field(visnet_config, dataset_info):
    legacy_config = visnet_config.model_copy(update={"use_legacy_visnet": True})
    visnet_model = Visnet(legacy_config, dataset_info)
    visnet_ff = ForceField.from_mlip_network(
        visnet_model,
        seed=42,
        required_properties=Properties(stress=True),
    )
    return ForceField(
        visnet_ff.predictor,
        standardize_parameters(visnet_ff.params),
    )


@pytest.fixture(scope="session")
def partial_charges_visnet_force_field(visnet_config, dataset_info):
    partial_charges_config = visnet_config.model_copy(
        update={"predict_partial_charges": True}
    )
    visnet_model = Visnet(partial_charges_config, dataset_info)
    visnet_ff = ForceField.from_mlip_network(
        visnet_model,
        seed=42,
        required_properties=Properties(stress=True, partial_charges=True),
    )
    return ForceField(
        visnet_ff.predictor,
        visnet_ff.params,
    )


@pytest.fixture(scope="session")
def lri_visnet_force_field(visnet_config, dataset_info):
    lri_config = visnet_config.model_copy(
        update={"predict_partial_charges": True, "use_coulomb_term": True}
    )
    lri_dataset_info = dataset_info.model_copy(
        update={"long_range_cutoff_angstrom": 5.0}
    )
    visnet_model = Visnet(lri_config, lri_dataset_info)
    visnet_ff = ForceField.from_mlip_network(
        visnet_model,
        seed=42,
        required_properties=Properties(stress=True, partial_charges=True),
    )
    return ForceField(
        visnet_ff.predictor,
        visnet_ff.params,
    )


@pytest.fixture(scope="session")
def total_charge_embedding_visnet_force_field(visnet_config, dataset_info):
    total_charge_embedding_config = visnet_config.model_copy(
        update={"use_total_charge_embedding": True}
    )
    visnet_model = Visnet(total_charge_embedding_config, dataset_info)
    visnet_ff = ForceField.from_mlip_network(
        visnet_model,
        seed=42,
        required_properties=Properties(stress=True),
    )
    return ForceField(
        visnet_ff.predictor,
        standardize_parameters(visnet_ff.params),
    )


@pytest.fixture(scope="session")
def nequip_force_field(nequip_config, dataset_info):
    nequip_model = Nequip(nequip_config, dataset_info)
    nequip_ff = ForceField.from_mlip_network(
        nequip_model,
        seed=42,
        required_properties=Properties(stress=True),
    )
    return ForceField(
        nequip_ff.predictor,
        standardize_parameters(nequip_ff.params),
    )


@pytest.fixture(scope="session")
def multi_head_nequip_force_field(nequip_config, multi_head_dataset_info):
    multi_head_config = nequip_config.model_copy(update={"num_readout_heads": 2})
    nequip_model = Nequip(multi_head_config, multi_head_dataset_info)
    nequip_ff = ForceField.from_mlip_network(
        nequip_model,
        seed=42,
        required_properties=Properties(stress=True),
        inference_context=InferenceContext(
            dataset_name=multi_head_dataset_info.dataset_name[0]
        ),
    )
    return ForceField(
        nequip_ff.predictor,
        standardize_parameters(nequip_ff.params),
        inference_context=nequip_ff.inference_context,
    )


@pytest.fixture(scope="session")
def partial_charges_nequip_force_field(nequip_config, dataset_info):
    partial_charges_config = nequip_config.model_copy(
        update={"predict_partial_charges": True}
    )
    nequip_model = Nequip(partial_charges_config, dataset_info)
    nequip_ff = ForceField.from_mlip_network(
        nequip_model,
        seed=42,
        required_properties=Properties(stress=True, partial_charges=True),
    )
    return ForceField(
        nequip_ff.predictor,
        nequip_ff.params,
    )


@pytest.fixture(scope="session")
def lri_nequip_force_field(nequip_config, dataset_info):
    lri_config = nequip_config.model_copy(
        update={"predict_partial_charges": True, "use_coulomb_term": True}
    )
    lri_dataset_info = dataset_info.model_copy(
        update={"long_range_cutoff_angstrom": 5.0}
    )
    nequip_model = Nequip(lri_config, lri_dataset_info)
    nequip_ff = ForceField.from_mlip_network(
        nequip_model,
        seed=42,
        required_properties=Properties(stress=True, partial_charges=True),
    )
    return ForceField(
        nequip_ff.predictor,
        nequip_ff.params,
    )


@pytest.fixture(scope="session")
def esen_force_field(esen_config, dataset_info):
    esen_model = Esen(esen_config, dataset_info)
    esen_ff = ForceField.from_mlip_network(
        esen_model,
        seed=42,
        required_properties=Properties(stress=True),
    )
    return ForceField(
        esen_ff.predictor,
        standardize_parameters(esen_ff.params),
    )


@pytest.fixture(scope="session")
def partial_charges_esen_force_field(esen_config, dataset_info):
    partial_charges_config = esen_config.model_copy(
        update={"predict_partial_charges": True}
    )
    esen_model = Esen(partial_charges_config, dataset_info)
    esen_ff = ForceField.from_mlip_network(
        esen_model,
        seed=42,
        required_properties=Properties(stress=True, partial_charges=True),
    )
    return ForceField(esen_ff.predictor, esen_ff.params)


@pytest.fixture(scope="session")
def lri_esen_force_field(esen_config, dataset_info):
    lri_config = esen_config.model_copy(
        update={"predict_partial_charges": True, "use_coulomb_term": True}
    )
    lri_dataset_info = dataset_info.model_copy(
        update={"long_range_cutoff_angstrom": 5.0}
    )
    esen_model = Esen(lri_config, lri_dataset_info)
    esen_ff = ForceField.from_mlip_network(
        esen_model,
        seed=42,
        required_properties=Properties(stress=True, partial_charges=True),
    )
    return ForceField(
        esen_ff.predictor,
        esen_ff.params,
    )


@pytest.fixture(scope="session")
def mace_force_field_v1(mace_config, dataset_info):
    mace_kwargs = mace_config.model_dump(mode="json")
    num_bessel = mace_kwargs.pop("num_rbf")
    mace_kwargs.pop("radial_mlp_hidden")
    mace_kwargs.pop("radial_mlp_activation")
    symmetric_contraction_backend = mace_kwargs.pop("symmetric_contraction_backend")
    symmetric_basis = symmetric_contraction_backend == "e3nn_symmetric"
    mace_config_v1 = models_v1.Mace.Config(
        **mace_kwargs,
        num_bessel=num_bessel,
        symmetric_tensor_product_basis=symmetric_basis,
    )
    mace_model = models_v1.Mace(mace_config_v1, dataset_info)
    mace_predictor = models_v1.ForceFieldPredictorV1(
        required_properties=Properties(stress=True),
        mlip_network=mace_model,
    )
    params = mace_predictor.init(KEY, get_dummy_graph())
    params = standardize_parameters(params)
    return ForceField(mace_predictor, params)


@pytest.fixture(scope="session")
def visnet_force_field_v1(visnet_config, dataset_info):
    visnet_config_v1 = models_v1.Visnet.Config(**visnet_config.model_dump(mode="json"))
    visnet_model = models_v1.Visnet(visnet_config_v1, dataset_info)
    visnet_predictor = models_v1.ForceFieldPredictorV1(
        required_properties=Properties(stress=True),
        mlip_network=visnet_model,
    )
    params = visnet_predictor.init(KEY, get_dummy_graph())
    params = standardize_parameters(params)
    return ForceField(visnet_predictor, params)


@pytest.fixture(scope="session")
def nequip_force_field_v1(nequip_config, dataset_info):
    nequip_kwargs = nequip_config.model_dump(mode="json")
    num_bessel = nequip_kwargs.pop("num_rbf")
    mlp_hidden = nequip_kwargs.pop("radial_mlp_hidden")
    nequip_kwargs["node_irreps"] = nequip_kwargs.pop("target_irreps")
    nequip_kwargs["radial_net_nonlinearity"] = nequip_kwargs.pop(
        "radial_mlp_activation"
    )
    nequip_kwargs["radial_net_n_hidden"] = mlp_hidden[0]
    nequip_kwargs["radial_net_n_layers"] = len(mlp_hidden)
    nequip_kwargs["scalar_mlp_std"] = nequip_kwargs.pop("radial_mlp_variance_scale")
    nequip_config_v1 = models_v1.Nequip.Config(**nequip_kwargs, num_bessel=num_bessel)
    nequip_model = models_v1.Nequip(nequip_config_v1, dataset_info)
    nequip_predictor = models_v1.ForceFieldPredictorV1(
        required_properties=Properties(stress=True),
        mlip_network=nequip_model,
    )
    params = nequip_predictor.init(KEY, get_dummy_graph())
    params = standardize_parameters(params)
    return ForceField(nequip_predictor, params)


class QuadraticMLIP(MLIPNetwork):
    """A simple energy model with quadratic interaction potentials.

    Should be reused to isolate tests on `ForceFieldPredictor` variants
    from our larger `MLIPNetwork` architectures.

    Its simple form also allows for numerical checks, e.g. on Hessian predictions.
    """

    class Config(MLIPNetworkConfig):
        stiffness: list[float]
        length: list[float]
        use_coulomb_term: bool = False
        predict_partial_charges: bool = False
        use_total_charge_embedding: bool = False

    config: Config
    dataset_info: DatasetInfo
    available_properties: Properties = Properties(
        energy=True,
        forces=True,
        stress=True,
        hessian=True,
    )

    def setup(self):
        self.stiffness_scaling = self.param(
            "stiffness_scaling", nn.initializers.ones, (len(self.config.stiffness),)
        )

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        graph = SpeciesAssignmentBlock(self.dataset_info)(graph)
        node_features = self._compute_node_energies(
            graph.edge_vectors(),
            graph.nodes.features["species"],
            graph.senders,
            graph.receivers,
        )
        return graph.replace_nodes(
            features={"energy": node_features * graph.node_mask()},
        )

    def _compute_node_energies(self, vectors, species, senders, receivers):
        stiffness = jnp.array(self.config.stiffness) * self.stiffness_scaling
        length = jnp.array(self.config.length)
        specie = species[senders]
        rij = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1) + 1e-8)
        spring_terms = 0.5 * stiffness[specie] * (rij - length[specie]) ** 2
        node_energies = jnp.zeros(species.shape[0])
        node_energies = node_energies.at[receivers].add(spring_terms)
        return node_energies


@pytest.fixture(scope="session")
def quadratic_mlip(dataset_info) -> MLIPNetwork:
    num_species = len(dataset_info.allowed_atomic_numbers)
    cfg = QuadraticMLIP.Config(
        # Note that stiffness values that are significantly larger than 0.003 will
        # cause rapid implosion of the system during simulation, and hence lead to
        # issues with neighbor list reallocation when using reasonable values for
        # "edge_capacity_multiplier". Thus, expect simulation tests to fail when
        # changing the values below.
        stiffness=[0.003] * num_species,
        length=[0.87] * num_species,
    )
    return QuadraticMLIP(cfg, dataset_info)


@pytest.fixture(scope="session")
def quadratic_force_field(quadratic_mlip) -> ForceField:
    required_properties = Properties(stress=True)
    return ForceField.from_mlip_network(quadratic_mlip, required_properties)


@pytest.fixture(scope="session")
def quadratic_hessian_force_field(quadratic_mlip):
    """`ForceField` using `ConservativePredictor` for (energy, forces, stress)."""
    force_field = ForceField.from_mlip_network(
        quadratic_mlip,
        Properties(energy=True, forces=True, stress=False, hessian=True),
        seed=2,
    )
    return force_field


@pytest.fixture(scope="session")
def pad_graph() -> Callable[[Graph, int, int, int], Graph]:
    """A fixture to pad a single graph with dummy graphs."""

    def pad(
        graph: Graph,
        n_graph: int,
        max_n_node: int,
        max_n_edge: int,
    ) -> Graph:

        def pad_feature(n_tot: int, x: jax.Array | None):
            if x is None:
                return None
            pad_shape = (n_tot - x.shape[0], *x.shape[1:])
            zeros = jnp.zeros_like(x, shape=pad_shape)
            return jnp.concat([x, zeros], axis=0)

        n_graph_in = graph.num_graphs

        n_node_in = jnp.sum(graph.n_node)
        n_edge_in = jnp.sum(graph.n_edge)

        # first dummy node repeated on senders and receivers
        dummy_node = jnp.full(max_n_node - n_node_in, n_node_in)
        # first dummy graph has all padding nodes and edges
        dummy_n_node = (max_n_node - n_node_in)[None]
        dummy_n_edge = (max_n_edge - n_edge_in)[None]

        dummies_empty = jnp.full(n_graph - n_graph_in - 1, 0)

        return Graph(
            nodes=jax.tree.map(partial(pad_feature, max_n_node), graph.nodes),
            edges=jax.tree.map(partial(pad_feature, max_n_edge), graph.edges),
            globals=jax.tree.map(partial(pad_feature, n_graph), graph.globals),
            n_node=jnp.concat([graph.n_node, dummy_n_node, dummies_empty]),
            n_edge=jnp.concat([graph.n_edge, dummy_n_edge, dummies_empty]),
            senders=jnp.concat([graph.senders, dummy_node]),
            receivers=jnp.concat([graph.receivers, dummy_node]),
        )

    return pad
