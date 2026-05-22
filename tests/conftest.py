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

import pickle
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pydantic
import pytest
from ase.io import read as ase_read_atoms

from mlip.data import ChemicalSystem, DatasetInfo
from mlip.data.helpers import create_graph_from_chemical_system
from mlip.models import ForceField, Mace, Nequip, Visnet
from mlip.models.mlip_network import MLIPNetwork
from mlip.simulation.utils import create_graph_from_atoms

CUTOFF_ANGSTROM = 3.0
XYZ_FILE_PATH = Path(__file__).parent / "data" / "Dimethyl_sulfoxide.xyz"
MACE_PARAMS_PICKLE_FILE = Path(__file__).parent / "data" / "mace_test_params.pkl"
VISNET_PARAMS_PICKLE_FILE = Path(__file__).parent / "data" / "visnet_test_params.pkl"
NEQUIP_PARAMS_PICKLE_FILE = Path(__file__).parent / "data" / "nequip_test_params.pkl"


@pytest.fixture(scope="session")
def setup_system():
    atoms = ase_read_atoms(XYZ_FILE_PATH)
    positions = atoms.get_positions()

    senders, receivers = [], []
    for i in range(positions.shape[0]):
        for j in range(positions.shape[0]):
            if i != j and np.linalg.norm(positions[i] - positions[j]) < CUTOFF_ANGSTROM:
                senders.append(i)
                receivers.append(j)
    senders, receivers = np.asarray(senders), np.asarray(receivers)

    def displacement_fun(vec1, vec2):
        return vec1 - vec2

    allowed_z_numbers = {1, 6, 8, 16}
    graph = create_graph_from_atoms(
        atoms,
        senders,
        receivers,
        displacement_fun,
        allowed_atomic_numbers=allowed_z_numbers,
    )
    dataset_info = DatasetInfo(
        atomic_energies_map=dict.fromkeys(allowed_z_numbers, 0),
        avg_num_neighbors=6.8,
        avg_r_min_angstrom=None,
        cutoff_distance_angstrom=CUTOFF_ANGSTROM,
        scaling_mean=0.0,
        scaling_stdev=1.0,
    )

    return atoms, graph, dataset_info


@pytest.fixture(scope="session")
def setup_system_and_mace_model(setup_system):
    atoms, graph, dataset_info = setup_system

    mace_kwargs = {
        "num_layers": 2,
        "num_bessel": 8,
        "radial_envelope": "polynomial_envelope",
        "activation": "silu",
        "num_channels": 4,
        "readout_irreps": ("4x0e", "0e"),
        "correlation": 2,
        "node_symmetry": 2,
        "l_max": 2,
        "symmetric_tensor_product_basis": True,
    }

    mace_model = Mace(Mace.Config(**mace_kwargs), dataset_info)
    mace_ff = ForceField.from_mlip_network(
        mace_model,
        seed=42,
        predict_stress=False,
    )
    mace_initial_params = mace_ff.params

    with MACE_PARAMS_PICKLE_FILE.open("rb") as pkl_file:
        mace_params = pickle.load(pkl_file)

    assert jax.tree.map(np.shape, mace_initial_params) == jax.tree.map(
        np.shape, mace_params
    )

    mace_ff = ForceField(mace_ff.predictor, mace_params)
    mace_apply_fun = jax.jit(mace_ff.predictor.apply)

    return atoms, graph, mace_apply_fun, mace_ff


@pytest.fixture(scope="session")
def setup_system_and_visnet_model(setup_system):
    atoms, graph, dataset_info = setup_system

    visnet_kwargs = {
        "num_layers": 2,
        "num_channels": 8,
        "l_max": 2,
        "num_heads": 2,
        "num_rbf": 4,
        "activation": "silu",
        "attn_activation": "silu",
        "vecnorm_type": "max_min",
    }
    visnet_model = Visnet(Visnet.Config(**visnet_kwargs), dataset_info)
    visnet_ff = ForceField.from_mlip_network(
        visnet_model,
        seed=42,
        predict_stress=False,
    )
    visnet_initial_params = visnet_ff.params

    with VISNET_PARAMS_PICKLE_FILE.open("rb") as pkl_file:
        visnet_params = pickle.load(pkl_file)

    assert jax.tree.map(np.shape, visnet_initial_params) == jax.tree.map(
        np.shape, visnet_params
    )

    visnet_ff = ForceField(visnet_ff.predictor, visnet_params)
    visnet_apply_fun = jax.jit(visnet_ff.predictor.apply)

    return atoms, graph, visnet_apply_fun, visnet_ff


@pytest.fixture(scope="session")
def setup_system_and_nequip_model(setup_system):
    atoms, graph, dataset_info = setup_system

    nequip_kwargs = {
        "num_layers": 2,
        "node_irreps": "4x0e + 4x0o + 4x1o + 4x1e + 4x2e + 4x2o",
        "l_max": 2,
        "num_bessel": 8,
        "radial_net_nonlinearity": "swish",
        "radial_net_n_hidden": 8,
        "radial_net_n_layers": 2,
        "radial_envelope": "polynomial_envelope",
        "scalar_mlp_std": 4.0,
    }
    nequip_model = Nequip(Nequip.Config(**nequip_kwargs), dataset_info)
    nequip_ff = ForceField.from_mlip_network(
        nequip_model,
        seed=42,
        predict_stress=False,
    )
    nequip_initial_params = nequip_ff.params

    with NEQUIP_PARAMS_PICKLE_FILE.open("rb") as pkl_file:
        nequip_params = pickle.load(pkl_file)

    assert jax.tree.map(np.shape, nequip_initial_params) == jax.tree.map(
        np.shape, nequip_params
    )

    nequip_ff = ForceField(nequip_ff.predictor, nequip_params)
    nequip_apply_fun = jax.jit(nequip_ff.predictor.apply)

    return atoms, graph, nequip_apply_fun, nequip_ff


@pytest.fixture(scope="session")
def salt_graph() -> jraph.GraphsTuple:
    # NaCl CFC lattice
    salt = ChemicalSystem(
        atomic_numbers=np.array([11, 17]),
        atomic_species=np.array([0, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.6, 0.5, 0.5]]),
        cell=np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        pbc=(True, True, True),
    )

    # slightly smaller than lattice width
    cutoff = 0.95

    graph = create_graph_from_chemical_system(
        salt,
        cutoff,
        batch_it_with_minimal_dummy=True,
    )

    return graph


class QuadraticMLIP(MLIPNetwork):
    """A simple energy model with quadratic interaction potentials.

    Should be reused to isolate tests on `ForceFieldPredictor` variants
    from our larger `MLIPNetwork` architectures.

    Its simple form also allows for numerical checks, e.g. on Hessian predictions.
    """

    class Config(pydantic.BaseModel):
        stiffness: list[float]
        length: list[float]

    config: Config
    dataset_info: DatasetInfo

    @nn.compact
    def __call__(self, vectors, species, senders, receivers):
        stiffness = jnp.array(self.config.stiffness)
        length = jnp.array(self.config.length)
        specie = species[senders]
        rij = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        spring_terms = 0.5 * stiffness[specie] * (rij - length[specie]) ** 2
        node_energies = jnp.zeros(species.shape[0])
        node_energies = node_energies.at[receivers].add(spring_terms)
        return node_energies


@pytest.fixture(scope="session")
def quadratic_mlip() -> MLIPNetwork:
    dataset_info = DatasetInfo(
        atomic_energies_map={11: 0.0, 17: 0.0},
        cutoff_distance_angstrom=0.95,
    )
    cfg = QuadraticMLIP.Config(
        stiffness=[2.0, 2.0],
        length=[0.87, 0.87],
    )
    return QuadraticMLIP(cfg, dataset_info)
