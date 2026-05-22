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

import jax.numpy as jnp
import numpy as np
import pytest

from mlip.data import ChemicalSystem
from mlip.graph import Graph
from mlip.models.predictors import ForceFieldPredictor
from mlip.typing.properties import Properties


class DummyForceFieldPredictor(ForceFieldPredictor):
    """A dummy ForceFieldPredictor that implements the abstract methods."""

    def __call__(self, graph: Graph) -> Graph:
        return graph

    def compute_forces_and_stress(self, positions, strains, graph):
        return jnp.zeros_like(positions), graph


@pytest.fixture(scope="session")
def quadratic_energy_predictor(quadratic_mlip):
    """Return a dummy predictor for the quadratic MLIP (energy-only, no forces)."""
    return DummyForceFieldPredictor(
        mlip_network=quadratic_mlip,
        required_properties=Properties(energy=True, forces=False),
    )


@pytest.fixture(scope="session")
def salt_graph() -> Graph:
    """Mimic the `salt_graph` fixture from `conftest.py` as a `Graph`."""
    salt_system = ChemicalSystem(
        atomic_numbers=np.array([11, 17]),
        atomic_species=np.array([0, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.6, 0.5, 0.5]]),
        cell=np.eye(3),
        pbc=(True, True, True),
    )
    return Graph.from_chemical_system(salt_system, 0.95)


def test_compute_energy(quadratic_energy_predictor, quadratic_force_field, salt_graph):
    """Test ForceFieldPredictor.compute_energy computes and aggregates energies."""
    mlip_network = quadratic_energy_predictor.mlip_network

    predictor_params = quadratic_force_field.params
    mlip_params = {"params": predictor_params["params"]["mlip_network"]}

    strains = jnp.zeros_like(salt_graph.globals.cell)
    total_energy, _ = quadratic_energy_predictor.apply(
        predictor_params,
        salt_graph.nodes.positions,
        strains,
        salt_graph,
        method=quadratic_energy_predictor.compute_energy,
    )

    # Compute expected energy using the MLIP network and energy_head
    graph_before_energy_head = mlip_network.apply(mlip_params, salt_graph)
    expected_energy = float(
        jnp.sum(quadratic_energy_predictor._energy_head(graph_before_energy_head))
    )

    np.testing.assert_allclose(float(total_energy), expected_energy, rtol=1e-5)
