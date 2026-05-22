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

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mlip.data import ChemicalSystem
from mlip.graph import Graph
from mlip.graph.batching_helpers import pad_with_graphs
from mlip.models.force_field import ForceField
from mlip.models.predictors import ForceFieldPredictor
from mlip.models.predictors.conservative_predictor import ConservativePredictor
from mlip.typing.properties import Properties


def _salt_graph_from_positions(
    positions: np.ndarray, cell_length: float = 1.0
) -> Graph:
    """Helper used to create a salt_graph from positions and cell in tests."""
    salt = ChemicalSystem(
        atomic_numbers=np.array([11, 17]),
        atomic_species=np.array([0, 1]),
        positions=positions,
        cell=cell_length * np.eye(3),
        pbc=(True, True, True),
    )
    return Graph.from_chemical_system(salt, 0.95)


@pytest.fixture(scope="session")
def salt_graph() -> Graph:
    """Mimic the `salt_graph` fixture from `conftest.py` as a `Graph`."""
    positions = np.array([[0.0, 0.0, 0.0], [0.6, 0.5, 0.5]])
    return _salt_graph_from_positions(positions)


def test_compute_energy_not_overridden():
    """`ConservativePredictor` does not override `ForceFieldPredictor.compute_energy`.

    `ForceFieldPredictor.compute_energy` is tested in `test_predictor.py`.
    """
    assert ConservativePredictor.compute_energy is ForceFieldPredictor.compute_energy


def test_compute_forces_and_stress(quadratic_force_field, salt_graph):
    out_graph = jax.jit(quadratic_force_field.calculate)(salt_graph)
    forces = out_graph.nodes.forces
    stress = out_graph.globals.stress
    pressure = out_graph.globals.pressure
    assert forces is not None and forces.shape == salt_graph.nodes.positions.shape
    assert stress is not None and stress.shape == (1, 3, 3)
    assert pressure is not None and pressure.shape == (1,)


def test_single_predictions_consistent_with_padding(quadratic_force_field, salt_graph):
    """Assert predictions are consistent on a single graph with/without padding."""
    num_nodes = salt_graph.n_node[0]
    num_edges = salt_graph.n_edge[0]

    # Add minimal padding
    padded_salt_graph = pad_with_graphs(
        salt_graph,
        n_node=num_nodes + 1,
        n_edge=num_edges + 1,
        n_graph=2,
    )

    jit_ff = jax.jit(quadratic_force_field)
    graph_prediction = jit_ff(salt_graph)
    padded_graph_prediction = jit_ff(padded_salt_graph)

    assert graph_prediction.energy.shape == (1,)
    assert graph_prediction.forces.shape == (num_nodes, 3)
    assert graph_prediction.stress.shape == (1, 3, 3)
    assert graph_prediction.pressure.shape == (1,)

    assert padded_graph_prediction.energy.shape == (2,)
    assert padded_graph_prediction.forces.shape == (num_nodes + 1, 3)
    assert padded_graph_prediction.stress.shape == (2, 3, 3)
    assert padded_graph_prediction.pressure.shape == (2,)

    assert jnp.allclose(
        graph_prediction.energy, padded_graph_prediction.energy[:1], atol=1e-5
    )
    assert jnp.allclose(
        graph_prediction.forces, padded_graph_prediction.forces[:num_nodes], atol=1e-5
    )
    assert jnp.allclose(
        graph_prediction.stress, padded_graph_prediction.stress[:1], atol=1e-5
    )
    assert jnp.allclose(
        graph_prediction.pressure, padded_graph_prediction.pressure[:1], atol=1e-5
    )


def test_forces_equal_negative_energy_gradient(quadratic_force_field, salt_graph):
    """Assert forces are negative gradients of the energy with respect to positions."""
    jit_ff = jax.jit(quadratic_force_field)

    def energy_fn(positions):
        new_graph = salt_graph.replace_nodes(positions=positions)
        return jnp.squeeze(jit_ff(new_graph).energy)

    positions = salt_graph.nodes.positions
    grad_fn = jax.jit(jax.grad(energy_fn))
    energy_grads = grad_fn(positions)

    assert energy_grads.shape == positions.shape
    assert jnp.any(jnp.abs(energy_grads) > 0)

    predicted_forces = jit_ff(salt_graph).forces
    assert energy_grads.shape == predicted_forces.shape
    assert jnp.allclose(predicted_forces, -energy_grads, atol=1e-5, rtol=1e-5)


def test_stress_is_translation_invariant(quadratic_force_field):
    """Assert stress is invariant under translation over the cell boundary."""
    base_positions = np.array([[0.0, 0.0, 0.0], [0.6, 0.5, 0.5]])
    base_graph = _salt_graph_from_positions(base_positions)

    # Translate system such that 2nd node is translated over the boundary and wrapped.
    translation = np.array([0.8, 0.0, 0.0])
    translated_graph = _salt_graph_from_positions((base_positions + translation) % 1.0)

    jit_ff = jax.jit(quadratic_force_field)
    pred_base = jit_ff(base_graph).stress
    pred_translated = jit_ff(translated_graph).stress
    assert jnp.allclose(pred_base[0], pred_translated[0], atol=1e-5, rtol=1e-5)


def test_stress_is_symmetric(quadratic_force_field, salt_graph):
    stress = jax.jit(quadratic_force_field)(salt_graph).stress
    assert jnp.allclose(stress[0], stress[0].transpose(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "distance,is_positive,is_zero",
    [(0.8, True, False), (0.9, False, False), (0.87, False, True)],
)
def test_pressure_sign(
    quadratic_force_field, distance: float, is_positive: bool, is_zero: bool
):
    """Assert sign of the pressure is consistent with atomic distances.

    Energy minimum of `quadratic_mlip` is at distance = 0.87.
    We use a cell length of 2.0 to prevent the atoms from "seeing themselves",
    such that we can easily infer the expected sign of the potential pressure.
    """
    positions = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    graph = _salt_graph_from_positions(positions, cell_length=2.0)
    prediction = jax.jit(quadratic_force_field)(graph).pressure
    assert (prediction[0] > 0) == is_positive
    assert (prediction[0] == 0) == is_zero


@pytest.fixture
def force_field_energy_only(quadratic_mlip):
    """`ForceField` using `ConservativePredictor` for (energy)."""
    force_field = ForceField.from_mlip_network(
        quadratic_mlip,
        Properties(energy=True, forces=False, stress=False),
        seed=2,
    )
    return force_field


def test_only_energy_required(quadratic_force_field, force_field_energy_only):
    assert quadratic_force_field.predictor._only_energy_required() is False
    assert force_field_energy_only.predictor._only_energy_required() is True


def test_only_energy_required_runs_no_grad(force_field_energy_only, salt_graph):
    with patch("jax.grad") as mock_grad:
        force_field_energy_only.calculate(salt_graph)
    mock_grad.assert_not_called()
