import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from mlip.data import ChemicalSystem
from mlip.data.helpers import create_graph_from_chemical_system
from mlip.models import ForceField


def _salt_graph_from_positions(
    positions: np.ndarray, cell_length: float = 1.0
) -> jraph.GraphsTuple:
    """Mimic the salt_graph fixture with custom positions and cell."""
    salt = ChemicalSystem(
        atomic_numbers=np.array([11, 17]),
        atomic_species=np.array([0, 1]),
        positions=positions,
        cell=cell_length * np.eye(3),
        pbc=(True, True, True),
    )
    return create_graph_from_chemical_system(
        salt, 0.95, batch_it_with_minimal_dummy=True
    )


@pytest.mark.parametrize(
    "distance,is_positive,is_zero",
    [(0.8, True, False), (0.9, False, False), (0.87, False, True)],
)
def test_potential_pressure_sign(
    quadratic_mlip, distance: float, is_positive: bool, is_zero: bool
):
    """Assert sign of the potential pressure is consistent with atomic distances.

    Energy minimum of `quadratic_mlip` is at distance = 0.87.
    We use a cell length of 2.0 to prevent the atoms from "seeing themselves",
    such that we can easily infer the expected sign of the potential pressure.
    """
    mlip_network = quadratic_mlip
    predictor = ForceField.from_mlip_network(mlip_network, seed=2, predict_stress=True)

    positions = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    graph = _salt_graph_from_positions(positions, cell_length=2.0)
    prediction = predictor(graph).pressure
    assert (prediction[0] > 0) == is_positive
    assert (prediction[0] == 0) == is_zero


def test_stress_is_translation_invariant(quadratic_mlip):
    """Assert stress is invariant under translation over the cell boundary."""
    mlip_network = quadratic_mlip
    predictor = ForceField.from_mlip_network(mlip_network, seed=2, predict_stress=True)

    base_positions = np.array([[0.0, 0.0, 0.0], [0.6, 0.5, 0.5]])
    base_graph = _salt_graph_from_positions(base_positions)

    # Translate system such that 2nd node is translated over the boundary and wrapped.
    translation = np.array([0.8, 0.0, 0.0])
    translated_graph = _salt_graph_from_positions((base_positions + translation) % 1.0)

    pred_base = predictor(base_graph).stress
    pred_translated = predictor(translated_graph).stress
    assert jnp.allclose(pred_base[0], pred_translated[0], atol=1e-5, rtol=1e-5)


def test_stress_is_symmetric(salt_graph, quadratic_mlip):
    """Assert stress tensor is symmetric."""
    graph = salt_graph
    mlip_network = quadratic_mlip
    predictor = ForceField.from_mlip_network(mlip_network, seed=2, predict_stress=True)
    stress = predictor(graph).stress
    assert jnp.allclose(stress[0], stress[0].transpose(), atol=1e-5, rtol=1e-5)
