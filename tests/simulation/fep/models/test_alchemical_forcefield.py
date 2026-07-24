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

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.testing import assert_allclose

import mlip.simulation.fep.models.alchemical_forcefield as alchemical_forcefield_module
from mlip.models.force_field import ForceField
from mlip.models.mlip_network import MLIPNetwork
from mlip.simulation.fep.alchemical_graph import AlchemicalGraph
from mlip.simulation.fep.alchemical_graph.masking import (
    get_alchemical_edge_mask,
    prune_edges_with_mask,
)
from mlip.simulation.fep.enums import FEPStage
from mlip.simulation.fep.models.alchemical_forcefield import AlchemicalForceField
from mlip.simulation.fep.models.alchemical_predictor import (
    LinearAlchemicalPredictorV1,
)
from mlip.typing.properties import Properties

LAMBDA_VALUES = [
    [0.0, 0.0],
    [0.1, 0.1],
    [0.5, 0.2],
    [1.0, 1.0],
    [1.0, 0.2],
    [1.0, 0.1],
    [1.0, 0.0],
]


@pytest.fixture(scope="module")
def base_forcefield(quadratic_force_field) -> ForceField:
    return quadratic_force_field


@pytest.fixture(scope="module")
def alchemical_forcefield(base_forcefield: ForceField) -> AlchemicalForceField:
    alchemical_ff = AlchemicalForceField.from_mlip_network(
        mlip_network=base_forcefield.predictor.mlip_network,
        required_properties=Properties(),
        fep_stage=FEPStage.AR,
    )
    return AlchemicalForceField(alchemical_ff.predictor, base_forcefield.params)


@pytest.fixture(scope="module")
def graph_rep(graph_a: AlchemicalGraph) -> AlchemicalGraph:
    """Graph for the repulsion term (contains only boundary edges)."""
    sr_mask, _ = get_alchemical_edge_mask(graph_a)
    return prune_edges_with_mask(graph_a, sr_mask)


@pytest.fixture(name="preds_a", scope="module")
def base_prediction_a(base_forcefield: ForceField, graph_a: AlchemicalGraph) -> dict:
    return asdict(base_forcefield(graph_a))


@pytest.fixture(name="preds_b", scope="module")
def base_prediction_b(base_forcefield: ForceField, graph_b: AlchemicalGraph) -> dict:
    return asdict(base_forcefield(graph_b))


def _repulsion_energy(
    positions: np.ndarray,
    strains: np.ndarray,
    repulsion_weight: float,
    alchemical_forcefield: AlchemicalForceField,
    graph_rep: AlchemicalGraph,
) -> tuple[Array, AlchemicalGraph]:
    """Compute the repulsion term via `_compute_repulsion_energy`."""
    graph_rep = graph_rep.replace_globals(
        alchemical_lambda=jnp.asarray(graph_rep.globals.alchemical_lambda)
        .at[:, 1]
        .set(repulsion_weight)
    )
    return alchemical_forcefield.predictor._compute_repulsion_energy(
        positions, strains, graph_rep
    )


def _compute_repulsion_predictions(
    alchemical_forcefield: AlchemicalForceField,
    repulsion_weight: float,
    graph_rep: AlchemicalGraph,
) -> dict[str, np.ndarray]:
    """Compute the repulsion term's energy, forces, stress and pressure."""
    positions = graph_rep.nodes.positions
    strains = jnp.zeros_like(graph_rep.globals.cell)

    (gradients, _), graph_out = jax.grad(_repulsion_energy, (0, 1), has_aux=True)(
        positions, strains, repulsion_weight, alchemical_forcefield, graph_rep
    )
    gradients = gradients.at[-1].set(0.0)

    return {"energy": np.array(graph_out.globals.energy), "forces": -gradients}


@pytest.mark.parametrize("lambda_values", LAMBDA_VALUES)
def test_predictions(
    base_forcefield: ForceField,
    graph_a: AlchemicalGraph,
    graph_rep: AlchemicalGraph,
    preds_a: dict,
    preds_b: dict,
    lambda_values: tuple[float, float],
) -> None:
    """Test that the AlchemicalForceField combines components correctly:

    U(lambda_1, lambda_2) = lambda_1 * U_A + (1 - lambda_1) * U_B + U_R(lambda_2)
    """
    edge_weight, repulsion_weight = lambda_values
    fep_stage = FEPStage.from_edge_weight(edge_weight)
    alchemical_ff = AlchemicalForceField.from_mlip_network(
        mlip_network=base_forcefield.predictor.mlip_network,
        fep_stage=fep_stage,
    )
    alchemical_forcefield = AlchemicalForceField(
        alchemical_ff.predictor, base_forcefield.params
    )

    graph_a = graph_a.replace_globals(
        alchemical_lambda=np.array([list(lambda_values), [0.0, 0.0]])
    )
    prediction = asdict(alchemical_forcefield(graph_a))
    ab_weights = (edge_weight, 1 - edge_weight)

    preds_r = _compute_repulsion_predictions(
        alchemical_forcefield, repulsion_weight, graph_rep
    )
    for key in ["energy", "forces"]:
        pred_a, pred_b, pred_r = (preds_a.get(key), preds_b.get(key), preds_r.get(key))
        expected_pred = ab_weights[0] * pred_a + ab_weights[1] * pred_b + pred_r

        assert_allclose(
            prediction.get(key), expected_pred, rtol=1e-3, err_msg=f"Key: {key}"
        )


@pytest.mark.parametrize("current_lambda", LAMBDA_VALUES)
def test_compute_alchemical_values(
    alchemical_forcefield: AlchemicalForceField,
    graph_a: AlchemicalGraph,
    graph_rep: AlchemicalGraph,
    preds_a: dict,
    preds_b: dict,
    current_lambda: tuple[float, float],
) -> None:
    """compute_alchemical_values returns energies matching the linear combination."""
    graph_a = graph_a.replace_globals(
        alchemical_lambda=np.array([list(current_lambda), [0.0, 0.0]])
    )
    reference_lambdas = np.array(LAMBDA_VALUES)

    per_lambda_energies = alchemical_forcefield.predictor.apply(
        alchemical_forcefield.params,
        graph_a,
        reference_lambdas,
        None,
        None,
        method=alchemical_forcefield.predictor.compute_alchemical_values,
    )

    assert per_lambda_energies.shape == (len(reference_lambdas),)

    # Manually compute expected energies using stored U_A, U_B and repulsion values.
    u_a = np.array(preds_a["energy"])[0]
    u_b = np.array(preds_b["energy"])[0]
    positions = graph_rep.nodes.positions
    strains = jnp.zeros_like(graph_rep.globals.cell)
    expected = []
    for edge_weight, repulsion_weight in reference_lambdas:
        u_r, _ = _repulsion_energy(
            positions, strains, repulsion_weight, alchemical_forcefield, graph_rep
        )
        expected.append(edge_weight * u_a + (1 - edge_weight) * u_b + u_r)
    assert_allclose(per_lambda_energies, np.array(expected), rtol=1e-4)


def test_from_mlip_network_v1(
    quadratic_mlip: MLIPNetwork, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v1 `MLIPNetwork` should select the matching V1-compatible predictor."""
    monkeypatch.setattr(
        alchemical_forcefield_module, "MLIPNetworkV1", type(quadratic_mlip)
    )
    monkeypatch.setattr(
        type(quadratic_mlip), "calculate", lambda self, g: self(g), raising=False
    )

    alchemical_ff = AlchemicalForceField.from_mlip_network(
        mlip_network=quadratic_mlip,
        fep_stage=FEPStage.AR,
    )
    assert type(alchemical_ff.predictor) is LinearAlchemicalPredictorV1
