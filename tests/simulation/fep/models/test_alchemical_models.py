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

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mlip.models import Visnet
from mlip.models.force_field import ForceField
from mlip.simulation.fep.alchemical_graph import AlchemicalGraph
from mlip.simulation.fep.models import (
    AlchemicalEsen,
    AlchemicalForceField,
    AlchemicalMace,
    AlchemicalNequip,
    AlchemicalVisnet,
)
from mlip.typing.properties import Properties

_DEFAULT_BOUNDARY_TOL = dict(rtol=1e-5, atol=1e-6)

MODEL_CASES = {
    "mace": (
        AlchemicalMace,
        "mace_config",
        "mace_force_field",
        dict(rtol=1e-4, atol=1e-3),
    ),
    "nequip": (
        AlchemicalNequip,
        "nequip_config",
        "nequip_force_field",
        _DEFAULT_BOUNDARY_TOL,
    ),
    "visnet": (
        AlchemicalVisnet,
        "small_valid_visnet_config",
        "small_valid_visnet_force_field",
        _DEFAULT_BOUNDARY_TOL,
    ),
    "esen": (
        AlchemicalEsen,
        "esen_config",
        "esen_force_field",
        dict(rtol=2e-4, atol=5e-3),
    ),
}

_EXPECTED_ENERGIES = {
    ("mace", 1.0): -19.46984,
    ("mace", 0.75): -40.012615,
    ("mace", 0.5): -41.680176,
    ("mace", 0.25): -42.50582,
    ("mace", 0.0): -43.214268,
    ("nequip", 1.0): 42.094654,
    ("nequip", 0.75): 48.673145,
    ("nequip", 0.5): 56.060966,
    ("nequip", 0.25): 63.40018,
    ("nequip", 0.0): 67.899216,
    ("visnet", 1.0): -42.801548,
    ("visnet", 0.75): -42.801544,
    ("visnet", 0.5): -42.801521,
    ("visnet", 0.25): -42.801422,
    ("visnet", 0.0): -42.799911,
    ("esen", 1.0): 1722.3068,
    ("esen", 0.75): 1699.5587,
    ("esen", 0.5): 1661.8357,
    ("esen", 0.25): 1643.6152,
    ("esen", 0.0): 1446.4893,
}


@pytest.fixture(scope="module", params=list(MODEL_CASES), ids=list(MODEL_CASES))
def model_case(request):
    return (request.param, *MODEL_CASES[request.param])


@pytest.fixture(scope="module")
def small_valid_visnet_config(visnet_config):
    """Config with a `vecnorm_type` accepted by `AlchemicalVisnet`."""
    return visnet_config.model_copy(
        update={
            "vecnorm_type": "none",
            "num_layers": 1,
            "num_channels": 2,
            "num_heads": 1,
            "l_max": 1,
            "num_rbf": 2,
        }
    )


@pytest.fixture(scope="module")
def small_valid_visnet_force_field(
    small_valid_visnet_config, dataset_info
) -> ForceField:
    visnet_model = Visnet(small_valid_visnet_config, dataset_info)
    return ForceField.from_mlip_network(
        visnet_model, seed=42, required_properties=Properties(stress=True)
    )


@pytest.fixture(scope="module")
def base_force_field(request, model_case: tuple) -> ForceField:
    """The plain (non-alchemical) force field for the current `model_case`."""
    _, _, _, force_field_fixture, _ = model_case
    return request.getfixturevalue(force_field_fixture)


@pytest.fixture(scope="module")
def alchemical_force_field(
    request, model_case: tuple, dataset_info, base_force_field: ForceField
) -> AlchemicalForceField:
    """`AlchemicalForceField` wrapping the alchemical variant for `model_case`."""
    _, alchemical_cls, config_fixture, _, _ = model_case
    config = request.getfixturevalue(config_fixture)
    alchemical_model = alchemical_cls(config, dataset_info)
    alchemical_ff = AlchemicalForceField.from_mlip_network(
        mlip_network=alchemical_model,
        required_properties=Properties(stress=True),
    )
    return AlchemicalForceField(alchemical_ff.predictor, base_force_field.params)


@pytest.mark.parametrize(
    "edge_weight,expected_graph_fixture", [(1.0, "graph_a"), (0.0, "graph_b")]
)
def test_boundary_lambdas_match_base_model(
    request: pytest.FixtureRequest,
    base_force_field: ForceField,
    alchemical_force_field: AlchemicalForceField,
    graph_a: AlchemicalGraph,
    model_case: tuple,
    edge_weight: float,
    expected_graph_fixture: str,
) -> None:
    """Check the alchemical predictions at weights 0 and 1 match the base model."""
    model_name, _, _, _, tol = model_case
    expected_graph = request.getfixturevalue(expected_graph_fixture)
    lam = jax.numpy.asarray(graph_a.globals.alchemical_lambda)
    lam = lam.at[:, 0].set(edge_weight).at[:, 1].set(0.0)
    graph_a = graph_a.replace_globals(alchemical_lambda=lam)

    prediction = alchemical_force_field(graph_a)
    expected_energy = np.array([_EXPECTED_ENERGIES[(model_name, edge_weight)], 0.0])
    assert_allclose(prediction.energy, expected_energy, **tol)

    expected = base_force_field(expected_graph)
    assert_allclose(prediction.energy, expected.energy, **tol)
    assert_allclose(prediction.forces, expected.forces, **tol)
    assert_allclose(prediction.stress, expected.stress, **tol)


@pytest.mark.parametrize("edge_weight", [0.75, 0.5, 0.25])
def test_intermediate_alchemical_energies(
    alchemical_force_field: AlchemicalForceField,
    graph_a: AlchemicalGraph,
    model_case: tuple,
    edge_weight: float,
) -> None:
    """Check the alchemical energy at weights 0 and 1 matches the base model."""
    model_name, _, _, _, tol = model_case
    lam = jax.numpy.asarray(graph_a.globals.alchemical_lambda)
    lam = lam.at[:, 0].set(edge_weight).at[:, 1].set(0.0)
    graph_a = graph_a.replace_globals(alchemical_lambda=lam)

    expected_energy = np.array([_EXPECTED_ENERGIES[(model_name, edge_weight)], 0.0])
    prediction = alchemical_force_field(graph_a)

    assert_allclose(prediction.energy, expected_energy, **tol)
    assert np.all(np.isfinite(prediction.energy))
    assert np.all(np.isfinite(prediction.forces))
    assert np.all(np.isfinite(prediction.stress))
