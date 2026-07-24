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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md.dataclasses import dataclass as jax_compatible_dataclass

from mlip.simulation.fep.sampler.utils.replica_exchange import (
    _metropolis_criterion,  # noqa: PLC2701
    _swap_lambdas,  # noqa: PLC2701
    perform_replica_exchange,
)
from mlip.simulation.jax_md.helpers import KCAL_PER_MOL_PER_ELECTRON_VOLT


@jax_compatible_dataclass
class MockSystemState:
    """Minimal stand-in for `FEPSystemState`."""

    lambda_values: float
    label: str


@jax_compatible_dataclass
class MockSimulationState:
    """Minimal stand-in for `JaxMDSimulationState`."""

    system_state: MockSystemState
    steps_completed: int
    label: str


@dataclass
class MetropolisCriterionTest:
    """Test case for the metropolis criterion."""

    up_i: float
    down_j: float
    prob: float
    curr_i: float = 1.0
    curr_j: float = 1.0
    temp: float = 300.0


@pytest.mark.parametrize(
    "metropolis_criterion_test_case",
    [
        MetropolisCriterionTest(up_i=0.0, down_j=0.0, prob=1.0),
        MetropolisCriterionTest(up_i=1.0, down_j=1.0, prob=1.0),
        MetropolisCriterionTest(up_i=10.0, down_j=10.0, prob=0.0),
        MetropolisCriterionTest(up_i=0.5, down_j=2.0, prob=0.43227),
        MetropolisCriterionTest(up_i=0.5, down_j=2.0, temp=1000.0, prob=0.77755),
    ],
)
def test_metropolis_criterion(
    metropolis_criterion_test_case: MetropolisCriterionTest,
) -> None:
    """Test that the metropolis criterion is computed correctly."""
    case = metropolis_criterion_test_case
    cycle_energies_i = {"current": case.curr_i, "up": case.up_i, "down": None}
    cycle_energies_j = {"current": case.curr_j, "up": None, "down": case.down_j}
    _, prob = _metropolis_criterion(cycle_energies_i, cycle_energies_j, case.temp)
    assert np.isclose(prob, case.prob, atol=1e-5)


def test_swap_lambdas_fixes_lambda_values() -> None:
    """Test that swapping moves configurations while lambda values stay put."""
    state_a = MockSimulationState(
        system_state=MockSystemState(lambda_values=0.0, label="a"),
        steps_completed=0,
        label="a",
    )
    state_b = MockSimulationState(
        system_state=MockSystemState(lambda_values=1.0, label="b"),
        steps_completed=100,
        label="b",
    )

    new_states = _swap_lambdas(0, 1, [state_a, state_b])

    assert [s.label for s in new_states] == ["b", "a"]
    assert [s.steps_completed for s in new_states] == [100, 0]

    assert [s.system_state.label for s in new_states] == ["b", "a"]
    assert [s.system_state.lambda_values for s in new_states] == [0.0, 1.0]


@pytest.mark.parametrize(
    "episode_idx, expected_pair, expect_accept",
    [(0, (0, 1), True), (1, (1, 2), True), (0, (0, 1), False)],
)
def test_perform_replica_exchange(
    episode_idx: int, expected_pair: tuple[int, int], expect_accept: bool
) -> None:
    """Test that the correct pair is attempted, and accept/reject is applied."""
    num_lambdas = 3
    latest_repex_energies = {i: np.zeros(num_lambdas) for i in range(num_lambdas)}
    repex_seed = 0

    if not expect_accept:
        curr_i, up_i, curr_j, down_j = 1.0, 0.5, 1.0, 2.0
        i, j = expected_pair
        # `_extract_lambda_energy` multiplies by this factor, so divide it out.
        latest_repex_energies[i][i] = curr_i / KCAL_PER_MOL_PER_ELECTRON_VOLT
        latest_repex_energies[i][j] = up_i / KCAL_PER_MOL_PER_ELECTRON_VOLT
        latest_repex_energies[j][j] = curr_j / KCAL_PER_MOL_PER_ELECTRON_VOLT
        latest_repex_energies[j][i] = down_j / KCAL_PER_MOL_PER_ELECTRON_VOLT
        repex_seed = 2

    lambda_internal_states = [
        MockSimulationState(
            system_state=MockSystemState(lambda_values=float(i), label=f"replica_{i}"),
            steps_completed=0,
            label=f"replica_{i}",
        )
        for i in range(num_lambdas)
    ]

    new_states, log, _ = perform_replica_exchange(
        latest_repex_energies=latest_repex_energies,
        episode_idx=episode_idx,
        temperature_schedule=lambda _: 300.0,
        lambda_internal_states=lambda_internal_states,
        lambda_values=jnp.arange(num_lambdas),
        repex_rng=jax.random.PRNGKey(repex_seed),
        replica_exchange_log=[],
    )

    assert [(entry["lambda_i"], entry["lambda_j"]) for entry in log] == [expected_pair]
    assert [bool(entry["accept"]) for entry in log] == [expect_accept]
    if not expect_accept:
        assert 0.0 < float(log[0]["acceptance_probability"]) < 1.0

    i, j = expected_pair
    if expect_accept:
        assert new_states[i].label == f"replica_{j}"
        assert new_states[j].label == f"replica_{i}"
    else:
        assert new_states[i].label == f"replica_{i}"
        assert new_states[j].label == f"replica_{j}"

    assert new_states[i].system_state.lambda_values == float(i)
    assert new_states[j].system_state.lambda_values == float(j)

    (untouched,) = {0, 1, 2} - {i, j}
    assert new_states[untouched].label == f"replica_{untouched}"
