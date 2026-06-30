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
import jax_md
import pytest
from jax import random
from jax_md import dataclasses

from mlip.simulation.jax_md.auxiliary_properties import AuxiliaryProperties
from mlip.simulation.jax_md.helpers import (
    KCAL_PER_MOL_PER_ELECTRON_VOLT,
    PRESSURE_CONVERSION_FACTOR,
    TEMPERATURE_CONVERSION_FACTOR,
    TIMESTEP_CONVERSION_FACTOR,
)
from mlip.simulation.jax_md.npt_montecarlo_langevin import (
    NPTLangevinState,
    apply_montecarlo_barostat,
    npt_montecarlo_langevin,
)
from mlip.simulation.montecarlo_barostat import (
    INITIAL_MAX_DELTA_VOLUME_FRACTION,
    _box_to_volume,  # noqa: PLC2701
)

TEMPERATURE = 300.0 * TEMPERATURE_CONVERSION_FACTOR
PRESSURE = 1.01325 * PRESSURE_CONVERSION_FACTOR
MOLECULE_INDICES = jnp.array([0] * 4 + [1] * 6)
MOL_COUNTS = jnp.array([4, 6])
POSITIONS = jax.random.uniform(random.PRNGKey(42), (10, 3), minval=-10.0, maxval=10.0)
BOX = jnp.array([10.0, 10.0, 10.0])
BAROSTAT_INTERVAL = 5


@pytest.fixture
def rng_key():
    return random.PRNGKey(42)


def _create_simple_energy_fn():
    """Create a simple harmonic energy function for testing."""

    def energy_fn(positions, box=None, **kwargs):
        return jnp.sum(positions**2) * KCAL_PER_MOL_PER_ELECTRON_VOLT

    return energy_fn


@pytest.fixture
def init_and_step_fn():
    """Create init_fn and step_fn for the `npt_mc_langevin` simulator."""
    _, shift_fn = jax_md.space.periodic_general(
        BOX, fractional_coordinates=False, wrapped=False
    )

    energy_fn = _create_simple_energy_fn()

    def force_fn(positions, box=None, **kwargs):
        return (
            -jax.grad(lambda p: energy_fn(p, box=box, **kwargs))(positions),
            AuxiliaryProperties(energy=energy_fn(positions, box=box, **kwargs)),
        )

    init_fn, step_fn = npt_montecarlo_langevin(
        langevin_force_fn=force_fn,
        barostat_energy_fn=energy_fn,
        shift_fn=shift_fn,
        dt=1.0 * TIMESTEP_CONVERSION_FACTOR,
        kT=TEMPERATURE,
        pressure=PRESSURE,
        molecule_indices=MOLECULE_INDICES,
        barostat_interval=BAROSTAT_INTERVAL,
    )
    return init_fn, step_fn


@pytest.fixture
def npt_state(init_and_step_fn, rng_key):
    init_fn, _ = init_and_step_fn
    init_state = init_fn(rng_key, POSITIONS, BOX, mass=1.0)
    return init_state


def _create_mock_energy_fn(energy_old: float, energy_new: float):
    """Create a mock energy function that returns energies based on call order."""
    call_count = [0]

    def energy_fn(positions, box=None, **kwargs):
        energy = energy_old if call_count[0] == 0 else energy_new
        call_count[0] += 1
        return jnp.array(energy * KCAL_PER_MOL_PER_ELECTRON_VOLT)

    return energy_fn


@pytest.mark.parametrize(
    "energy_delta, volume_delta",
    [
        (0.001, -100.0),
        (0.001, 100.0),
        (-0.1, -100.0),
        (0.1, 100.0),
        (1.0, -100.0),
        (1.0, 100.0),
    ],
)
def test_apply_montecarlo_barostat_integration(
    init_and_step_fn, rng_key, energy_delta, volume_delta
):
    """Test that `apply_montecarlo_barostat` updates the state."""
    init_fn, _ = init_and_step_fn
    npt_state = init_fn(rng_key, POSITIONS, BOX, mass=1.0)

    patch_path = "mlip.simulation.jax_md.npt_montecarlo_langevin"

    for if_accept in [True, False]:
        attempted_old = npt_state.barostat_state.num_attempted
        accepted_old = npt_state.barostat_state.num_accepted
        positions_old = npt_state.position.copy()
        box_old = npt_state.box.copy()
        volume_old = _box_to_volume(box_old, 3)
        volume_new = volume_old + volume_delta
        length_scale = (volume_new / volume_old) ** (1 / 3)
        forces_old = npt_state.force.copy()

        box_new = box_old * length_scale
        pos_new = positions_old * length_scale

        energy_fn = _create_mock_energy_fn(0.0, energy_delta)

        def mock_propose(state, box, pos):
            return state, volume_old, volume_new, box_new, pos_new

        def mock_accept(state, e_old, e_new, v_old, v_new, kT, n_mol):
            new_state = dataclasses.replace(
                state,
                num_attempted=state.num_attempted + 1,
                num_attempted_since_tune=state.num_attempted_since_tune + 1,
                num_accepted=state.num_accepted + int(if_accept),
                num_accepted_since_tune=state.num_accepted_since_tune + int(if_accept),
            )
            return new_state, if_accept

        def mock_tune(state, volume_old):
            new_state = dataclasses.replace(
                state,
                num_attempted_since_tune=0,
                num_accepted_since_tune=0,
            )
            return new_state

        mock_force_fn = lambda pos, box=None, **kw: (  # noqa: E731
            jnp.zeros_like(pos),
            AuxiliaryProperties(energy=jnp.array([0, 0])),
        )

        with (
            patch(patch_path + ".propose_volume_change", side_effect=mock_propose),
            patch(patch_path + ".accept_volume_change", side_effect=mock_accept),
            patch(patch_path + ".tune_barostat", side_effect=mock_tune),
        ):
            new_state = apply_montecarlo_barostat(
                npt_state, energy_fn, mock_force_fn, kT=1.0
            )

        if if_accept:
            expected_box = box_new
            expected_positions = pos_new
            updated_forces = True
        else:
            expected_box = box_old
            expected_positions = positions_old
            updated_forces = False
        expected_num_accepted = accepted_old + int(if_accept)

        assert new_state.barostat_state.num_attempted == attempted_old + 1
        assert new_state.barostat_state.num_accepted == expected_num_accepted
        assert new_state.barostat_state.num_attempted_since_tune == 0
        assert new_state.barostat_state.num_accepted_since_tune == 0
        assert jnp.allclose(new_state.box, expected_box)
        assert jnp.allclose(new_state.position, expected_positions)
        if updated_forces:
            assert not jnp.allclose(new_state.force, forces_old)
        else:
            assert jnp.allclose(new_state.force, forces_old)


def test_npt_mc_langevin_init_fn(init_and_step_fn, rng_key):
    """Test that init_fn properly initializes an NPTLangevinState."""
    init_fn, _ = init_and_step_fn
    state = init_fn(rng_key, POSITIONS, BOX, mass=1.0)

    assert isinstance(state, NPTLangevinState)
    assert state.langevin_state is not None
    assert jnp.array_equal(state.box, BOX)
    assert state.step_count == 0
    assert jnp.allclose(state.position, POSITIONS)

    bs = state.barostat_state
    assert bs.target_pressure == PRESSURE
    assert bs.num_attempted == 0
    assert bs.num_accepted == 0
    assert jnp.array_equal(bs.mol_counts, MOL_COUNTS)
    # Check max_delta_volume is 1% of initial volume
    initial_volume = jnp.prod(BOX)
    expected_max_delta = initial_volume * INITIAL_MAX_DELTA_VOLUME_FRACTION
    assert jnp.isclose(bs.max_delta_volume, expected_max_delta, rtol=1e-10)


def test_npt_mc_langevin_step_fn(init_and_step_fn, rng_key):
    """Test that step_fn properly advances the simulation state."""
    init_fn, step_fn = init_and_step_fn
    state = init_fn(rng_key, POSITIONS, BOX, mass=1.0)

    initial_step_count = state.step_count
    new_state = step_fn(state)

    assert isinstance(new_state, NPTLangevinState)
    assert new_state.step_count == initial_step_count + 1
    assert new_state.langevin_state is not None
    assert new_state.barostat_state is not None
    assert new_state.barostat_state.num_attempted == 0

    # Positions updated, box unchanged
    assert not jnp.allclose(new_state.position, POSITIONS)
    assert jnp.allclose(new_state.box, BOX)


def test_npt_mc_langevin_multiple_steps(init_and_step_fn, rng_key):
    """Test running multiple steps of the NPT MC Langevin simulation."""
    init_fn, step_fn = init_and_step_fn
    state = init_fn(rng_key, POSITIONS, BOX, mass=1.0)

    for i in range(2 * BAROSTAT_INTERVAL):
        state = step_fn(state)
        assert state.step_count == i + 1
        assert state.barostat_state.num_attempted == ((i + 1) // BAROSTAT_INTERVAL)
