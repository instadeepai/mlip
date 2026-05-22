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

import functools
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from ase import units
from jax import random
from jax_md import dataclasses

from mlip.simulation.montecarlo_barostat import (
    INITIAL_MAX_DELTA_VOLUME_FRACTION,
    TUNE_FREQUENCY,
    MonteCarloBarostatState,
    _box_to_volume,  # noqa: PLC2701
    _scale_molecule_centroids,  # noqa: PLC2701
    accept_volume_change,
    create_high_precision_force_field,
    propose_volume_change,
    tune_barostat,
)

TEMPERATURE = 300.0
PRESSURE = 1.01325 * units.bar
MOLECULE_INDICES = jnp.array([0] * 4 + [1] * 6)
MOL_COUNTS = jnp.array([4, 6])
POSITIONS = jax.random.uniform(random.PRNGKey(42), (10, 3), minval=-10.0, maxval=10.0)
BOX = jnp.array([10.0, 10.0, 10.0])
VOLUME = _box_to_volume(BOX, 3)


def test_create_high_precision_force_field(mace_force_field, salt_graph):
    original_config = mace_force_field.predictor.mlip_network.config.model_dump()

    high_precision_force_field = create_high_precision_force_field(mace_force_field)
    new_config = high_precision_force_field.predictor.mlip_network.config.model_dump()

    # Check that only atomic_energies and deterministic_scatter_ops are changed.
    assert not new_config["add_atomic_energies"]
    assert new_config["deterministic_scatter_ops"]
    for key, value in new_config.items():
        if key in ["add_atomic_energies", "deterministic_scatter_ops"]:
            assert value != original_config[key]
        else:
            assert value == original_config[key]

    # Check that the energy head has been replaced with a deterministic version
    energy_head = high_precision_force_field.predictor.energy_head
    assert isinstance(energy_head, functools.partial)
    assert energy_head.func is mace_force_field.predictor.energy_head
    assert energy_head.keywords == {"deterministic": True}

    # Testing on CPU, so both models are deterministic either way.
    # Check that the energy is higher with atomic_energies=`zero`,
    # and that no forces are predicted.
    old_preds = jax.jit(mace_force_field)(salt_graph)
    new_preds = jax.jit(high_precision_force_field)(salt_graph)
    assert new_preds.energy[0] > old_preds.energy[0]
    assert new_preds.forces is None


@pytest.mark.parametrize("scale_factor", [0.5, 1.0, 1.5])
def test_scale_molecule_centroids(scale_factor):
    """Test the scale_molecule_centroids function.

    Expected behaviour:
        * Maintains distances between atoms in the same molecule.
        * Scales distances between the COM of different molecules by the scale factor.
    """
    mol_0_idx = jnp.where(MOLECULE_INDICES == 0)[0]
    mol_1_idx = jnp.where(MOLECULE_INDICES == 1)[0]

    pos_old = POSITIONS
    pos_new = _scale_molecule_centroids(
        pos_old, MOLECULE_INDICES, MOL_COUNTS, scale_factor
    )

    # Check that the distances between atoms in the same molecule are preserved
    dists_old = jnp.linalg.norm(pos_old[:, None, :] - pos_old[None, :, :], axis=-1)
    dists_new = jnp.linalg.norm(pos_new[:, None, :] - pos_new[None, :, :], axis=-1)

    def _compare_dists(idx1, idx2):
        return jnp.allclose(
            dists_new[idx1[:, None], idx2[None, :]],
            dists_old[idx1[:, None], idx2[None, :]],
            rtol=1e-6,
        )

    assert _compare_dists(mol_0_idx, mol_0_idx)
    assert _compare_dists(mol_1_idx, mol_1_idx)

    # Check that COM distances are scaled correctly
    com_0_old, com_1_old = pos_old[mol_0_idx].mean(0), pos_old[mol_1_idx].mean(0)
    com_0_new, com_1_new = pos_new[mol_0_idx].mean(0), pos_new[mol_1_idx].mean(0)
    com_dist_old = jnp.linalg.norm(com_1_old - com_0_old)
    com_dist_new = jnp.linalg.norm(com_1_new - com_0_new)
    assert jnp.allclose(com_dist_new, scale_factor * com_dist_old, rtol=1e-6)


@pytest.fixture
def barostat_state():
    return MonteCarloBarostatState(
        target_pressure=PRESSURE,
        num_attempted=0,
        num_accepted=0,
        num_attempted_since_tune=0,
        num_accepted_since_tune=0,
        max_delta_volume=INITIAL_MAX_DELTA_VOLUME_FRACTION * VOLUME,
        rng=random.PRNGKey(42),
        mol_counts=MOL_COUNTS,
        mol_indices=MOLECULE_INDICES,
    )


@pytest.mark.parametrize("max_delta_volume", [10.0, 100.0, 1000.0])
def test_propose_volume_change(barostat_state, max_delta_volume):
    """Test the propose_volume_change function.

    Expected behaviour:
        * Proposes a volume change in [-max_delta_volume, max_delta_volume].
        * Scales the positions by the length scale.
    """
    barostat_state = dataclasses.replace(
        barostat_state, max_delta_volume=max_delta_volume
    )

    for sample_value in [0.0, 0.5, 1.0]:
        rng_old = barostat_state.rng
        with patch("jax.random.uniform") as mock_rand:
            # Force a sampled value in [0.0, 1.0]
            mock_rand.side_effect = [sample_value]
            barostat_state_new, volume_old, volume_new, box_new, positions_new = (
                propose_volume_change(barostat_state, BOX, POSITIONS)
            )
        expected_volume_new = volume_old + (sample_value * 2 - 1) * max_delta_volume
        expected_length_scale = (
            jnp.maximum(expected_volume_new, 1e-6) / volume_old
        ) ** (1.0 / 3)
        expected_positions_new = _scale_molecule_centroids(
            POSITIONS, MOLECULE_INDICES, MOL_COUNTS, expected_length_scale
        )
        expected_box_new = BOX * expected_length_scale
        assert jnp.isclose(volume_old, _box_to_volume(BOX, 3))
        assert jnp.isclose(volume_new, expected_volume_new)
        assert jnp.array_equal(positions_new, expected_positions_new)
        assert jnp.array_equal(box_new, expected_box_new)
        # Assert rng is different
        assert not jnp.array_equal(barostat_state_new.rng, rng_old)


@pytest.fixture(
    params=[
        (-0.001, -100.0, 0.844008),
        (0.001, -100.0, 0.781175),
        (-0.1, -100.0, 1.0),
        (0.1, -100.0, 0.016968),
        (1.0, -100.0, 0.0),
    ],
    ids=["case0", "case1", "case2", "case3", "case4"],
)
def accept_volume_change_test_case(request):
    """Shared test cases for the ASE and JAX-MD MC Barostat acceptance probability.

    Returns:
        (energy_delta, volume_delta, expected_prob)
    """
    return request.param


def test_accept_volume_change(barostat_state, accept_volume_change_test_case):
    """Test the accept_volume_change function.

    Expected behaviour:
        * Decides whether to accept the volume change based on the Metropolis criterion.
    """
    energy_delta, volume_delta, expected_prob = accept_volume_change_test_case
    energy_old, energy_new = 0.0, energy_delta
    volume_old, volume_new = (
        _box_to_volume(BOX, 3),
        _box_to_volume(BOX, 3) + volume_delta,
    )
    kT = units.kB * TEMPERATURE
    num_molecules = jnp.max(MOLECULE_INDICES) + 1

    for sample_value in [expected_prob + 1e-6, expected_prob - 1e-6]:
        accepted_old = barostat_state.num_accepted
        attempted_old = barostat_state.num_attempted
        attempted_since_tune_old = barostat_state.num_attempted_since_tune
        accepted_since_tune_old = barostat_state.num_accepted_since_tune
        with patch("jax.random.uniform") as mock_rand:
            # Force a sampled value in [0.0, 1.0]
            mock_rand.side_effect = [sample_value]
            barostat_state_new, accepted = accept_volume_change(
                barostat_state,
                energy_old,
                energy_new,
                volume_old,
                volume_new,
                kT,
                num_molecules,
            )

        # Check that the state was updated correctly
        if volume_new <= 0:
            if_accept = False
        else:
            if_accept = True if expected_prob == 1.0 else sample_value < expected_prob

        assert accepted == if_accept
        assert barostat_state_new.num_attempted == attempted_old + 1
        assert (
            barostat_state_new.num_attempted_since_tune == attempted_since_tune_old + 1
        )
        assert barostat_state_new.num_accepted == accepted_old + jnp.where(
            if_accept, 1, 0
        )
        assert (
            barostat_state_new.num_accepted_since_tune
            == accepted_since_tune_old + jnp.where(if_accept, 1, 0)
        )
        assert not jnp.array_equal(barostat_state_new.rng, barostat_state.rng)


@pytest.fixture(
    params=[
        (100.0, 0.1, 100.0 / 1.1),
        (100.0, 0.5, 100.0),
        (100.0, 0.9, 100.0 * 1.1),
        (1e4, 0.9, 1000.0 * 0.3),  # Capped at 30% of initial volume
    ],
    ids=["case0", "case1", "case2", "case3"],
)
def tune_barostat_test_case(request):
    """Shared test cases for the ASE and JAX-MD MC Barostat volume tuning.

    Returns:
        (max_delta_volume, acceptance_rate, expected_max_delta_volume)
    """
    return request.param


def test_montecarlo_barostat_tuning(barostat_state, tune_barostat_test_case):
    """Test that the barostat tunes max_delta_volume correctly."""
    max_delta_volume, acceptance_rate, expected_max_delta_volume = (
        tune_barostat_test_case
    )

    for num_attempted in [TUNE_FREQUENCY - 1, TUNE_FREQUENCY]:
        num_accepted = int(num_attempted * acceptance_rate)

        barostat_state = dataclasses.replace(
            barostat_state,
            max_delta_volume=max_delta_volume,
            num_attempted=num_attempted,
            num_accepted=num_accepted,
            num_attempted_since_tune=num_attempted,
            num_accepted_since_tune=num_accepted,
        )

        barostat_state_new = tune_barostat(barostat_state, _box_to_volume(BOX, 3))

        if num_attempted >= TUNE_FREQUENCY:  # Should have tuned
            assert barostat_state_new.num_attempted == num_attempted
            assert barostat_state_new.num_accepted == num_accepted
            assert barostat_state_new.num_attempted_since_tune == 0
            assert barostat_state_new.num_accepted_since_tune == 0
            assert jnp.isclose(
                barostat_state_new.max_delta_volume,
                expected_max_delta_volume,
                rtol=1e-10,
            )
        else:  # No tuning
            assert barostat_state_new.num_attempted == num_attempted
            assert barostat_state_new.num_accepted == num_accepted
            assert barostat_state_new.num_attempted_since_tune == num_attempted
            assert barostat_state_new.num_accepted_since_tune == num_accepted
            assert jnp.isclose(
                barostat_state_new.max_delta_volume, max_delta_volume, rtol=1e-10
            )
