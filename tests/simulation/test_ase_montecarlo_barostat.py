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

import numpy as np
import pytest
from ase import units
from ase.calculators.calculator import Calculator
from jax_md import dataclasses

from mlip.simulation.ase.ase_montecarlo_barostat import ASEMonteCarloBarostat
from mlip.simulation.ase.mlip_ase_calculator import MLIPForceFieldASECalculator
from mlip.simulation.montecarlo_barostat import INITIAL_MAX_DELTA_VOLUME_FRACTION

TEMPERATURE = 300.0
PRESSURE = 1.01325
MOLECULE_INDICES = np.array([0] * 4 + [1] * 6)
MOL_COUNTS = np.array([4, 6])


@pytest.fixture
def system_and_model(setup_system, mace_force_field):
    system, force_field = setup_system[0], mace_force_field
    atoms = system.copy()
    atoms.set_cell([10.0, 10.0, 10.0])
    atoms.pbc = [True, True, True]
    return atoms, force_field


def _create_ase_montecarlo_barostat(atoms, force_field):
    temperature_getter = lambda: TEMPERATURE  # noqa: E731
    base_calculator = MLIPForceFieldASECalculator(atoms, 1.25, force_field)
    return ASEMonteCarloBarostat(
        atoms=atoms,
        base_calculator=base_calculator,
        temperature_getter=temperature_getter,
        pressure_bar=PRESSURE,
        molecule_indices=MOLECULE_INDICES,
        random_seed=42,
    )


def test_ase_montecarlo_barostat_initialization(system_and_model):
    """Test MonteCarloBarostat initialization."""
    atoms, force_field = system_and_model
    barostat = _create_ase_montecarlo_barostat(atoms, force_field)

    assert barostat.atoms is atoms
    assert barostat._calculator.model_params is force_field.params
    assert barostat.temperature_getter() == TEMPERATURE
    assert barostat.barostat_state.target_pressure == PRESSURE * units.bar
    assert np.array_equal(barostat.barostat_state.mol_indices, MOLECULE_INDICES)
    assert np.array_equal(barostat.barostat_state.mol_counts, MOL_COUNTS)
    assert barostat.barostat_state.max_delta_volume > 0
    assert barostat.barostat_state.num_attempted == 0
    assert barostat.barostat_state.num_accepted == 0
    assert barostat.barostat_state.num_attempted_since_tune == 0
    assert barostat.barostat_state.num_accepted_since_tune == 0

    initial_volume = atoms.get_volume()
    expected_max_delta = initial_volume * INITIAL_MAX_DELTA_VOLUME_FRACTION
    assert np.isclose(
        barostat.barostat_state.max_delta_volume, expected_max_delta, rtol=1e-10
    )


class MockCalculator(Calculator):
    """Mock calculator that returns energy based on call order."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, energy_old: float, energy_new: float):
        super().__init__()
        self.energy_old = energy_old
        self.energy_new = energy_new
        self.call_count = 0

    def calculate(self, atoms, properties=None, system_changes=None):
        """Returns the energy based on call order."""
        if self.call_count == 0:
            energy = self.energy_old
        else:
            energy = self.energy_new

        self.call_count += 1
        self.results = {"energy": energy, "forces": np.zeros((len(atoms), 3))}


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
def test_ase_montecarlo_barostat_step_integration(
    system_and_model, energy_delta, volume_delta
):
    """Test that the `step` function updates the atoms correctly."""
    atoms, force_field = system_and_model
    barostat = _create_ase_montecarlo_barostat(atoms, force_field)
    energy_old, energy_new = 0.0, energy_delta
    initial_atoms = atoms.copy()

    for if_accept in [False, True]:
        barostat.atoms = initial_atoms.copy()
        mock_calc = MockCalculator(energy_old, energy_new)
        barostat._calculator = mock_calc
        barostat.atoms.calc = mock_calc

        attempted_old = barostat.barostat_state.num_attempted
        accepted_old = barostat.barostat_state.num_accepted
        positions_old = barostat.atoms.get_positions().copy()
        cell_old = barostat.atoms.get_cell().array.copy()
        volume_old = barostat.atoms.get_volume()
        volume_new = volume_old + volume_delta
        length_scale = (volume_new / volume_old) ** (1 / 3)

        box_new = cell_old * length_scale
        pos_new = positions_old * length_scale

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

        barostat._propose_step = mock_propose
        barostat._accept_step = mock_accept
        barostat._tune_barostat_step = mock_tune

        barostat.step()

        if if_accept:
            expected_cell = box_new
            expected_positions = pos_new
            expected_energy = energy_new
        else:
            expected_cell = cell_old
            expected_positions = positions_old
            expected_energy = energy_old
        expected_num_accepted = accepted_old + int(if_accept)

        assert barostat.barostat_state.num_attempted == attempted_old + 1
        assert barostat.barostat_state.num_accepted == expected_num_accepted
        assert barostat.barostat_state.num_attempted_since_tune == 0
        assert barostat.barostat_state.num_accepted_since_tune == 0
        assert np.allclose(barostat.atoms.get_cell().array, expected_cell)
        assert np.allclose(barostat.atoms.get_positions(), expected_positions)
        assert barostat.atoms.calc.results["energy"] == expected_energy
