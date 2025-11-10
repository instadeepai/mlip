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

from copy import deepcopy
from unittest.mock import Mock

import ase
import numpy as np
import pytest
from pydantic import ValidationError

from mlip.simulation.configs.simulation_config import TemperatureScheduleConfig
from mlip.simulation.enums import SimulationType, TemperatureScheduleMethod
from mlip.simulation.jax_md.jax_md_simulation_engine import JaxMDSimulationEngine


def test_md_can_be_run_with_jax_md_backend(setup_system_and_mace_model):
    atoms, _, _, mace_ff = setup_system_and_mace_model

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        num_steps=20,
        snapshot_interval=2,
        num_episodes=5,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=None,
        edge_capacity_multiplier=1.25,
    )

    intermediate_steps = []

    def _mock_logger(state):
        intermediate_steps.append(state.step)

    engine = JaxMDSimulationEngine(atoms, mace_ff, md_config)
    engine.attach_logger(_mock_logger)

    engine.run()

    assert engine.state.step == 20
    assert engine.state.compute_time_seconds > 0.0
    assert engine.state.temperature.shape == (10,)
    assert engine.state.kinetic_energy.shape == (10,)
    assert engine.state.positions.shape == (10, 10, 3)
    assert engine.state.forces.shape == (10, 10, 3)
    assert engine.state.velocities.shape == (10, 10, 3)
    assert intermediate_steps == [4, 8, 12, 16, 20]


def test_batched_md_can_be_run_with_jax_md_backend_for_three_identical_systems(
    setup_system_and_mace_model,
):
    atoms, _, _, mace_ff = setup_system_and_mace_model

    # Run three systems at the same time
    systems = [atoms, deepcopy(atoms), deepcopy(atoms)]

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        num_steps=20,
        snapshot_interval=2,
        num_episodes=5,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=None,
        edge_capacity_multiplier=1.25,
    )

    intermediate_steps = []

    def _mock_logger(state):
        intermediate_steps.append(state.step)

    engine = JaxMDSimulationEngine(systems, mace_ff, md_config)
    engine.attach_logger(_mock_logger)

    engine.run()

    assert engine.state.step == 20
    assert engine.state.compute_time_seconds > 0.0
    assert intermediate_steps == [4, 8, 12, 16, 20]
    assert len(engine.state.temperature) == 3
    assert len(engine.state.kinetic_energy) == 3
    assert len(engine.state.positions) == 3
    assert len(engine.state.forces) == 3
    assert len(engine.state.velocities) == 3

    for i in range(3):
        assert engine.state.temperature[i].shape == (10,)
        assert engine.state.kinetic_energy[i].shape == (10,)
        assert engine.state.positions[i].shape == (10, 10, 3)
        assert engine.state.forces[i].shape == (10, 10, 3)
        assert engine.state.velocities[i].shape == (10, 10, 3)

    for i in [1, 2]:
        assert np.allclose(
            engine.state.positions[i], engine.state.positions[0], atol=1e-5
        )
        assert np.allclose(engine.state.forces[i], engine.state.forces[0], atol=1e-5)
        assert np.allclose(
            engine.state.velocities[i], engine.state.velocities[0], atol=1e-5
        )
        assert np.allclose(
            engine.state.kinetic_energy[i], engine.state.kinetic_energy[0], atol=1e-5
        )
        assert np.allclose(
            engine.state.temperature[i], engine.state.temperature[0], atol=1e-5
        )


def test_batched_md_can_be_run_with_jax_md_backend_for_two_different_systems(
    setup_system_and_mace_model,
):
    atoms, _, _, mace_ff = setup_system_and_mace_model

    # Add one atom to the molecule
    mod_numbers = list(atoms.numbers) + [6]
    mod_pos = np.concatenate(
        [atoms.get_positions(), np.array([[0.1, 0.2, 0.3]])], axis=0
    )
    mod_atoms = ase.Atoms(numbers=mod_numbers, positions=mod_pos)

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        num_steps=20,
        snapshot_interval=2,
        num_episodes=5,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=None,
        edge_capacity_multiplier=1.25,
    )

    engine_1 = JaxMDSimulationEngine([atoms, mod_atoms], mace_ff, md_config)
    engine_1.run()

    assert engine_1.state.step == 20
    assert len(engine_1.state.positions) == 2
    assert engine_1.state.positions[0].shape == (10, 10, 3)
    assert engine_1.state.positions[1].shape == (10, 11, 3)

    engine_2 = JaxMDSimulationEngine(atoms, mace_ff, md_config)
    engine_2.run()

    engine_3 = JaxMDSimulationEngine(mod_atoms, mace_ff, md_config)
    engine_3.run()

    tol = 1e-3
    assert np.allclose(engine_1.state.positions[0], engine_2.state.positions, atol=tol)
    assert np.allclose(engine_1.state.positions[1], engine_3.state.positions, atol=tol)
    assert np.allclose(engine_1.state.forces[0], engine_2.state.forces, atol=tol)
    assert np.allclose(engine_1.state.forces[1], engine_3.state.forces, atol=tol)
    assert np.allclose(
        engine_1.state.velocities[0], engine_2.state.velocities, atol=tol
    )
    assert np.allclose(
        engine_1.state.velocities[1], engine_3.state.velocities, atol=tol
    )
    assert np.allclose(
        engine_1.state.kinetic_energy[0], engine_2.state.kinetic_energy, atol=tol
    )
    assert np.allclose(
        engine_1.state.kinetic_energy[1], engine_3.state.kinetic_energy, atol=tol
    )
    # Use relative tolerance for temperatures as these values are quite large
    assert np.allclose(
        engine_1.state.temperature[0], engine_2.state.temperature, rtol=tol
    )
    assert np.allclose(
        engine_1.state.temperature[1], engine_3.state.temperature, rtol=tol
    )


def test_batched_and_regular_md_yield_same_results(
    setup_system_and_mace_model,
):
    atoms, _, _, mace_ff = setup_system_and_mace_model

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        num_steps=20,
        snapshot_interval=2,
        num_episodes=5,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=None,
        edge_capacity_multiplier=1.25,
    )

    engine_1 = JaxMDSimulationEngine([atoms], mace_ff, md_config)
    engine_1.run()

    engine_2 = JaxMDSimulationEngine(atoms, mace_ff, md_config)
    engine_2.run()

    tol = 1e-3
    assert np.allclose(engine_1.state.positions[0], engine_2.state.positions, atol=tol)
    assert np.allclose(engine_1.state.forces[0], engine_2.state.forces, atol=tol)
    assert np.allclose(
        engine_1.state.velocities[0], engine_2.state.velocities, atol=tol
    )
    assert np.allclose(
        engine_1.state.kinetic_energy[0], engine_2.state.kinetic_energy, atol=tol
    )
    # Use relative tolerance for temperatures as these values are quite large
    assert np.allclose(
        engine_1.state.temperature[0], engine_2.state.temperature, rtol=tol
    )


def test_jax_md_config_validation_works():
    with pytest.raises(ValidationError) as exc1:
        JaxMDSimulationEngine.Config(
            simulation_type=SimulationType.MD,
            num_steps=20,
            snapshot_interval=3,
            num_episodes=5,
            timestep_fs=1.0,
            temperature_kelvin=300.0,
            box=None,
            edge_capacity_multiplier=1.25,
            temperature_schedule_config=TemperatureScheduleConfig(
                method=TemperatureScheduleMethod.CONSTANT,
                temperature=300.0,
            ),
        )

    assert "Snapshot interval must evenly divide steps per episode." in str(exc1.value)

    with pytest.raises(ValidationError) as exc2:
        JaxMDSimulationEngine.Config(
            simulation_type=SimulationType.MD,
            num_steps=20,
            snapshot_interval=1,
            num_episodes=3,
            timestep_fs=1.0,
            temperature_kelvin=300.0,
            box=None,
            edge_capacity_multiplier=1.25,
        )

    assert "Number of episodes must evenly divide total steps." in str(exc2.value)

    with pytest.raises(ValidationError) as exc3:
        JaxMDSimulationEngine.Config(
            simulation_type=SimulationType.MD,
            num_steps=20,
            log_interval=1,
            num_episodes=1,
            timestep_fs=0.0,
            temperature_kelvin=300.0,
            box=None,
            edge_capacity_multiplier=1.25,
        )

    assert "timestep_fs" in str(exc3.value)
    assert str(exc3.value).count("Input should be greater than 0") == 1


def test_md_can_be_restarted_from_velocities_with_jax_md_backend(
    setup_system_and_mace_model,
):
    atoms, _, _, mace_ff = setup_system_and_mace_model
    _atoms = deepcopy(atoms)

    velocities_to_restore = np.ones(_atoms.get_positions().shape)
    _atoms.set_velocities(velocities_to_restore)

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        num_steps=5,
        num_episodes=1,
    )

    engine = JaxMDSimulationEngine(_atoms, mace_ff, md_config)
    engine.run()

    assert engine.state.velocities.shape[0] == 5
    assert np.allclose(engine.state.velocities[0], velocities_to_restore)

    for i in range(1, 5):
        assert not np.allclose(engine.state.velocities[i], velocities_to_restore)


def test_single_atom_system_cannot_be_simulated():
    md_config = Mock()
    force_field = Mock()
    proper_system = ase.Atoms("COH")
    invalid_system = ase.Atoms("H")

    with pytest.raises(ValueError) as exc1:
        JaxMDSimulationEngine([proper_system, invalid_system], force_field, md_config)

    assert "Single atom system detected in batch, not supported yet." in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        JaxMDSimulationEngine(invalid_system, force_field, md_config)

    assert "Single atom systems are not supported yet." in str(exc2.value)


def test_empty_atoms_inputs_cannot_be_simulated():
    md_config = Mock()
    force_field = Mock()

    with pytest.raises(ValueError) as exc1:
        JaxMDSimulationEngine([], force_field, md_config)

    assert "Passed 'atoms' argument is empty." in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        JaxMDSimulationEngine(ase.Atoms(), force_field, md_config)

    assert "Passed 'atoms' argument is empty." in str(exc2.value)

    proper_system = ase.Atoms("COH")
    with pytest.raises(ValueError) as exc3:
        JaxMDSimulationEngine([proper_system, ase.Atoms()], force_field, md_config)

    assert "Empty 'ase.Atoms' detected in batch." in str(exc3.value)
