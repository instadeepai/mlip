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
