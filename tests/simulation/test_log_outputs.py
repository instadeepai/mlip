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

from mlip.simulation.ase.ase_simulation_engine import ASESimulationEngine
from mlip.simulation.configs import SimulationLogOutputs
from mlip.simulation.enums import SimulationType
from mlip.simulation.jax_md.jax_md_simulation_engine import JaxMDSimulationEngine


def test_jax_md_engine_only_saves_what_in_log_outputs(
    quadratic_force_field, setup_system
):
    atoms, _ = setup_system

    log_outputs = SimulationLogOutputs(
        kinetic_energy=False, positions=False, forces=False
    )

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        num_steps=8,
        num_episodes=2,
        log_outputs=log_outputs,
    )

    engine = JaxMDSimulationEngine(atoms, quadratic_force_field, md_config)
    engine.run()

    assert engine.state.step == 8
    assert engine.state.compute_time_seconds > 0.0
    assert engine.state.temperature.shape == (8,)
    assert engine.state.velocities.shape == (8, 10, 3)
    assert engine.state.kinetic_energy is None
    assert engine.state.positions is None
    assert engine.state.forces is None


def test_ase_engine_only_saves_what_in_log_outputs(
    quadratic_force_field, setup_system
) -> None:
    atoms = deepcopy(setup_system[0])

    log_outputs = SimulationLogOutputs(
        temperature=False, velocities=False, forces=False
    )

    md_config = ASESimulationEngine.Config(
        simulation_type=SimulationType.MD,
        num_steps=8,
        snapshot_interval=2,
        log_interval=2,
        log_outputs=log_outputs,
    )

    engine = ASESimulationEngine(atoms, quadratic_force_field, md_config)
    engine.run()

    assert engine.state.step == 8
    assert engine.state.compute_time_seconds > 0.0
    assert engine.state.kinetic_energy.shape == (5,)
    assert engine.state.positions.shape == (5, 10, 3)
    assert engine.state.temperature is None
    assert engine.state.forces is None
    assert engine.state.velocities is None
