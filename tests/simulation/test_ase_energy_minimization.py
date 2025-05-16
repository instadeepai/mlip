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
from mlip.simulation.enums import SimulationType
from mlip.simulation.state import SimulationState


def test_minimization_can_be_run_with_ase_backend(setup_system_and_mace_model) -> None:
    init_conf, _, _, mace_ff = setup_system_and_mace_model
    atoms = deepcopy(init_conf)

    md_config = ASESimulationEngine.Config(
        simulation_type=SimulationType.MINIMIZATION,
        num_steps=20,
        snapshot_interval=2,
        log_interval=2,
        timestep_fs=5.0,
        max_force_convergence_threshold=0.005,
        box=None,
        edge_capacity_multiplier=1.25,
    )

    intermediate_steps = []

    def _mock_logger(state: SimulationState) -> None:
        intermediate_steps.append(state.step)
        assert state.temperature is None
        assert state.forces is not None

    engine = ASESimulationEngine(atoms, mace_ff, md_config)
    engine.attach_logger(_mock_logger)

    engine.run()

    assert engine.state.step == 20
    assert engine.state.compute_time_seconds > 0.0
    assert engine.state.temperature is None
    assert engine.state.kinetic_energy is None
    assert engine.state.positions.shape == (11, 10, 3)
    assert engine.state.forces.shape == (11, 10, 3)
    assert engine.state.velocities is None
    assert intermediate_steps == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
