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

from mlip.simulation.enums import SimulationType
from mlip.simulation.jax_md.jax_md_simulation_engine import JaxMDSimulationEngine


@pytest.mark.parametrize("is_batched", [True, False])
def test_minimization_can_be_run_with_jax_md_backend(
    setup_system_and_mace_model, is_batched
):
    atoms, _, _, mace_ff = setup_system_and_mace_model

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MINIMIZATION,
        num_steps=20,
        snapshot_interval=2,
        num_episodes=2,
        timestep_fs=5.0,
        box=None,
        edge_capacity_multiplier=1.25,
    )

    intermediate_steps = []

    def _mock_logger(state):
        intermediate_steps.append(state.step)
        assert state.temperature is None
        assert state.forces is not None

    if is_batched:
        engine = JaxMDSimulationEngine(
            [atoms, deepcopy(atoms), deepcopy(atoms)], mace_ff, md_config
        )
    else:
        engine = JaxMDSimulationEngine(atoms, mace_ff, md_config)

    engine.attach_logger(_mock_logger)

    engine.run()

    assert engine.state.step == 20
    assert engine.state.compute_time_seconds > 0.0
    assert engine.state.temperature is None
    assert engine.state.kinetic_energy is None
    assert engine.state.velocities is None
    assert intermediate_steps == [10, 20]

    expected_first_atom_forces = np.array([0.0498, -0.0216, 0.0118])

    if is_batched:
        assert len(engine.state.forces) == 3
        assert len(engine.state.positions) == 3

        for i in range(3):
            assert engine.state.forces[i].shape == (10, 10, 3)
            assert engine.state.positions[i].shape == (10, 10, 3)

        assert np.allclose(engine.state.forces[0], engine.state.forces[1], atol=1e-3)
        assert np.allclose(
            engine.state.positions[0], engine.state.positions[1], atol=1e-3
        )
        assert np.allclose(engine.state.forces[0], engine.state.forces[2], atol=1e-3)
        assert np.allclose(
            engine.state.positions[0], engine.state.positions[2], atol=1e-3
        )
        assert np.allclose(
            engine.state.forces[0][0][0], expected_first_atom_forces, atol=1e-3
        )

    else:
        assert engine.state.forces.shape == (10, 10, 3)
        assert engine.state.positions.shape == (10, 10, 3)
        assert np.allclose(
            engine.state.forces[0][0], expected_first_atom_forces, atol=1e-3
        )
