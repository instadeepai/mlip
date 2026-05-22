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
from ase import Atoms
from pydantic import ValidationError

from mlip.simulation import SimulationType
from mlip.simulation.ts_search import NEBSimulationEngine


def test_neb_can_be_run_with_ase_backend(quadratic_force_field) -> None:
    atoms_initial = Atoms("HCN", positions=[(0, 0, 0), (1.07, 0, 0), (2.22, 0, 0)])
    atoms_final = Atoms("HCN", positions=[(1.18 + 0.99, 0, 0), (0, 0, 0), (1.18, 0, 0)])

    neb_config = NEBSimulationEngine.Config(
        num_images=9,
        num_steps=1,
        snapshot_interval=1,
        log_interval=1,
        box=None,
        edge_capacity_multiplier=1.25,
        max_force_convergence_threshold=0.1,
    )

    intermediate_steps = []

    def _mock_logger(state):
        intermediate_steps.append(state.step)

    engine = NEBSimulationEngine(
        [atoms_initial, atoms_final], quadratic_force_field, neb_config
    )
    engine.attach_logger(_mock_logger)

    engine.run()

    assert engine.state.positions.shape == (9, 3, 3)

    neb_final_force = np.sqrt((engine.state.forces[-1] ** 2).sum(axis=1).max())
    assert neb_final_force == pytest.approx(3.736355, abs=1e-3)


def test_ase_with_transition_state_guess(quadratic_force_field) -> None:
    atoms_initial = Atoms("HCN", positions=[(0, 0, 0), (1.07, 0, 0), (2.22, 0, 0)])
    atoms_ts_guess = Atoms("HCN", positions=[(0.6, 0.5, 0), (0.9, 0, 0), (2.0, 0, 0)])
    atoms_final = Atoms("HCN", positions=[(1.18 + 0.99, 0, 0), (0, 0, 0), (1.18, 0, 0)])

    neb_config = NEBSimulationEngine.Config(
        num_images=9,
        num_steps=1,
        snapshot_interval=1,
        log_interval=1,
        box=None,
        edge_capacity_multiplier=1.25,
        max_force_convergence_threshold=0.1,
    )

    engine = NEBSimulationEngine(
        [atoms_initial, atoms_ts_guess, atoms_final],
        quadratic_force_field,
        neb_config,
    )
    engine.run()

    assert len(engine.neb.images) == 9
    assert engine.state.positions.shape == (9, 3, 3)

    ts_guess_index = neb_config.num_images // 2
    np.testing.assert_allclose(
        engine.state.positions[ts_guess_index],
        atoms_ts_guess.positions,
        atol=0.05,
    )


def test_ase_with_multiple_images_continued_from_previous(
    quadratic_force_field,
) -> None:
    atoms_initial = Atoms("HCN", positions=[(0, 0, 0), (1.07, 0, 0), (2.22, 0, 0)])
    atoms_final = Atoms("HCN", positions=[(1.18 + 0.99, 0, 0), (0, 0, 0), (1.18, 0, 0)])

    initial_config = NEBSimulationEngine.Config(
        num_images=9,
        num_steps=1,
        snapshot_interval=1,
        log_interval=1,
        box=None,
        edge_capacity_multiplier=1.25,
        max_force_convergence_threshold=0.1,
    )
    initial_engine = NEBSimulationEngine(
        [atoms_initial, atoms_final], quadratic_force_field, initial_config
    )
    initial_engine.run()

    images_after_first_run = [img.copy() for img in initial_engine.neb.images]
    assert len(images_after_first_run) == 9

    continue_config = NEBSimulationEngine.Config(
        num_images=9,
        num_steps=1,
        snapshot_interval=1,
        log_interval=1,
        box=None,
        edge_capacity_multiplier=1.25,
        max_force_convergence_threshold=0.1,
        continue_from_previous_run=True,
    )
    continue_engine = NEBSimulationEngine(
        images_after_first_run, quadratic_force_field, continue_config
    )

    # test if no interpolation before run()
    for supplied, used in zip(images_after_first_run, continue_engine.images):
        np.testing.assert_allclose(used.positions, supplied.positions, atol=1e-12)

    continue_engine.run()

    assert len(continue_engine.neb.images) == 9
    assert continue_engine.state.positions.shape == (9, 3, 3)


def test_neb_must_be_ts_search_sim_type() -> None:
    with pytest.raises(ValidationError):
        NEBSimulationEngine.Config(
            simulation_type=SimulationType.MD,
            num_steps=1,
        )
    NEBSimulationEngine.Config(
        simulation_type=SimulationType.TS_SEARCH,
        num_steps=1,
    )
