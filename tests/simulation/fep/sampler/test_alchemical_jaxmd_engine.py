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

from mlip.simulation.enums import MDIntegrator, SimulationType
from mlip.simulation.fep.enums import FEPStage
from mlip.simulation.fep.models.alchemical_forcefield import AlchemicalForceField
from mlip.simulation.fep.sampler.alchemical_jaxmd_engine import (
    AlchemicalJaxMDSimulationEngine,
)

N_STEPS = 4
SNAPSHOT_INTERVAL = 2
ALCHEMICAL_ATOM_INDICES = np.arange(2)

ALL_MD_INTEGRATORS = ["nvt_langevin", "npt_mc_langevin", "nve_velocity_verlet"]

ENGINE_RUN_CASES = [
    (md_integrator, "quadratic_force_field") for md_integrator in ALL_MD_INTEGRATORS
]


@pytest.mark.parametrize("md_integrator, force_field_name", ENGINE_RUN_CASES)
def test_alchemical_engine_run(force_field_name, request, setup_system, md_integrator):
    """AlchemicalJaxMDSimulationEngine runs standalone for a single lambda window."""
    force_field = request.getfixturevalue(force_field_name)
    atoms, _ = setup_system
    atoms = deepcopy(atoms)
    if force_field.long_range_cutoff_distance is not None:
        atoms.info["charge"] = 1.0

    force_field = AlchemicalForceField.from_mlip_network(
        force_field.predictor.mlip_network,
        fep_stage=FEPStage.AR,
        required_properties=force_field.predictor.required_properties,
    )
    reference_lambdas = np.array([[0.0, 0.0], [0.5, 0.2], [0.5, 0.5], [1.0, 1.0]])
    md_integrator = MDIntegrator(md_integrator)

    md_config = AlchemicalJaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=md_integrator,
        num_steps=N_STEPS,
        snapshot_interval=SNAPSHOT_INTERVAL,
        box=10.0,
        num_episodes=2,
        molecule_indices=[0] * len(atoms),
    )

    intermediate_steps = []

    def _mock_logger(state):
        intermediate_steps.append(state.step)

    engine = AlchemicalJaxMDSimulationEngine(
        atoms,
        force_field,
        md_config,
        lambda_values=np.array([0.5, 0.5]),
        alchemical_atom_indices=ALCHEMICAL_ATOM_INDICES,
        reference_lambdas=reference_lambdas,
        alchemical_energy_batch_size=None,
    )
    engine.attach_logger(_mock_logger)
    engine.run()

    num_atoms = len(atoms)
    num_snapshots = N_STEPS // SNAPSHOT_INTERVAL

    assert engine.state.step == N_STEPS
    assert engine.state.compute_time_seconds > 0.0
    assert engine.state.temperature.shape == (num_snapshots,)
    assert engine.state.kinetic_energy.shape == (num_snapshots,)
    assert engine.state.positions.shape == (num_snapshots, num_atoms, 3)
    assert engine.state.forces.shape == (num_snapshots, num_atoms, 3)
    assert engine.state.velocities.shape == (num_snapshots, num_atoms, 3)
    assert engine.state.potential_energy.shape == (num_snapshots,)
    assert engine.state.per_lambda_energies.shape == (
        num_snapshots,
        len(reference_lambdas),
    )
    assert not np.allclose(engine.state.per_lambda_energies, 0.0)
    assert engine.state.final_positions.shape == (num_atoms, 3)
    assert engine.state.final_velocities.shape == (num_atoms, 3)
    assert intermediate_steps == [2, 4]

    if md_integrator.ensemble == "npt":
        assert engine.state.cell.shape == (num_snapshots, 3, 3)
        assert engine.state.final_cell.shape == (3, 3)

    if force_field_name == "lri_quadratic_force_field":
        assert engine.state.partial_charges.shape == (num_snapshots, num_atoms)
    else:
        assert engine.state.partial_charges is None
