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
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import read as ase_read_atoms

from mlip.data import ChemicalSystem
from mlip.graph import Graph
from mlip.simulation.enums import (
    MDIntegrator,
    SimulationType,
)
from mlip.simulation.metadynamics.config import MetadynamicsConfig
from mlip.simulation.metadynamics.jax_md_metad_engine import (
    JaxMDMetadynamicsSimulationEngine,
)
from mlip.simulation.metadynamics.potential_terms import (
    AngleCVConfig,
    AngleWallConfig,
    CoordinationNumberCVConfig,
    DihedralCVConfig,
    DistanceCVConfig,
    DistanceWallConfig,
    PositionalRestraintConfig,
)

SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent / "sample_data"

NUM_STEPS = 8
SNAPSHOT_INTERVAL = 2


@pytest.fixture(scope="session")
def setup_metadynamics_system(dataset_info) -> tuple[Atoms, Graph]:
    """Returns an `(atoms, graph)` tuple for the alanine-dipeptide system.

    Allowed atomic numbers are [1, 6, 7, 8] <=> ["H","C","N","O"].
    """
    atoms = ase_read_atoms(SAMPLE_DATA_DIR / "alanine_dipeptide_vacuum.xyz")
    graph_cutoff_angstrom = dataset_info.graph_cutoff_angstrom

    chemical_system = ChemicalSystem.from_ase_atoms(atoms)
    graph = Graph.from_chemical_system(chemical_system, graph_cutoff_angstrom)

    return atoms, graph


def get_metadynamics_config(cv_type: str) -> MetadynamicsConfig:
    # CVs for alanine dipeptide backbone
    distance_cv = DistanceCVConfig(atom_indices_1=(6,), atom_indices_2=(16,))
    distance_2_cv = DistanceCVConfig(atom_indices_1=(0, 2), atom_indices_2=(16, 17))
    angle_cv = AngleCVConfig(atom_indices=(6, 7, 8))
    dihedral_cv = DihedralCVConfig(atom_indices=(0, 6, 7, 8))
    coordnum_cv = CoordinationNumberCVConfig(central_idx=7, element="C")

    cv_map = {
        "distance": ([distance_cv], [0.1]),
        "distance_distance": ([distance_cv, distance_2_cv], [0.1, 0.5]),
        "distance_coordnum": ([distance_cv, coordnum_cv], [0.1, 0.5]),
        "angle_dihedral": ([angle_cv, dihedral_cv], [0.15, 0.3]),
    }
    bias_cvs, bias_sigmas = cv_map[cv_type]

    walls = [
        DistanceWallConfig(
            atom_indices_1=[6], atom_indices_2=[16], lower=2.0, upper=5.5, kappa=50.0
        ),
        AngleWallConfig(
            atom_indices=(6, 7, 8), lower_rad=1.6, upper_rad=2.3, kappa=100.0
        ),
    ]
    restraints = [PositionalRestraintConfig(atom_indices=[0, 7, 17], kappa=100.0)]

    return MetadynamicsConfig(
        bias_cvs=bias_cvs,
        bias_sigmas=bias_sigmas,
        walls=walls,
        restraints=restraints,
        bias_factor=15.0,
        deposition_interval=2,
        max_gaussians=20000,
        initial_height=1.0,
    )


@pytest.mark.parametrize("md_integrator", ["nvt_langevin", "npt_mc_langevin"])
@pytest.mark.parametrize(
    "cv_type", ["distance", "distance_distance", "distance_coordnum", "angle_dihedral"]
)
def test_run_with_jax_md_metadynamics_engine(
    quadratic_force_field, setup_metadynamics_system, md_integrator, cv_type
):
    metadynamics_config = get_metadynamics_config(cv_type)
    force_field = quadratic_force_field
    atoms, _ = setup_metadynamics_system
    atoms = deepcopy(atoms)
    num_atoms = len(atoms)
    num_cvs = len(get_metadynamics_config(cv_type).bias_cvs)

    md_integrator = MDIntegrator(md_integrator)

    # Dummy molecule indices: treat each half as a separate molecule for NPT
    molecule_indices = [0] * (num_atoms // 2) + [1] * (num_atoms // 2)
    md_integrator = MDIntegrator(md_integrator)

    md_config = JaxMDMetadynamicsSimulationEngine.Config(
        metadynamics_config=metadynamics_config,
        simulation_type=SimulationType.MD,
        md_integrator=md_integrator,
        num_steps=NUM_STEPS,
        snapshot_interval=SNAPSHOT_INTERVAL,
        num_episodes=2,
        timestep_fs=1.0,
        box=10.0,
        molecule_indices=molecule_indices,
    )

    intermediate_steps = []

    def _mock_logger(state):
        intermediate_steps.append(state.step)

    engine = JaxMDMetadynamicsSimulationEngine(atoms, force_field, md_config)
    engine.attach_logger(_mock_logger)

    engine.run()

    num_frames = NUM_STEPS // SNAPSHOT_INTERVAL
    assert engine.state.step == 8
    assert engine.state.compute_time_seconds > 0.0
    assert engine.state.temperature.shape == (num_frames,)
    assert engine.state.kinetic_energy.shape == (num_frames,)
    assert engine.state.positions.shape == (num_frames, num_atoms, 3)
    assert engine.state.forces.shape == (num_frames, num_atoms, 3)
    assert engine.state.velocities.shape == (num_frames, num_atoms, 3)
    assert intermediate_steps == [4, 8]
    if md_integrator.ensemble == "npt":
        assert engine.state.cell.shape == (num_frames, 3, 3)

    assert engine.state.bias_potential.shape == (num_frames,)
    assert engine.state.bias_cv_values.shape == (num_frames, num_cvs)
    # Hills deposited every 2 steps, but not at step 0
    assert engine.state.gaussian_heights.shape == (3,)
    assert engine.state.gaussian_centers.shape == (3, num_cvs)
