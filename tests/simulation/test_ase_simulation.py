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

import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from mlip.data.chemical_system import ChemicalSystem
from mlip.graph import Graph
from mlip.simulation.ase.ase_simulation_engine import ASESimulationEngine
from mlip.simulation.ase.mlip_ase_calculator import MLIPForceFieldASECalculator
from mlip.simulation.configs.ase_config import TemperatureScheduleConfig
from mlip.simulation.enums import (
    MDIntegrator,
    SimulationType,
    TemperatureScheduleMethod,
)


@pytest.mark.parametrize(
    "force_field_name",
    ["quadratic_force_field", "lri_mace_force_field"],
)
@pytest.mark.parametrize("md_integrator", ["nvt_langevin", "npt_mc_langevin"])
def test_md_can_be_run_with_ase_backend(
    force_field_name, request, setup_system, md_integrator
) -> None:
    force_field = request.getfixturevalue(force_field_name)
    atoms = deepcopy(setup_system[0])
    if force_field.long_range_cutoff_distance is not None:
        atoms.info["charge"] = 1.0

    md_integrator = MDIntegrator(md_integrator)

    # Make dummy molecule_indices for system
    molecule_indices = [0] * 4 + [1] * 6

    md_config = ASESimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=md_integrator,
        num_steps=20,
        snapshot_interval=2,
        log_interval=2,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=10.0,
        pressure_bar=1.01325,
        molecule_indices=molecule_indices,
        edge_capacity_multiplier=1.25,
        friction=0.1,
        temperature_schedule_config=TemperatureScheduleConfig(
            method=TemperatureScheduleMethod.TRIANGLE,
            max_temperature=301.0,
            min_temperature=300.0,
            heating_period=4,
        ),
    )

    intermediate_steps = []

    def _mock_logger(state):
        intermediate_steps.append(state.step)

    engine = ASESimulationEngine(atoms, force_field, md_config)
    engine.attach_logger(_mock_logger)

    engine.run()

    assert engine.state.step == 20
    assert engine.state.compute_time_seconds > 0.0
    assert engine.state.temperature.shape == (11,)
    assert engine.state.kinetic_energy.shape == (11,)
    assert engine.state.positions.shape == (11, 10, 3)
    assert engine.state.forces.shape == (11, 10, 3)
    assert engine.state.velocities.shape == (11, 10, 3)
    assert intermediate_steps == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    if md_integrator.ensemble == "npt":
        assert engine.state.cell.shape == (11, 3, 3)


def test_ase_config_validation_works() -> None:
    with pytest.raises(ValidationError) as exc:
        ASESimulationEngine.Config(
            simulation_type=SimulationType.MD,
            md_integrator=MDIntegrator.NVT_LANGEVIN,
            num_steps=20,
            log_interval=1,
            timestep_fs=0.0,
            temperature_kelvin=300.0,
            box=None,
            edge_capacity_multiplier=1.25,
            friction=0.1,
        )

    assert "timestep_fs" in str(exc.value)
    assert str(exc.value).count("Input should be greater than 0") == 1


def test_md_can_be_restarted_from_velocities_with_ase_backend(
    quadratic_force_field,
    setup_system,
):
    _atoms = deepcopy(setup_system[0])

    velocities_to_restore = np.ones(_atoms.get_positions().shape)
    _atoms.set_velocities(velocities_to_restore)

    md_config = ASESimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator.NVT_LANGEVIN,
        num_steps=5,
        log_interval=1,
    )

    engine = ASESimulationEngine(_atoms, quadratic_force_field, md_config)
    engine.run()

    assert engine.state.velocities.shape[0] == 6
    assert np.allclose(engine.state.velocities[0], velocities_to_restore)

    for i in range(1, 5):
        assert not np.allclose(engine.state.velocities[i], velocities_to_restore)


@pytest.mark.parametrize("atoms_cell", [None, np.eye(3) * 10.0])
def test_ase_engine_sets_cell_from_config(
    quadratic_force_field, setup_system, atoms_cell
) -> None:
    atoms = deepcopy(setup_system[0])
    config_box_length = 25.0

    _atoms = deepcopy(atoms)
    if atoms_cell is None:  # If atoms have no cell, use config.box
        _atoms.set_cell(None)
        _atoms.set_pbc(None)
        target_cell = np.eye(3) * config_box_length
    else:  # If atoms have a cell, ignore config.box
        _atoms.set_cell(atoms_cell)
        _atoms.set_pbc(True)
        target_cell = atoms_cell

    md_config = ASESimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator.NVT_LANGEVIN,
        num_steps=1,
        box=config_box_length,
    )
    engine = ASESimulationEngine(_atoms, quadratic_force_field, md_config)
    assert (engine.atoms.get_cell() == target_cell).all()
    assert engine.atoms.get_pbc().all()


def test_ase_engine_stops_exploded_simulation_early(
    quadratic_force_field, setup_system
):
    _atoms = deepcopy(setup_system[0])
    _quadratic_ff = deepcopy(quadratic_force_field)

    def exploding_force_field_predictor_apply(model_params, batched_graph) -> Graph:
        batched_graph = batched_graph.replace_nodes(
            forces=batched_graph.nodes.positions * jnp.inf
        )
        return batched_graph.replace_globals(energy=jnp.inf)

    md_config = ASESimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator.NVT_LANGEVIN,
        num_steps=10,
    )
    _quadratic_ff.predictor.apply = exploding_force_field_predictor_apply
    engine = ASESimulationEngine(_atoms, _quadratic_ff, md_config)

    engine.run()
    # Stops after first step
    assert engine.state.step == 1
    assert engine.state.temperature.shape == (2,)
    assert engine.state.kinetic_energy.shape == (2,)
    assert engine.state.positions.shape == (2, 10, 3)
    assert engine.state.forces.shape == (2, 10, 3)
    assert engine.state.velocities.shape == (2, 10, 3)


@pytest.mark.parametrize(
    "force_field_name",
    ["mace_force_field", "lri_mace_force_field"],
)
def test_ase_calculator_forces_match_direct_force_field_call(
    force_field_name, request, setup_system
):
    """Forces from the ASE calculator must match a direct ForceField inference
    call on a graph built via the canonical
    `Graph.from_chemical_system(...)` path. The calculator's only job is to
    construct that graph (plus padding) on the fly per step — its output
    should be numerically equivalent on the real nodes. Covers both the
    plain (no long-range) and the LRI-enabled paths."""
    force_field = request.getfixturevalue(force_field_name)
    atoms = deepcopy(setup_system[0])
    if force_field.long_range_cutoff_distance is not None:
        atoms.info["charge"] = 1.0

    ref_graph = Graph.from_chemical_system(
        ChemicalSystem.from_ase_atoms(atoms),
        graph_cutoff_angstrom=force_field.cutoff_distance,
        long_range_cutoff_angstrom=force_field.long_range_cutoff_distance,
    )
    ref_pred = force_field(ref_graph)
    ref_forces = np.asarray(ref_pred.forces)

    calculator = MLIPForceFieldASECalculator(
        atoms,
        edge_capacity_multiplier=1.25,
        force_field=force_field,
    )
    calculator.calculate(atoms, properties=["forces", "energy"])
    sim_forces = calculator.results["forces"]

    assert sim_forces.shape == ref_forces.shape
    np.testing.assert_allclose(sim_forces, ref_forces, rtol=1e-4, atol=1e-4)


def test_ase_calculator_raises_when_charge_missing_and_embedding_enabled(
    total_charge_embedding_visnet_force_field, setup_system
):
    atoms = deepcopy(setup_system[0])
    atoms.info.pop("charge", None)

    md_config = ASESimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator("nvt_langevin"),
        num_steps=1,
        snapshot_interval=1,
        log_interval=1,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=10.0,
        edge_capacity_multiplier=1.25,
        friction=0.1,
    )

    md_config = md_config.model_copy(update={"set_none_charge_to_zero": False})
    with pytest.raises(ValueError, match="total charge embedding"):
        ASESimulationEngine(atoms, total_charge_embedding_visnet_force_field, md_config)


def test_ase_calculator_accepts_charge_when_embedding_enabled(
    total_charge_embedding_visnet_force_field, setup_system
):
    atoms = deepcopy(setup_system[0])
    atoms.info["charge"] = 0.0

    md_config = ASESimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator("nvt_langevin"),
        num_steps=1,
        snapshot_interval=1,
        log_interval=1,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=10.0,
        edge_capacity_multiplier=1.25,
        friction=0.1,
    )

    engine = ASESimulationEngine(
        atoms, total_charge_embedding_visnet_force_field, md_config
    )
    engine.run()  # No error.


def test_ase_calculator_accepts_set_none_charge_to_zero_when_embedding_enabled(
    total_charge_embedding_visnet_force_field, setup_system
):
    atoms = deepcopy(setup_system[0])
    atoms.info.pop("charge", None)

    md_config = ASESimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator("nvt_langevin"),
        num_steps=1,
        snapshot_interval=1,
        log_interval=1,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=10.0,
        edge_capacity_multiplier=1.25,
        friction=0.1,
        set_none_charge_to_zero=True,
    )

    engine = ASESimulationEngine(
        atoms, total_charge_embedding_visnet_force_field, md_config
    )
    engine.run()  # No error: charge becomes 0.


def test_ase_calculator_derives_charge_from_partial_charges_when_embedding_enabled(
    total_charge_embedding_visnet_force_field, setup_system
):
    atoms = deepcopy(setup_system[0])
    atoms.info.pop("charge", None)
    atoms.info["partial_charges"] = np.zeros(len(atoms))

    md_config = ASESimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator("nvt_langevin"),
        num_steps=1,
        snapshot_interval=1,
        log_interval=1,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=10.0,
        edge_capacity_multiplier=1.25,
        friction=0.1,
    )

    engine = ASESimulationEngine(
        atoms, total_charge_embedding_visnet_force_field, md_config
    )
    engine.run()  # No error: charge is derived from partial_charges (sum = 0).


def test_ase_simulation_reproducible(quadratic_force_field, setup_system):
    atoms = deepcopy(setup_system[0])

    md_config = ASESimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator.NVT_LANGEVIN,
        num_steps=5,
    )

    final_states = []
    for _ in range(2):
        _atoms = deepcopy(atoms)
        _quadratic_ff = deepcopy(quadratic_force_field)
        engine = ASESimulationEngine(_atoms, _quadratic_ff, md_config)
        engine.run()
        final_states.append(engine.state)

    for key in [
        "positions",
        "forces",
        "velocities",
        "kinetic_energy",
        "temperature",
        "step",
    ]:
        assert np.allclose(
            getattr(final_states[0], key), getattr(final_states[1], key)
        ), f"'{key}' does not match."
