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
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from mlip.data.chemical_system import ChemicalSystem
from mlip.graph import Graph
from mlip.simulation.configs.simulation_config import TemperatureScheduleConfig
from mlip.simulation.enums import (
    MDIntegrator,
    SimulationType,
    TemperatureScheduleMethod,
)
from mlip.simulation.jax_md.jax_md_simulation_engine import JaxMDSimulationEngine


@pytest.mark.parametrize(
    "force_field_name", ["mace_force_field", "lri_mace_force_field"]
)
def test_jax_md_step_zero_forces_match_direct_force_field_call(
    force_field_name, request, setup_system
):
    """Forces logged at step 0 by the JAX-MD engine must match a direct
    ForceField inference call on a graph built via the canonical
    `Graph.from_chemical_system(...)` path. The engine uses `displ_fun`
    while the reference uses `shifts`, but for a non-PBC system both reduce
    to `positions[r] - positions[s]`, so results must agree numerically.
    Covers both the plain (no long-range) and the LRI-enabled paths."""
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

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        num_steps=1,
        snapshot_interval=1,
        num_episodes=1,
        timestep_fs=0.5,
    )
    engine = JaxMDSimulationEngine(atoms, force_field, md_config)
    engine.run()

    # state.forces[0] is logged before the step is applied, in eV/Å
    sim_step0_forces = np.asarray(engine.state.forces[0])
    assert sim_step0_forces.shape == ref_forces.shape
    np.testing.assert_allclose(sim_step0_forces, ref_forces, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "force_field_name", ["quadratic_force_field", "lri_mace_force_field"]
)
@pytest.mark.parametrize("md_integrator", ["nvt_langevin", "npt_mc_langevin"])
def test_md_can_be_run_with_jax_md_backend(
    force_field_name, request, setup_system, md_integrator
):
    force_field = request.getfixturevalue(force_field_name)
    atoms, _ = setup_system
    atoms = deepcopy(atoms)
    if force_field.long_range_cutoff_distance is not None:
        atoms.info["charge"] = 1.0

    md_integrator = MDIntegrator(md_integrator)

    # Make dummy molecule_indices for system
    molecule_indices = [0] * 4 + [1] * 6
    md_integrator = MDIntegrator(md_integrator)

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=md_integrator,
        num_steps=20,
        snapshot_interval=2,
        num_episodes=5,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=10.0,
        pressure_bar=1.01325,
        molecule_indices=molecule_indices,
        edge_capacity_multiplier=1.25,
    )

    intermediate_steps = []

    def _mock_logger(state):
        intermediate_steps.append(state.step)

    engine = JaxMDSimulationEngine(atoms, force_field, md_config)
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
    if md_integrator.ensemble == "npt":
        assert engine.state.cell.shape == (10, 3, 3)


def test_jax_md_engine_rejects_cutoff_exceeding_half_box(
    lri_mace_force_field, setup_system
):
    """Cutoffs > L/2 cannot be handled by the standard NL, and the multi-image
    fallback is numerically non-conservative. The engine must raise instead
    of silently falling back to a broken code path.
    """
    force_field = deepcopy(lri_mace_force_field)
    force_field.predictor.mlip_network.dataset_info = (
        force_field.predictor.mlip_network.dataset_info.model_copy(
            update={"long_range_cutoff_angstrom": 20.0}
        )
    )
    atoms = deepcopy(setup_system[0])
    atoms.info["charge"] = 1.0
    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        num_steps=1,
        snapshot_interval=1,
        num_episodes=1,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=10.0,  # 20 > 10 / 2, triggers the guard
        edge_capacity_multiplier=1.25,
    )
    with pytest.raises(ValueError, match="exceeds half"):
        JaxMDSimulationEngine(atoms, force_field, md_config)


@pytest.mark.parametrize("md_integrator", ["nvt_langevin", "npt_mc_langevin"])
def test_batched_md_can_be_run_with_jax_md_backend_for_three_identical_systems(
    quadratic_force_field, setup_system, md_integrator
):
    atoms, _ = setup_system
    atoms = deepcopy(atoms)
    md_integrator = MDIntegrator(md_integrator)

    # Run three systems at the same time
    systems = [atoms, deepcopy(atoms), deepcopy(atoms)]

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=md_integrator,
        num_steps=20,
        snapshot_interval=2,
        num_episodes=5,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=10.0,
        edge_capacity_multiplier=1.25,
        molecule_indices=[[0] * len(atoms)] * 3,
    )

    intermediate_steps = []

    def _mock_logger(state):
        intermediate_steps.append(state.step)

    engine = JaxMDSimulationEngine(systems, quadratic_force_field, md_config)
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
        for key in [
            "positions",
            "forces",
            "velocities",
            "kinetic_energy",
            "temperature",
        ]:
            assert np.allclose(
                getattr(engine.state, key)[i], getattr(engine.state, key)[0], atol=1e-5
            )

        if md_integrator.ensemble == "npt":
            assert np.allclose(engine.state.cell[i], engine.state.cell[0], atol=1e-5)


@pytest.mark.parametrize("md_integrator", ["nvt_langevin", "npt_mc_langevin"])
def test_batched_md_can_be_run_with_jax_md_backend_for_two_different_systems(
    quadratic_force_field, setup_system, md_integrator
):
    atoms, _ = setup_system
    atoms = deepcopy(atoms)
    md_integrator = MDIntegrator(md_integrator)
    atoms.set_cell([10.0, 10.0, 10.0])

    # Add one atom to the molecule
    mod_numbers = list(atoms.numbers) + [6]
    mod_pos = np.concatenate(
        [atoms.get_positions(), np.array([[0.1, 0.2, 0.3]])], axis=0
    )
    mod_atoms = ase.Atoms(numbers=mod_numbers, positions=mod_pos)
    mod_atoms.set_cell([8.0, 8.0, 8.0])

    n_atoms, n_mod_atoms = len(atoms), len(mod_atoms)
    base_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=md_integrator,
        num_steps=20,
        snapshot_interval=2,
        num_episodes=5,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=None,
        edge_capacity_multiplier=1.25,
    )
    batched_config = base_config.model_copy(
        update={"molecule_indices": [[0] * n_atoms, [0] * n_mod_atoms]}
    )
    atoms_config = base_config.model_copy(update={"molecule_indices": [0] * n_atoms})
    mod_atoms_config = base_config.model_copy(
        update={"molecule_indices": [0] * n_mod_atoms}
    )

    engine_1 = JaxMDSimulationEngine(
        [atoms, mod_atoms], quadratic_force_field, batched_config
    )
    engine_1.run()

    assert engine_1.state.step == 20
    assert len(engine_1.state.positions) == 2
    assert engine_1.state.positions[0].shape == (10, 10, 3)
    assert engine_1.state.positions[1].shape == (10, 11, 3)

    engine_2 = JaxMDSimulationEngine(atoms, quadratic_force_field, atoms_config)
    engine_2.run()

    engine_3 = JaxMDSimulationEngine(mod_atoms, quadratic_force_field, mod_atoms_config)
    engine_3.run()

    state_1, state_2, state_3 = engine_1.state, engine_2.state, engine_3.state

    def _assert_engines_match(key, atol, rtol):
        kwargs = {
            k: v for k, v in {"atol": atol, "rtol": rtol}.items() if v is not None
        }
        assert np.allclose(getattr(state_1, key)[0], getattr(state_2, key), **kwargs)
        assert np.allclose(getattr(state_1, key)[1], getattr(state_3, key), **kwargs)

    tol = 1e-3
    for key in ["positions", "forces", "velocities", "kinetic_energy"]:
        _assert_engines_match(key, atol=tol, rtol=None)

    # Use relative tolerance for temperatures as these values are quite large
    _assert_engines_match("temperature", atol=None, rtol=tol)

    if md_integrator.ensemble == "npt":
        _assert_engines_match("cell", atol=tol, rtol=None)


@pytest.mark.parametrize(
    "force_field_name", ["quadratic_force_field", "lri_mace_force_field"]
)
@pytest.mark.parametrize("md_integrator", ["nvt_langevin", "npt_mc_langevin"])
def test_batched_and_regular_md_yield_same_results(
    force_field_name, request, setup_system, md_integrator
):
    force_field = request.getfixturevalue(force_field_name)
    atoms, _ = setup_system
    atoms = deepcopy(atoms)
    md_integrator = MDIntegrator(md_integrator)

    if force_field.long_range_cutoff_distance is not None:
        atoms.info["charge"] = 1.0

    base_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=md_integrator,
        num_steps=20,
        snapshot_interval=2,
        num_episodes=5,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=10.0,
        edge_capacity_multiplier=1.25,
    )
    batched_config = base_config.model_copy(
        update={"molecule_indices": [[0] * len(atoms)]}
    )
    single_config = base_config.model_copy(
        update={"molecule_indices": [0] * len(atoms)}
    )

    engine_1 = JaxMDSimulationEngine([atoms], force_field, batched_config)
    engine_1.run()

    engine_2 = JaxMDSimulationEngine(atoms, force_field, single_config)
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
    if md_integrator.ensemble == "npt":
        assert np.allclose(engine_1.state.cell[0], engine_2.state.cell, atol=tol)


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
    quadratic_force_field, setup_system
):
    atoms, _ = setup_system
    _atoms = deepcopy(atoms)

    velocities_to_restore = np.ones(_atoms.get_positions().shape)
    _atoms.set_velocities(velocities_to_restore)

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator.NVT_LANGEVIN,
        num_steps=5,
        num_episodes=1,
    )

    engine = JaxMDSimulationEngine(_atoms, quadratic_force_field, md_config)
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


def test_jax_md_engine_stops_exploded_simulation_early(
    quadratic_force_field, setup_system
):
    atoms, _ = setup_system
    _atoms = deepcopy(atoms)
    _quadratic_ff = deepcopy(quadratic_force_field)

    def exploding_force_field_predictor_apply(model_params, batched_graph) -> Graph:
        batched_graph = batched_graph.replace_nodes(
            forces=batched_graph.nodes.positions * jnp.inf
        )
        return batched_graph.replace_globals(energy=jnp.inf)

    num_steps, num_episodes = 20, 5
    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator.NVT_LANGEVIN,
        num_steps=num_steps,
        snapshot_interval=1,
        num_episodes=num_episodes,
    )
    _quadratic_ff.predictor.apply = exploding_force_field_predictor_apply
    engine = JaxMDSimulationEngine(_atoms, _quadratic_ff, md_config)

    engine.run()
    # Stops after first episode
    expected_steps = num_steps // num_episodes
    assert engine.state.step == expected_steps
    assert engine.state.temperature.shape == (expected_steps,)
    assert engine.state.kinetic_energy.shape == (expected_steps,)
    assert engine.state.positions.shape == (expected_steps, 10, 3)
    assert engine.state.forces.shape == (expected_steps, 10, 3)
    assert engine.state.velocities.shape == (expected_steps, 10, 3)


def test_jax_md_simulation_reproducible(quadratic_force_field, setup_system):
    atoms, _ = setup_system

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        num_steps=5,
    )

    final_states = []
    for _ in range(2):
        _atoms = deepcopy(atoms)
        _mace_ff = deepcopy(quadratic_force_field)
        engine = JaxMDSimulationEngine(_atoms, _mace_ff, md_config)
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


def test_jax_md_engine_raises_when_charge_missing_and_embedding_enabled(
    total_charge_embedding_visnet_force_field, setup_system
):
    atoms = deepcopy(setup_system[0])
    atoms.info.pop("charge", None)

    # Make dummy molecule_indices for system.
    molecule_indices = [0] * 4 + [1] * 6

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator("nvt_langevin"),
        num_steps=1,
        snapshot_interval=1,
        num_episodes=1,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=10.0,
        molecule_indices=molecule_indices,
        edge_capacity_multiplier=1.25,
    )

    md_config = md_config.model_copy(update={"set_none_charge_to_zero": False})
    with pytest.raises(ValueError, match="total charge embedding"):
        JaxMDSimulationEngine(
            atoms, total_charge_embedding_visnet_force_field, md_config
        )


def test_jax_md_engine_runs_when_charge_present_with_embedding_enabled(
    total_charge_embedding_visnet_force_field, setup_system
):
    atoms = deepcopy(setup_system[0])
    atoms.info["charge"] = 0.0

    molecule_indices = [0] * 4 + [1] * 6

    md_config = JaxMDSimulationEngine.Config(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator("nvt_langevin"),
        num_steps=1,
        snapshot_interval=1,
        num_episodes=1,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
        box=10.0,
        molecule_indices=molecule_indices,
        edge_capacity_multiplier=1.25,
    )

    JaxMDSimulationEngine(
        atoms, total_charge_embedding_visnet_force_field, md_config
    )  # No error at init.
