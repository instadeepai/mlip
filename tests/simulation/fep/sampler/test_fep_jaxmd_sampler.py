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
from unittest.mock import patch

import numpy as np
import pytest

import mlip.simulation.fep.models.alchemical_models as alchemical_models_module
from mlip.models.force_field import ForceField
from mlip.simulation.enums import MDIntegrator, SimulationType
from mlip.simulation.fep.models import AlchemicalPredictor, LinearAlchemicalPredictor
from mlip.simulation.fep.models.alchemical_models import AlchemicalMLIPNetwork
from mlip.simulation.fep.sampler import (
    FEPSimulationSampler,
    FEPSimulationSamplerConfig,
)
from mlip.simulation.fep.sampler.alchemical_jaxmd_engine import (
    AlchemicalJaxMDSimulationEngine,
)
from mlip.simulation.jax_md.helpers import VELOCITY_CONVERSION_FACTOR

N_STEPS = 4
SNAPSHOT_INTERVAL = 2
N_ATOMS = 10
# Specify first 3 atoms as one molecule, rest as another
MOLECULE_INDICES = [0] * (N_ATOMS - 3) + [1] * 3

ALCHEMICAL_ATOM_INDICES = np.arange(2)
TWO_LAMBDAS = np.array([[0.5, 0.5], [1.0, 0.2]])


@pytest.fixture(scope="module")
def base_force_field(quadratic_force_field) -> ForceField:
    return quadratic_force_field


@pytest.fixture(scope="module")
def atoms(setup_system):
    init_conf, _ = setup_system
    return deepcopy(init_conf)


def _prepare_sampler(
    atoms,
    force_field,
    lambda_values,
    *,
    md_integrator=MDIntegrator.NPT_MC_LANGEVIN,
    **config_overrides,
) -> FEPSimulationSampler:
    engine_config_kwargs = dict(
        simulation_type=SimulationType.MD,
        md_integrator=md_integrator,
        num_steps=N_STEPS,
        snapshot_interval=SNAPSHOT_INTERVAL,
        box=10.0,
        num_episodes=2,
        molecule_indices=MOLECULE_INDICES,
        pressure_bar=1.0,
        barostat_update_interval=2,
    )
    sampler_config_kwargs = dict(
        checkpoint_interval_episodes=1,
        num_equilibration_episodes=0,
        use_alchemical_mlip=False,
    )
    sampler_config_kwargs.update(config_overrides)
    return FEPSimulationSampler(
        atoms,
        force_field,
        FEPSimulationSamplerConfig(
            simulation_config=AlchemicalJaxMDSimulationEngine.Config(
                **engine_config_kwargs
            ),
            alchemical_atom_indices=ALCHEMICAL_ATOM_INDICES,
            lambda_edge_values=np.asarray(lambda_values)[:, 0],
            lambda_repulsion_values=np.asarray(lambda_values)[:, 1],
            **sampler_config_kwargs,
        ),
    )


@pytest.fixture(scope="module")
def two_lambda_sampler(atoms, base_force_field) -> FEPSimulationSampler:
    return _prepare_sampler(atoms, base_force_field, TWO_LAMBDAS)


@pytest.mark.parametrize("use_alchemical_mlip", [False, True])
def test_fep_sampler_builds_alchemical_force_field(
    atoms, base_force_field, use_alchemical_mlip, monkeypatch
):
    quadratic_cls = type(base_force_field.predictor.mlip_network)
    alchemical_quadratic_cls = type(
        "AlchemicalQuadraticMLIP", (AlchemicalMLIPNetwork, quadratic_cls), {}
    )
    monkeypatch.setitem(
        alchemical_models_module._ALCHEMICAL_MLIP_NETWORKS,
        quadratic_cls,
        alchemical_quadratic_cls,
    )

    sampler = _prepare_sampler(
        atoms, base_force_field, TWO_LAMBDAS, use_alchemical_mlip=use_alchemical_mlip
    )
    predictor = sampler._engines[0]._force_field.predictor
    if use_alchemical_mlip:
        assert isinstance(predictor, AlchemicalPredictor)
        assert not isinstance(predictor, LinearAlchemicalPredictor)
    else:
        assert isinstance(predictor, LinearAlchemicalPredictor)


def test_fep_sampler_runs(atoms, two_lambda_sampler) -> None:
    """FEP sampler runs to completion and produces correct output shapes per window.

    Also tests that an overflow is prioritised over an explosion, in which case
    `_global_reallocate_neighbors` is run, reallocates, then the run continues.
    """
    num_atoms = len(atoms)
    num_snapshots = N_STEPS // SNAPSHOT_INTERVAL
    sampler = two_lambda_sampler

    # Trigger overflow and explosion on the 3rd overflow check.
    overflow_responses = ([False] * 2) + [True] + ([False] * 100)
    exploded_responses = ([False] * 2) + [True] + ([False] * 100)

    def mock_did_overflow(state):
        return overflow_responses.pop(0)

    def mock_did_explode(state):
        return exploded_responses.pop(0)

    with (
        patch.object(
            AlchemicalJaxMDSimulationEngine,
            "_did_neighbor_buffer_overflow",
            side_effect=mock_did_overflow,
        ),
        patch.object(
            AlchemicalJaxMDSimulationEngine,
            "_has_simulation_exploded",
            side_effect=mock_did_explode,
        ),
        patch.object(
            sampler,
            "_global_reallocate_neighbors",
            wraps=sampler._global_reallocate_neighbors,
        ) as spy_reallocate,
    ):
        sampler.run()
        spy_reallocate.assert_called_once()

    assert len(sampler.replica_exchange_log) > 0

    engine_states = sampler.engine_states
    assert len(engine_states) == len(TWO_LAMBDAS)

    for i in range(len(TWO_LAMBDAS)):
        state = engine_states[i]

        assert state.step == N_STEPS
        assert state.compute_time_seconds > 0.0
        assert state.temperature.shape == (num_snapshots,)
        assert state.kinetic_energy.shape == (num_snapshots,)
        assert state.positions.shape == (num_snapshots, num_atoms, 3)
        assert state.forces.shape == (num_snapshots, num_atoms, 3)
        assert state.velocities.shape == (num_snapshots, num_atoms, 3)
        assert state.cell.shape == (num_snapshots, 3, 3)

        assert np.all(np.isfinite(state.positions))
        assert np.all(np.isfinite(state.forces))
        assert np.all(np.isfinite(state.per_lambda_energies))

        n_lambdas = len(TWO_LAMBDAS)
        assert state.per_lambda_energies.shape == (num_snapshots, n_lambdas)

        assert state.final_positions.shape == (num_atoms, 3)
        assert state.final_cell.shape == (3, 3)
        assert state.final_velocities.shape == (num_atoms, 3)


def test_fep_sampler_independent_mode(two_lambda_sampler, monkeypatch) -> None:
    """Sampler runs without replica exchange when use_replica_exchange=False."""
    sampler = two_lambda_sampler
    monkeypatch.setattr(sampler._config, "use_replica_exchange", False)
    with patch.object(sampler, "_perform_replica_exchange") as mock_repex:
        sampler.run()
        mock_repex.assert_not_called()

    for i in range(len(TWO_LAMBDAS)):
        assert sampler._engines[i].state.step == N_STEPS


def test_run_loop_terminates_early_on_explosion(two_lambda_sampler):
    sampler = two_lambda_sampler

    with (
        patch.object(
            AlchemicalJaxMDSimulationEngine,
            "_did_neighbor_buffer_overflow",
            return_value=False,
        ),
        patch.object(
            AlchemicalJaxMDSimulationEngine,
            "_has_simulation_exploded",
            return_value=True,
        ),
        patch.object(sampler, "_perform_replica_exchange") as mock_repex,
    ):
        sampler.run()
        mock_repex.assert_not_called()

    for i in range(len(TWO_LAMBDAS)):
        assert sampler._engines[i].state.step == 2


def test_sampler_init_with_list_atoms_uses_per_lambda_state(
    setup_system, base_force_field
) -> None:
    """Per-lambda positions and velocities from list[Atoms] are restored (NVT)."""
    init_conf, _ = setup_system
    lambda_values = np.array([[0.0, 0.0], [0.5, 0.5]])
    n_atoms = len(init_conf)

    offsets = [np.zeros(3), np.array([0.1, 0.0, 0.0])]
    per_lambda_atoms = []
    for i in range(len(lambda_values)):
        a = deepcopy(init_conf)
        a.set_positions(a.get_positions() + offsets[i])
        per_lambda_atoms.append(a)

    prescribed_vel = np.ones((n_atoms, 3)) * 0.01
    per_lambda_atoms[1].set_velocities(prescribed_vel)

    sampler = _prepare_sampler(
        per_lambda_atoms,
        base_force_field,
        lambda_values,
        md_integrator=MDIntegrator.NVT_LANGEVIN,
    )

    for i in range(len(lambda_values)):
        expected_pos = per_lambda_atoms[i].get_positions()
        actual_pos = np.asarray(
            sampler._engines[i].get_internal_state().jax_md_state.position
        )
        np.testing.assert_allclose(actual_pos, expected_pos, atol=1e-5)

    state_1 = sampler._engines[1].get_internal_state().jax_md_state
    mass_1 = np.asarray(state_1.mass)
    actual_momentum = np.asarray(state_1.momentum)
    expected_momentum = prescribed_vel * VELOCITY_CONVERSION_FACTOR * mass_1
    np.testing.assert_allclose(actual_momentum, expected_momentum, rtol=1e-4)

    momentum_0 = np.asarray(
        sampler._engines[0].get_internal_state().jax_md_state.momentum
    )
    assert not np.allclose(momentum_0, expected_momentum)
