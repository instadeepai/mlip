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

import functools
import logging
from dataclasses import replace as dataclass_replace
from typing import Callable

import ase
import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from jax_md import dataclasses
from jax_md.partition import NeighborList

from mlip.graph import Graph
from mlip.models.inference_context import apply_inference_context_to_graph
from mlip.simulation.configs.jax_md_config import JaxMDSimulationConfig
from mlip.simulation.fep.alchemical_graph import AlchemicalGraph
from mlip.simulation.fep.enums import FEPStage
from mlip.simulation.fep.models import AlchemicalForceField
from mlip.simulation.fep.sampler.states import (
    FEPEpisodeLog,
    FEPSimulationState,
    FEPSystemState,
)
from mlip.simulation.jax_md.helpers import (
    init_simulation_algorithm,
    is_episode_log,
    is_neighbor_list,
    is_system_state,
    update_graph_in_simulation_step,
)
from mlip.simulation.jax_md.jax_md_simulation_engine import (
    JaxMDSimulationEngine,
    UpdateGraphInSimStepFun,
)
from mlip.simulation.jax_md.states import JaxMDSimulationState
from mlip.simulation.montecarlo_barostat import create_high_precision_force_field
from mlip.simulation.temperature_scheduling import get_temperature_schedule
from mlip.utils.jax_utils import high_precision_matmul_context

logger = logging.getLogger("mlip")


class AlchemicalJaxMDSimulationEngine(JaxMDSimulationEngine):
    """Alchemical simulation engine using the JAX-MD backend.

    An extension of JaxMDSimulationEngine that:
        - Uses FEPSystemState to carry lambda_values through the simulation loop.
        - Uses FEPEpisodeLog to additionally log alchemical energies at each snapshot.
        - Computes per-lambda energies at each reference lambda using a separate
            high-precision force field call.
        - Exposes run_compiled_episode() / get_internal_state() / set_internal_state()
            / build_per_device_step_fun(), used by `FEPSimulationSampler` for FEP.
    """

    Config = JaxMDSimulationConfig
    simulation_state_class = FEPSimulationState

    def __init__(
        self,
        atoms: ase.Atoms,
        force_field: AlchemicalForceField,
        config: JaxMDSimulationConfig,
        alchemical_atom_indices: np.ndarray,
        lambda_values: np.ndarray,
        reference_lambdas: np.ndarray,
        alchemical_energy_batch_size: int | None,
    ) -> None:
        """Initialize an alchemical simulation engine.

        Args:
            atoms: System to simulate.
            force_field: The alchemical force field.
            config: Simulation configuration.
            alchemical_atom_indices: Indices of the alchemical atoms.
            lambda_values: [edge_weight, repulsion_weight].
            reference_lambdas: All lambda values used across the FEP calculation
                (required to log energies at each lambda).
            alchemical_energy_batch_size: Batch size to use when computing per-lambda
                energies. See `FEPSimulationSamplerConfig` for details.
        """
        if isinstance(atoms, list):
            raise ValueError(
                "AlchemicalJaxMDSimulationEngine does not support batched simulations."
            )
        self._internal_state: JaxMDSimulationState | None = None
        self._alchemical_atom_indices = alchemical_atom_indices
        self._lambda_values = jnp.array(lambda_values).reshape(1, 2)
        self._steps_per_episode = config.num_steps // config.num_episodes
        self._reference_lambdas = jnp.array(reference_lambdas)
        self._alchemical_energy_batch_size = alchemical_energy_batch_size
        super().__init__(atoms, force_field, config)

    @property
    def fep_stage(self) -> FEPStage | None:
        if not hasattr(self._force_field.predictor, "fep_stage"):
            return None
        edge_weight = float(self._lambda_values[0, 0])
        return FEPStage.from_edge_weight(edge_weight)

    def _system_state_from_neighbors(
        self,
        neighbors: NeighborList,
        long_range_neighbors: NeighborList | None = None,
    ) -> FEPSystemState:
        if long_range_neighbors is not None:
            return jax.tree.map(
                lambda n, lr: FEPSystemState(
                    neighbors=n,
                    long_range_neighbors=lr,
                    lambda_values=self._lambda_values,
                ),
                neighbors,
                long_range_neighbors,
                is_leaf=is_neighbor_list,
            )
        return jax.tree.map(
            lambda n: FEPSystemState(neighbors=n, lambda_values=self._lambda_values),
            neighbors,
            is_leaf=is_neighbor_list,
        )

    def _init_base_graph(
        self,
        atoms: Atoms,
        neighbors: NeighborList,
        long_range_neighbors: NeighborList | None = None,
    ) -> AlchemicalGraph:
        """Creates an `AlchemicalGraph` with lambda and alchemical globals set."""
        graph = super()._init_base_graph(atoms, neighbors, long_range_neighbors)
        graph = AlchemicalGraph.from_graph(graph)
        return graph.replace_globals(
            alchemical_lambda=self._lambda_values,
            alchemical_atom_indices=self._alchemical_atom_indices,
        )

    def _setup_sim_functions(self, base_graph: Graph) -> tuple[Callable, Callable]:
        """Extends parent to also build compute_alchemical_values_fun."""
        make_model_calculate_fun = functools.partial(
            self._get_model_calculate_fun,
            graph=base_graph,
        )
        sim_init_fun, sim_apply_fun = init_simulation_algorithm(
            make_model_calculate_fun,
            self._force_field,
            self._shift_fun,
            self._config,
            self._initial_box,
            self._fractional_coordinates,
        )

        precise_force_field = create_high_precision_force_field(self._force_field)
        precise_predictor = precise_force_field.predictor
        model_params = precise_force_field.params
        precise_inference_context = (
            precise_force_field.inference_context.resolve(
                precise_force_field.dataset_info
            )
            if precise_force_field.inference_context is not None
            else None
        )
        static_base_graph = base_graph
        _update_graph = self._get_update_graph_in_sim_step_fun()
        _is_batched = False
        _num_unique_edge_weights = len(
            np.unique(np.asarray(self._reference_lambdas)[:, 0])
        )
        _alchemical_energy_batch_size = self._alchemical_energy_batch_size

        def _compute_alchemical_values_pure(
            positions,
            system_state,
            box,
            reference_lambdas,
            params,
            num_unique_edge_weights,
            batch_size,
        ):
            updated_graph = _update_graph(
                system_state, positions, static_base_graph, _is_batched, box=box
            )
            if precise_inference_context is not None:
                updated_graph = apply_inference_context_to_graph(
                    updated_graph, inference_context=precise_inference_context
                )
            with high_precision_matmul_context():
                return precise_predictor.apply(
                    params,
                    updated_graph,
                    reference_lambdas,
                    num_unique_edge_weights,
                    batch_size,
                    method=precise_predictor.compute_alchemical_values,
                )

        compute_alchemical_values_fun = functools.partial(
            jax.jit(
                _compute_alchemical_values_pure,
                static_argnames=("num_unique_edge_weights", "batch_size"),
            ),
            reference_lambdas=self._reference_lambdas,
            params=model_params,
            num_unique_edge_weights=_num_unique_edge_weights,
            batch_size=_alchemical_energy_batch_size,
        )

        pure_simulation_step_fun = functools.partial(
            self._simulation_step_fun,
            apply_fun=sim_apply_fun,
            temperature_schedule=get_temperature_schedule(
                self._config.temperature_schedule_config, self._config.num_steps
            ),
            is_md_simulation=self.is_md_simulation,
            is_npt_simulation=self.is_npt_simulation,
            initial_box=self._initial_box,
            compute_alchemical_values_fun=compute_alchemical_values_fun,
            snapshot_interval=self._config.snapshot_interval,
            steps_per_episode=self._steps_per_episode,
            use_fractional_coords=self._fractional_coordinates,
        )
        return sim_init_fun, pure_simulation_step_fun

    def _init_episode_log(self, num_atoms: int) -> FEPEpisodeLog:
        """Override parent method to return an `FEPEpisodeLog`."""
        episode_log = super()._init_episode_log(num_atoms)
        return FEPEpisodeLog(
            **{
                f.name: getattr(episode_log, f.name)
                for f in dataclasses.fields(episode_log)
            },
            per_lambda_energies=jnp.zeros((
                self._steps_per_episode,
                len(self._reference_lambdas),
            )),
        )

    @staticmethod
    def _simulation_step_fun(
        step_idx: int,
        internal_state: JaxMDSimulationState,
        apply_fun: Callable,
        temperature_schedule: Callable,
        is_md_simulation: bool,
        is_npt_simulation: bool,
        initial_box,
        compute_alchemical_values_fun: Callable,
        snapshot_interval: int,
        steps_per_episode: int,
        use_fractional_coords: bool,
    ) -> JaxMDSimulationState:
        """Base simulation step plus alchemical value logging at snapshot intervals."""
        internal_state = JaxMDSimulationEngine._simulation_step_fun(
            step_idx,
            internal_state,
            apply_fun,
            temperature_schedule,
            is_md_simulation,
            is_npt_simulation,
            initial_box,
            use_fractional_coords,
        )

        def _update_alchemical_values(
            state: JaxMDSimulationState,
        ) -> JaxMDSimulationState:
            box = state.jax_md_state.box if is_npt_simulation else initial_box
            log = state.episode_log
            positions = state.jax_md_state.position
            per_lambda_energies = compute_alchemical_values_fun(
                positions, state.system_state, box
            )
            new_log = jax.tree.map(
                lambda _log, ple: _log.set(
                    per_lambda_energies=_log.per_lambda_energies.at[step_idx].set(ple),
                ),
                log,
                per_lambda_energies,
                is_leaf=is_episode_log,
            )
            return state.set(episode_log=new_log)

        is_snapshot = (step_idx % snapshot_interval == 0) | (
            step_idx == steps_per_episode - 1
        )
        return jax.lax.cond(
            is_snapshot, _update_alchemical_values, lambda s: s, internal_state
        )

    def _update_state(self, episode_idx: int, episode_duration: float) -> None:
        super()._update_state(episode_idx, episode_duration)

        self.state.per_lambda_energies = self._concat(
            self.state.per_lambda_energies,
            self._extract_from_log("per_lambda_energies"),
        )
        self.state.final_positions = self.state.positions[-1]
        if self.is_md_simulation:
            self.state.final_velocities = self.state.velocities[-1]
        if self.is_npt_simulation:
            self.state.final_cell = self.state.cell[-1]

    @staticmethod
    def _get_update_graph_in_sim_step_fun() -> UpdateGraphInSimStepFun:
        """Returns the standard update function for a graph inside a simulation step."""

        def _update_graph_in_simulation_step(
            system_state: FEPSystemState | list[FEPSystemState],
            positions: np.ndarray | list[np.ndarray],
            graph: AlchemicalGraph,
            is_batched: bool,
            box: jax.Array | list[jax.Array] | None,
        ) -> Graph:
            graph = update_graph_in_simulation_step(
                system_state, positions, graph, is_batched, box
            )
            return graph.replace_globals(
                alchemical_lambda=jnp.asarray(system_state.lambda_values).reshape(-1, 2)
            )

        return _update_graph_in_simulation_step

    @staticmethod
    def run_compiled_episode(
        steps: int, step_fun: Callable, state: JaxMDSimulationState
    ) -> JaxMDSimulationState:
        """Episode loop. Callers are expected to `jax.jit` this explicitly."""
        return jax.lax.fori_loop(0, steps, step_fun, state)

    def get_internal_state(self) -> JaxMDSimulationState:
        """Return current internal state."""
        return self._internal_state

    def set_internal_state(self, state: JaxMDSimulationState) -> None:
        """Set internal state."""
        self._internal_state = state

    def build_per_device_step_fun(self) -> dict:
        """Build a compiled step function for this engine's stage, per local device.

        Returns:
            Dict like {device: step_fun}.
        """
        original_force_field = self._force_field
        original_reference_lambdas = self._reference_lambdas
        base_graph = self._build_base_graph()

        device_step_funs = {}
        for device in jax.local_devices():
            self._force_field = dataclass_replace(
                original_force_field,
                params=jax.device_put(original_force_field.params, device),
            )
            self._reference_lambdas = jax.device_put(original_reference_lambdas, device)
            _, pure_step_fun = self._setup_sim_functions(base_graph)
            device_step_funs[device] = pure_step_fun

        self._force_field = original_force_field
        self._reference_lambdas = original_reference_lambdas
        return device_step_funs

    def _build_base_graph(self) -> AlchemicalGraph:
        """Build a base graph from the current internal state's neighbor lists."""
        sys_state = self._internal_state.system_state
        neighbors = jax.tree.map(
            lambda s: s.neighbors, sys_state, is_leaf=is_system_state
        )
        lr = None
        if self.has_long_range_edges:
            lr = jax.tree.map(
                lambda s: s.long_range_neighbors, sys_state, is_leaf=is_system_state
            )
        return self._init_base_graph(self._atoms, neighbors, lr)
