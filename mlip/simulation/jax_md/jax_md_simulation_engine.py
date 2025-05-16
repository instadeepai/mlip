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
import time
from typing import Callable, TypeAlias

import ase
import jax
import jax.numpy as jnp
import jax_md
import jraph
import numpy as np
from jax_md import quantity
from jax_md.dataclasses import dataclass as jax_compatible_dataclass

from mlip.simulation.configs.jax_md_config import JaxMDSimulationConfig
from mlip.simulation.enums import SimulationType
from mlip.simulation.exceptions import SimulationIsNotInitializedError
from mlip.simulation.jax_md.helpers import (
    KCAL_PER_MOL_PER_ELECTRON_VOLT,
    TEMPERATURE_CONVERSION_FACTOR,
    VELOCITY_CONVERSION_FACTOR,
    batch_graph_with_one_dummy,
    get_masses,
    init_neighbor_lists,
    init_simulation_algorithm,
)
from mlip.simulation.jax_md.states import EpisodeLog, JaxMDSimulationState, SystemState
from mlip.simulation.simulation_engine import ForceField, SimulationEngine
from mlip.simulation.temperature_scheduling import get_temperature_schedule
from mlip.simulation.utils import create_graph_from_atoms

SIMULATION_RANDOM_SEED = 42

ModelEnergyFun: TypeAlias = Callable[[np.ndarray, SystemState], np.ndarray]
ModelForcesFun: TypeAlias = Callable[[np.ndarray, SystemState], np.ndarray]

logger = logging.getLogger("mlip")


class JaxMDSimulationEngine(SimulationEngine):
    """Simulation engine handling simulations with the JAX-MD backend.

    For MD, the NVT-Langevin algorithm is used
    (see `here <https://jax-md.readthedocs.io/en/main/
    jax_md.simulate.html#jax_md.simulate.nvt_langevin>`_).
    For energy minimization, the FIRE algorithm is used
    (see `here <https://jax-md.readthedocs.io/en/main/
    jax_md.minimize.html#jax_md.minimize.fire_descent>`_).
    """

    Config = JaxMDSimulationConfig

    def __init__(
        self,
        atoms: ase.Atoms,
        force_field: ForceField,
        config: JaxMDSimulationConfig,
    ) -> None:
        super().__init__(atoms, force_field, config)

    def _initialize(
        self,
        atoms: ase.Atoms,
        force_field: ForceField,
        config: JaxMDSimulationConfig,
    ) -> None:
        logger.debug("Initialization of simulation begins...")
        self._config = config
        positions = atoms.get_positions()
        self._num_atoms = positions.shape[0]
        self.state.atomic_numbers = atoms.numbers

        self._init_box()

        neighbors, self._neighbor_fun = init_neighbor_lists(
            self._displacement_fun,
            positions,
            force_field.cutoff_distance,
            self._config.edge_capacity_multiplier,
        )
        senders, receivers = neighbors.idx[1, :], neighbors.idx[0, :]
        graph = create_graph_from_atoms(
            atoms,
            senders,
            receivers,
            self._displacement_fun,
            force_field.allowed_atomic_numbers,
        )
        system_state = SystemState(neighbors=neighbors)

        model_calculate_fun = self._get_model_calculate_fun(graph, force_field)
        sim_init_fun, sim_apply_fun = init_simulation_algorithm(
            model_calculate_fun, self._shift_fun, self._config
        )
        self._pure_simulation_step_fun = functools.partial(
            self._simulation_step_fun,
            apply_fun=sim_apply_fun,
            temperature_schedule=get_temperature_schedule(
                self._config.temperature_schedule_config, self._config.num_steps
            ),
            is_md_simulation=self._config.simulation_type == SimulationType.MD,
        )
        jax_md_state = self._get_initial_jax_md_state(atoms, system_state, sim_init_fun)

        old_velocities = atoms.get_velocities()
        if old_velocities is not None and not np.all(old_velocities == 0.0):
            jax_md_state = self._set_state_velocities_to_restore_run(
                jax_md_state, old_velocities
            )

        self._steps_per_episode = self._config.num_steps // self._config.num_episodes
        self._internal_state = JaxMDSimulationState(
            jax_md_state=jax_md_state,
            system_state=system_state,
            episode_log=self._init_episode_log(),
            steps_completed=0,
        )
        logger.debug("Initialization of simulation completed.")

    def run(self) -> None:
        """See documentation of abstract parent class.

        For the JAX-MD backend, the simulation run is divided into episodes to ensure
        usage of jitting of MD/minimization steps for optimal performance.

        Important: The state of the simulation is updated and the loggers are called
        during this function.
        """
        logger.info("Starting simulation...")
        self._validate_initialization()
        is_md_simulation = self._config.simulation_type == SimulationType.MD
        episode_idx = 0

        while episode_idx < self._config.num_episodes:
            start_time = time.perf_counter()
            new_internal_state = jax.lax.fori_loop(
                0,
                self._steps_per_episode,
                self._pure_simulation_step_fun,
                self._internal_state,
            )
            if new_internal_state.system_state.neighbors.did_buffer_overflow:
                logger.info(
                    "Episode %s took %.2f seconds but has to be rerun due to neighbor"
                    " list overflow. Reallocating neighbors now...",
                    episode_idx + 1,
                    time.perf_counter() - start_time,
                )
                realloc_start_time = time.perf_counter()
                self._reallocate_neighbors()
                logger.info(
                    "Reallocating neighbours took %.3f seconds. Rerunning episode now.",
                    time.perf_counter() - realloc_start_time,
                )
                continue

            self._internal_state = new_internal_state
            end_time = time.perf_counter()
            episode_duration = end_time - start_time
            logger.info(
                "Episode %s completed in %.2f seconds.",
                episode_idx + 1,
                episode_duration,
            )
            self._update_state(episode_idx, episode_duration, is_md_simulation)
            for _logger in self.loggers:
                _logger(self.state)

            episode_idx += 1

        logger.info("Simulation completed.")

    def _validate_initialization(self):
        if self._pure_simulation_step_fun is None:
            raise SimulationIsNotInitializedError(
                "Simulation must be initialized before calling the run() function."
            )

    def _reallocate_neighbors(self) -> None:
        logger.debug("Neighbor lists require reallocation...")
        positions = self._internal_state.jax_md_state.position
        new_neighbors = self._neighbor_fun.allocate(positions)
        self._internal_state = self._internal_state.set(
            system_state=self._internal_state.system_state.set(neighbors=new_neighbors)
        )
        logger.debug("Reallocation of neighbor lists completed.")

    def _init_box(self) -> None:
        if self._config.box is None:
            self._displacement_fun, self._shift_fun = jax_md.space.free()
        else:
            box = (
                np.array(self._config.box)
                if isinstance(self._config.box, list)
                else self._config.box
            )
            self._displacement_fun, self._shift_fun = jax_md.space.periodic(
                box, wrapped=False
            )

    @staticmethod
    def _get_model_calculate_fun(
        graph: jraph.GraphsTuple, force_field_model: ForceField
    ) -> ModelEnergyFun | ModelForcesFun:
        def calc_func(
            positions: np.ndarray,
            system_state: SystemState,
            base_graph: jraph.GraphsTuple,
            force_field: ForceField,
        ) -> np.ndarray:
            batched_graph = batch_graph_with_one_dummy(
                system_state, positions, base_graph
            )
            force_field_output = force_field(batched_graph)
            output_forces = jnp.delete(force_field_output.forces, -1, axis=0)
            return output_forces * KCAL_PER_MOL_PER_ELECTRON_VOLT

        return functools.partial(
            calc_func,
            base_graph=graph,
            force_field=force_field_model,
        )

    def _get_initial_jax_md_state(
        self, atoms: ase.Atoms, system_state: SystemState, sim_init_fun: Callable
    ) -> jax_compatible_dataclass:
        args = (
            [jax.random.PRNGKey(SIMULATION_RANDOM_SEED)]
            if self._config.simulation_type == SimulationType.MD
            else []
        )
        args += [atoms.get_positions(), get_masses(atoms)]
        return sim_init_fun(*args, system_state=system_state)

    def _init_episode_log(self) -> EpisodeLog:
        is_md_simulation = self._config.simulation_type == SimulationType.MD
        one_dimensional = jnp.zeros((self._steps_per_episode,))
        three_dimensional = jnp.zeros((self._steps_per_episode, self._num_atoms, 3))

        return EpisodeLog(
            temperature=one_dimensional if is_md_simulation else jnp.empty(0),
            kinetic_energy=one_dimensional if is_md_simulation else jnp.empty(0),
            positions=three_dimensional,
            forces=three_dimensional,
            velocities=three_dimensional if is_md_simulation else jnp.empty(0),
        )

    @staticmethod
    def _simulation_step_fun(
        step_idx: int,
        internal_state: JaxMDSimulationState,
        apply_fun: Callable,
        temperature_schedule: Callable[[int], float],
        is_md_simulation: bool,
    ) -> JaxMDSimulationState:
        log = internal_state.episode_log
        jax_md_state = internal_state.jax_md_state

        current_force = jax_md_state.force / KCAL_PER_MOL_PER_ELECTRON_VOLT
        new_log = log.set(
            positions=log.positions.at[step_idx].set(jax_md_state.position),
            forces=log.forces.at[step_idx].set(current_force),
        )

        if is_md_simulation:
            current_temperature = quantity.temperature(
                momentum=jax_md_state.momentum, mass=jax_md_state.mass
            )
            current_temperature_kelvin = (
                current_temperature / TEMPERATURE_CONVERSION_FACTOR
            )

            current_kinetic_energy = quantity.kinetic_energy(
                momentum=jax_md_state.momentum, mass=jax_md_state.mass
            )
            current_kinetic_energy_ev = (
                current_kinetic_energy / KCAL_PER_MOL_PER_ELECTRON_VOLT
            )

            current_velocities = jax_md_state.velocity / VELOCITY_CONVERSION_FACTOR

            new_log = new_log.set(
                temperature=log.temperature.at[step_idx].set(
                    current_temperature_kelvin
                ),
                kinetic_energy=log.kinetic_energy.at[step_idx].set(
                    current_kinetic_energy_ev
                ),
                velocities=log.velocities.at[step_idx].set(current_velocities),
            )

        kwargs = {"system_state": internal_state.system_state}
        if is_md_simulation:
            kwargs["kT"] = (
                temperature_schedule(internal_state.steps_completed)
                * TEMPERATURE_CONVERSION_FACTOR
            )

        new_jax_md_state = apply_fun(jax_md_state, **kwargs)

        # The following code updates the neighbors, which is duplicate but has to
        # be also run here as jax-md does not currently allow to pass information
        # back to the outside from the force function. This can be optimized in
        # the future.
        old_neighbors = internal_state.system_state.neighbors
        new_neighbors = old_neighbors.update(new_jax_md_state.position)
        new_system_state = internal_state.system_state.set(neighbors=new_neighbors)

        steps_completed = internal_state.steps_completed + 1
        return internal_state.set(
            jax_md_state=new_jax_md_state,
            episode_log=new_log,
            system_state=new_system_state,
            steps_completed=steps_completed,
        )

    def _update_state(
        self, episode_idx: int, episode_duration: float, is_md_simulation: bool
    ) -> None:
        episode_log = self._internal_state.episode_log
        snapshot_interval = self._config.snapshot_interval

        def _concat(current: np.ndarray, new: np.ndarray) -> np.ndarray:
            """Append the new information from the latest episode
             to the current state. Information from every ``log_interval``
             snapshots is added to the state array.

            Args:
                current: The array representing the current state of one of
                        the state's attributes.
                new: The array representing one of the state's attributes in
                    the last episode.
            Returns:
                The updated array.
            """
            if current is None:
                return new[::snapshot_interval]
            return np.concatenate([current, new[::snapshot_interval]], axis=0)

        self.state.positions = _concat(self.state.positions, episode_log.positions)
        self.state.forces = _concat(self.state.forces, episode_log.forces)

        self.state.step = (episode_idx + 1) * self._steps_per_episode
        self.state.compute_time_seconds += episode_duration

        if is_md_simulation:
            self.state.temperature = _concat(
                self.state.temperature, episode_log.temperature
            )
            self.state.kinetic_energy = _concat(
                self.state.kinetic_energy, episode_log.kinetic_energy
            )
            self.state.velocities = _concat(
                self.state.velocities, episode_log.velocities
            )

    @staticmethod
    def _set_state_velocities_to_restore_run(
        jax_md_state: jax_compatible_dataclass, old_velocities: np.ndarray
    ) -> jax_compatible_dataclass:
        return jax_md_state.set(
            momentum=old_velocities * VELOCITY_CONVERSION_FACTOR * jax_md_state.mass
        )
