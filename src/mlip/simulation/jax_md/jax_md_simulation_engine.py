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
from jax.tree_util import tree_map
from jax_md import quantity
from jax_md.dataclasses import dataclass as jax_compatible_dataclass

from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.simulation.configs.jax_md_config import JaxMDSimulationConfig
from mlip.simulation.enums import SimulationType
from mlip.simulation.exceptions import SimulationIsNotInitializedError
from mlip.simulation.jax_md.helpers import (
    KCAL_PER_MOL_PER_ELECTRON_VOLT,
    TEMPERATURE_CONVERSION_FACTOR,
    VELOCITY_CONVERSION_FACTOR,
    get_masses,
    init_neighbor_lists,
    init_simulation_algorithm,
    is_episode_log,
    is_neighbor_fun,
    is_neighbor_list,
    is_system_state,
    update_graph_in_simulation_step,
)
from mlip.simulation.jax_md.states import EpisodeLog, JaxMDSimulationState, SystemState
from mlip.simulation.simulation_engine import ForceField, SimulationEngine
from mlip.simulation.temperature_scheduling import get_temperature_schedule
from mlip.simulation.utils import create_graph_from_atoms
from mlip.typing.graph_definition import GraphEdges

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

    Batched MD simulations are supported. Just pass a list of `ase.Atoms` objects
    to the constructor. See deep-dive tutorials on simulations for more information.
    """

    Config = JaxMDSimulationConfig

    def __init__(
        self,
        atoms: ase.Atoms | list[ase.Atoms],
        force_field: ForceField,
        config: JaxMDSimulationConfig,
    ) -> None:
        super().__init__(atoms, force_field, config)

    def _initialize(
        self,
        atoms: ase.Atoms | list[ase.Atoms],
        force_field: ForceField,
        config: JaxMDSimulationConfig,
    ) -> None:
        """Implementation of the initialization that is called in the parent class
        constructor. Contains JAX-MD specific initialization steps.

        Args:
            atoms: The atoms of the system to simulate. Can be a list of systems.
            force_field: The force field to use in the simulation.
            config: The configuration/settings of the simulation.
        """

        logger.debug("Initialization of simulation begins...")
        self._config = config
        self._atoms = atoms
        self._force_field = force_field

        positions = tree_map(lambda a: a.get_positions(), atoms)
        self._num_atoms = tree_map(lambda p: p.shape[0], positions)
        if isinstance(self._num_atoms, list) and 0 in self._num_atoms:
            raise ValueError("Empty 'ase.Atoms' detected in batch.")
        if isinstance(self._num_atoms, list) and 1 in self._num_atoms:
            raise ValueError("Single atom system detected in batch, not supported yet.")
        self.state.atomic_numbers = tree_map(lambda a: a.numbers, atoms)

        self._init_box()

        neighbors, self._neighbor_fun = init_neighbor_lists(
            self._displacement_fun,
            positions,
            force_field.cutoff_distance,
            self._config.edge_capacity_multiplier,
        )

        senders = tree_map(lambda n: n.idx[1, :], neighbors, is_leaf=is_neighbor_list)
        receivers = tree_map(lambda n: n.idx[0, :], neighbors, is_leaf=is_neighbor_list)
        graph = self._init_base_graph(
            atoms, senders, receivers, force_field.allowed_atomic_numbers
        )

        system_state = self._system_state_from_neighbors(neighbors)

        model_calculate_fun = self._get_model_calculate_fun(
            graph, force_field, is_batched_sim=isinstance(atoms, list)
        )
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

        old_velocities = tree_map(lambda a: a.get_velocities(), atoms)
        old_velocities_exist = tree_map(
            lambda v: v is not None and not np.all(v == 0.0), old_velocities
        )
        # In batched simulations, only use old velocities if all structures have them:
        if np.all(old_velocities_exist):
            jax_md_state = self._set_state_velocities_to_restore_run(
                jax_md_state, old_velocities
            )

        self._steps_per_episode = self._config.num_steps // self._config.num_episodes
        self._internal_state = JaxMDSimulationState(
            jax_md_state=jax_md_state,
            system_state=system_state,
            episode_log=tree_map(lambda a: self._init_episode_log(len(a)), atoms),
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
            if self._did_neighbor_buffer_overflow(new_internal_state):
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
        new_neighbors = tree_map(
            lambda n_fun, p: n_fun.allocate(p),
            self._neighbor_fun,
            positions,
            is_leaf=is_neighbor_fun,
        )
        self._internal_state = self._internal_state.set(
            system_state=tree_map(
                lambda s, n: s.set(neighbors=n),
                self._internal_state.system_state,
                new_neighbors,
                is_leaf=lambda x: is_system_state(x) or is_neighbor_list(x),
            )
        )
        self._update_base_graph_in_pure_sim_step_fun(new_neighbors)
        logger.debug("Reallocation of neighbor lists completed.")

    def _init_box(self) -> None:
        # TODO: test jax_md.periodic_general() for arbitrary lattices. For now, we
        #       check that the ase.Atoms do not have PBCs or cell, since Jax-MD only
        #       supports orthorhombic boxes that are passed from config.
        has_pbc = (
            any(atoms.pbc.any() for atoms in self._atoms)
            if isinstance(self._atoms, list)
            else self._atoms.pbc.any()
        )
        if self._config.box is None and not has_pbc:
            self._displacement_fun, self._shift_fun = jax_md.space.free()
        elif self._config.box is not None:
            box = (
                np.array(self._config.box)
                if isinstance(self._config.box, list)
                else self._config.box
            )
            self._displacement_fun, self._shift_fun = jax_md.space.periodic(
                box, wrapped=False
            )
        else:
            raise NotImplementedError(
                "Jax-MD can only be used with cubic boxes passed from config for now. "
                "To avoid this error, you can set atoms.pbc to False."
            )

    @staticmethod
    def _get_model_calculate_fun(
        graph: jraph.GraphsTuple, force_field_model: ForceField, is_batched_sim: bool
    ) -> ModelEnergyFun | ModelForcesFun:
        """This function returns the core force calculate function compatible with
        JAX-MD and also compatible with batched simulations if requested."""

        def calc_func(
            positions: np.ndarray,
            system_state: SystemState,
            base_graph: jraph.GraphsTuple,
            force_field: ForceField,
            is_batched: bool,
            split_idx: list[int] | None,
        ) -> np.ndarray | list[np.ndarray]:
            updated_graph = update_graph_in_simulation_step(
                system_state, positions, base_graph, is_batched
            )

            force_field_output = force_field(updated_graph)
            output_forces = jnp.delete(force_field_output.forces, -1, axis=0)
            output_forces = output_forces * KCAL_PER_MOL_PER_ELECTRON_VOLT

            # For batched simulations, split into list of forces
            if is_batched:
                return jnp.split(output_forces, split_idx, axis=0)

            return output_forces

        forces_split_idx = None
        if is_batched_sim:
            sizes = np.delete(graph.n_node, -1)
            forces_split_idx = [int(sum(sizes[:i])) for i in range(1, len(sizes))]

        return functools.partial(
            calc_func,
            base_graph=graph,
            force_field=force_field_model,
            is_batched=is_batched_sim,
            split_idx=forces_split_idx,
        )

    def _get_initial_jax_md_state(
        self,
        atoms: ase.Atoms | list[ase.Atoms],
        system_state: SystemState | list[SystemState],
        sim_init_fun: Callable,
    ) -> jax_compatible_dataclass:
        """Initializing JAX-MD state either batched or non-batched."""
        args = []
        if self._config.simulation_type == SimulationType.MD:
            args = [jax.random.PRNGKey(SIMULATION_RANDOM_SEED)]

        positions = tree_map(lambda a: a.get_positions(), atoms)
        masses = tree_map(lambda a: get_masses(a), atoms)
        args += [positions, masses]
        return sim_init_fun(*args, system_state=system_state)

    def _init_episode_log(self, num_atoms: int) -> EpisodeLog:
        is_md_simulation = self._config.simulation_type == SimulationType.MD
        one_dimensional = jnp.zeros((self._steps_per_episode,))
        three_dimensional = jnp.zeros((self._steps_per_episode, num_atoms, 3))

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
        """This function is the implementation of the core simulation step.

        Needs to be wrapped around `functools.partial` with some arguments fixed so
        that it can be jitted later on."""
        log = internal_state.episode_log
        jax_md_state = internal_state.jax_md_state

        current_force = tree_map(
            lambda f: f / KCAL_PER_MOL_PER_ELECTRON_VOLT, jax_md_state.force
        )
        new_log = tree_map(
            lambda _log, p, f: _log.set(
                positions=_log.positions.at[step_idx].set(p),
                forces=_log.forces.at[step_idx].set(f),
            ),
            log,
            jax_md_state.position,
            current_force,
            is_leaf=is_episode_log,
        )

        if is_md_simulation:
            current_temperature = tree_map(
                lambda mom, mass: quantity.temperature(momentum=mom, mass=mass),
                jax_md_state.momentum,
                jax_md_state.mass,
            )
            current_temperature_kelvin = tree_map(
                lambda t: t / TEMPERATURE_CONVERSION_FACTOR, current_temperature
            )

            current_kinetic_energy = tree_map(
                lambda mom, mass: quantity.kinetic_energy(momentum=mom, mass=mass),
                jax_md_state.momentum,
                jax_md_state.mass,
            )
            current_kinetic_energy_ev = tree_map(
                lambda kin: kin / KCAL_PER_MOL_PER_ELECTRON_VOLT, current_kinetic_energy
            )

            current_velocities = tree_map(
                lambda v: v / VELOCITY_CONVERSION_FACTOR, jax_md_state.velocity
            )

            new_log = tree_map(
                lambda _log, t, kin, v: _log.set(
                    temperature=_log.temperature.at[step_idx].set(t),
                    kinetic_energy=_log.kinetic_energy.at[step_idx].set(kin),
                    velocities=_log.velocities.at[step_idx].set(v),
                ),
                new_log,
                current_temperature_kelvin,
                current_kinetic_energy_ev,
                current_velocities,
                is_leaf=is_episode_log,
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
        old_neighbors = tree_map(
            lambda s: s.neighbors, internal_state.system_state, is_leaf=is_system_state
        )
        new_neighbors = tree_map(
            lambda n, p: n.update(p),
            old_neighbors,
            new_jax_md_state.position,
            is_leaf=is_neighbor_list,
        )
        new_system_state = tree_map(
            lambda s, n: s.set(neighbors=n),
            internal_state.system_state,
            new_neighbors,
            is_leaf=lambda x: is_system_state(x) or is_neighbor_list(x),
        )

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
        """Updates the simulation state that is publicly accessed by users of
        this class. This means taking the logged data from the episode log and
        concatenating it to the existing arrays in the state."""
        episode_log = self._internal_state.episode_log
        snapshot_interval = self._config.snapshot_interval

        def _concat(
            current: np.ndarray | list[np.ndarray] | None,
            new: np.ndarray | list[np.ndarray],
        ) -> np.ndarray | list[np.ndarray]:
            """Append the new information from the latest episode
            to the current state. Information from every ``log_interval``
            snapshots is added to the state array.

            Uses `tree_map` to work for batched simulations, too.

            Args:
                current: The array representing the current state of one of
                         the state's attributes.
                new: The array representing one of the state's attributes in
                     the last episode.
            Returns:
                The updated array.
            """
            if current is None:
                return tree_map(lambda n: n[::snapshot_interval], new)
            return tree_map(
                lambda c, n: np.concatenate([c, n[::snapshot_interval]], axis=0),
                current,
                new,
            )

        def _extract_from_log(attr_name: str) -> np.ndarray | list[np.ndarray]:
            """Small helper using `tree_map` to extract a property
            from the episode log.
            """
            return tree_map(
                lambda _log: getattr(_log, attr_name),
                episode_log,
                is_leaf=is_episode_log,
            )

        self.state.positions = _concat(
            self.state.positions, _extract_from_log("positions")
        )
        self.state.forces = _concat(self.state.forces, _extract_from_log("forces"))

        self.state.step = (episode_idx + 1) * self._steps_per_episode
        self.state.compute_time_seconds += episode_duration

        if is_md_simulation:
            self.state.temperature = _concat(
                self.state.temperature, _extract_from_log("temperature")
            )
            self.state.kinetic_energy = _concat(
                self.state.kinetic_energy, _extract_from_log("kinetic_energy")
            )
            self.state.velocities = _concat(
                self.state.velocities, _extract_from_log("velocities")
            )

    @staticmethod
    def _system_state_from_neighbors(
        neighbors: jax_md.partition.NeighborList,
    ) -> SystemState:
        return tree_map(
            lambda n: SystemState(neighbors=n),
            neighbors,
            is_leaf=is_neighbor_list,
        )

    @staticmethod
    def _set_state_velocities_to_restore_run(
        jax_md_state: jax_compatible_dataclass, old_velocities: np.ndarray
    ) -> jax_compatible_dataclass:
        return jax_md_state.set(
            momentum=old_velocities * VELOCITY_CONVERSION_FACTOR * jax_md_state.mass
        )

    @staticmethod
    def _did_neighbor_buffer_overflow(internal_state: JaxMDSimulationState) -> bool:
        """Checks and returns whether buffer of neighbor lists overflowed.

        Written so that it works with batched simulations, too.
        """
        did_buffer_overflow = tree_map(
            lambda s: s.neighbors.did_buffer_overflow,
            internal_state.system_state,
            is_leaf=is_system_state,
        )
        # In batched simulations, we rerun an episode if any of the
        # systems overflowed its buffer
        return np.any(did_buffer_overflow)

    def _init_base_graph(
        self,
        atoms: ase.Atoms | list[ase.Atoms],
        senders: np.ndarray | list[np.ndarray],
        receivers: np.ndarray | list[np.ndarray],
        allowed_atomic_numbers: set[int],
    ) -> jraph.GraphsTuple:
        """Initiates the base graph (batched or unbatched) for the simulation.

        This graph will be given to the jitted calculate function, which will then
        replace relevant parts of it in each simulation step.

        Args:
            atoms: The atoms or list of atoms.
            senders: The sender indices of the edges.
            receivers: The receiver indices of the edges.
            allowed_atomic_numbers: The allowed atomic numbers given by the model used.

        Returns:
            The base graph (either batched or unbatched).
        """
        graph = tree_map(
            lambda a, s, r: create_graph_from_atoms(
                a,
                s,
                r,
                self._displacement_fun,
                allowed_atomic_numbers,
            ),
            atoms,
            senders,
            receivers,
        )

        # Batched simulations
        if isinstance(atoms, list):
            assert isinstance(graph, list) and len(graph) == len(atoms)
            # Important note:
            # The edges cannot be batched because of the displ. func. in there.
            # This is because the displ. func. is a Callable object that the jraph
            # batching algorithm cannot handle. Hence, we first replace it with a
            # dummy and then later set it again.
            saved_edges = graph[0].edges
            dummy_edges = GraphEdges(shifts=None, displ_fun=None)
            graph = [g._replace(edges=dummy_edges) for g in graph]
            batched_graph = next(
                dynamically_batch(
                    graph,
                    n_node=sum(g.n_node.item(0) for g in graph) + 1,
                    n_edge=sum(g.n_edge.item(0) for g in graph) + 1,
                    n_graph=len(graph) + 1,
                )
            )
            return batched_graph._replace(edges=saved_edges)

        return graph

    def _update_base_graph_in_pure_sim_step_fun(
        self, neighbors: jax_md.partition.NeighborList
    ) -> None:
        """After reallocation of neighbors, the simulation step function needs to
        be updated because the `graph.n_edge` attribute has changed."""
        senders = tree_map(lambda n: n.idx[1, :], neighbors, is_leaf=is_neighbor_list)
        receivers = tree_map(lambda n: n.idx[0, :], neighbors, is_leaf=is_neighbor_list)
        new_base_graph = self._init_base_graph(
            self._atoms, senders, receivers, self._force_field.allowed_atomic_numbers
        )
        model_calculate_fun = self._get_model_calculate_fun(
            new_base_graph,
            self._force_field,
            is_batched_sim=isinstance(self._atoms, list),
        )
        _, sim_apply_fun = init_simulation_algorithm(
            model_calculate_fun, self._shift_fun, self._config
        )
        self._pure_simulation_step_fun.keywords["apply_fun"] = sim_apply_fun
