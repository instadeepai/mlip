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
import numpy as np
from jax_md import quantity
from jax_md.dataclasses import dataclass as jax_compatible_dataclass

from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.graph import Graph, GraphEdges
from mlip.simulation.configs.jax_md_config import JaxMDSimulationConfig
from mlip.simulation.enums import MDIntegrator, SimulationType
from mlip.simulation.exceptions import SimulationIsNotInitializedError
from mlip.simulation.jax_md.graph_creation import create_graph_from_atoms_and_edges
from mlip.simulation.jax_md.helpers import (
    KCAL_PER_MOL_PER_ELECTRON_VOLT,
    TEMPERATURE_CONVERSION_FACTOR,
    VELOCITY_CONVERSION_FACTOR,
    box_to_cell,
    get_masses,
    get_neighbor_list_receivers,
    get_neighbor_list_senders,
    init_displacement_fun,
    init_neighbor_lists,
    init_simulation_algorithm,
    is_episode_log,
    is_neighbor_fun,
    is_neighbor_list,
    is_system_state,
    make_batched_displ_fun,
    update_graph_in_simulation_step,
)
from mlip.simulation.jax_md.states import EpisodeLog, JaxMDSimulationState, SystemState
from mlip.simulation.simulation_engine import ForceField, SimulationEngine
from mlip.simulation.temperature_scheduling import get_temperature_schedule
from mlip.simulation.utils import (
    has_simulation_exploded,
    resolve_atoms_charge_for_model,
)

SIMULATION_RANDOM_SEED = 42

ModelEnergyFun: TypeAlias = Callable[[np.ndarray, SystemState], np.ndarray]
ModelForcesFun: TypeAlias = Callable[[np.ndarray, SystemState], np.ndarray]
UpdateGraphInSimStepFun: TypeAlias = Callable[
    [SystemState | list[SystemState], np.ndarray | list[np.ndarray], Graph, bool], Graph
]

logger = logging.getLogger("mlip")


class JaxMDSimulationEngine(SimulationEngine):
    """Simulation engine handling simulations with the JAX-MD backend.

    For MD, the NVT-Langevin algorithm is used
    (see `here <https://jax-md.readthedocs.io/en/main/
    jax_md.simulate.html#jax_md.simulate.nvt_langevin>`__).
    For energy minimization, the FIRE algorithm is used
    (see `here <https://jax-md.readthedocs.io/en/main/
    jax_md.minimize.html#jax_md.minimize.fire_descent>`__).

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

        self._atoms = resolve_atoms_charge_for_model(
            self._atoms, force_field, config.set_none_charge_to_zero
        )

        self.is_batched_sim = isinstance(atoms, list)
        self.is_md_simulation = self._config.simulation_type == SimulationType.MD
        self.is_npt_simulation = (
            self.is_md_simulation and self._config.md_integrator.ensemble == "npt"
        )

        if self.is_npt_simulation:
            if not isinstance(self._config.molecule_indices, list):
                raise ValueError(
                    "`molecule_indices` must be provided for NPT simulations."
                )
            if self.is_batched_sim and not isinstance(
                self._config.molecule_indices[0], list
            ):
                raise ValueError(
                    "`molecule_indices` must be a list of lists "
                    "for batched NPT simulations"
                )

        positions = jax.tree.map(lambda a: a.get_positions(), atoms)
        self._num_atoms = jax.tree.map(lambda p: p.shape[0], positions)
        if self.is_batched_sim and 0 in self._num_atoms:
            raise ValueError("Empty 'ase.Atoms' detected in batch.")
        if self.is_batched_sim and 1 in self._num_atoms:
            raise ValueError("Single atom system detected in batch, not supported yet.")
        self.state.atomic_numbers = jax.tree.map(lambda a: a.numbers, atoms)

        self._init_box_and_displacement_fun()

        neighbors, self._neighbor_fun = init_neighbor_lists(
            self._displacement_fun,
            positions,
            force_field.cutoff_distance,
            self._config.edge_capacity_multiplier,
            box=self._initial_box,
        )

        senders = jax.tree.map(
            get_neighbor_list_senders, neighbors, is_leaf=is_neighbor_list
        )
        receivers = jax.tree.map(
            get_neighbor_list_receivers, neighbors, is_leaf=is_neighbor_list
        )

        long_range_cutoff_distance = force_field.long_range_cutoff_distance
        self._has_long_range = long_range_cutoff_distance is not None
        if self._has_long_range:
            long_range_neighbors, self._long_range_neighbor_fun = init_neighbor_lists(
                self._displacement_fun,
                positions,
                long_range_cutoff_distance,
                self._config.edge_capacity_multiplier,
                box=self._initial_box,
            )
            senders_long_range = jax.tree.map(
                get_neighbor_list_senders,
                long_range_neighbors,
                is_leaf=is_neighbor_list,
            )
            receivers_long_range = jax.tree.map(
                get_neighbor_list_receivers,
                long_range_neighbors,
                is_leaf=is_neighbor_list,
            )
        else:
            self._long_range_neighbor_fun = None
            long_range_neighbors = None
            senders_long_range = None
            receivers_long_range = None

        graph = self._init_base_graph(
            atoms, senders, receivers, senders_long_range, receivers_long_range
        )

        system_state = self._system_state_from_neighbors(
            neighbors, long_range_neighbors
        )

        sim_init_fun, _pure_simulation_step_fun = self._setup_sim_functions(graph)
        self._pure_simulation_step_fun = _pure_simulation_step_fun

        jax_md_state = self._get_initial_jax_md_state(atoms, system_state, sim_init_fun)

        old_velocities = jax.tree.map(lambda a: a.get_velocities(), atoms)
        old_velocities_exist = jax.tree.map(
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
            episode_log=jax.tree.map(lambda a: self._init_episode_log(len(a)), atoms),
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
            self._update_state(episode_idx, episode_duration)
            for _logger in self.loggers:
                _logger(self.state)

            if self._has_simulation_exploded(new_internal_state):
                logger.warning("Simulation exploded. Stopping simulation.")
                break

            episode_idx += 1

        logger.info("Simulation completed.")

    def _setup_sim_functions(self, base_graph: Graph) -> tuple[Callable, Callable]:
        """Setting up the two core simulation functions.

        These are the simulation init function of JAX-MD required to set up the
        first JAX-MD state, and the pure simulation function which is the main function
        in the simulation for loop.

        Args:
            base_graph: The base graph already created from the input `ase.Atoms`.

        Returns:
            The tuple of these two simulation functions.
        """
        make_model_calculate_fun = functools.partial(
            self._get_model_calculate_fun,
            graph=base_graph,
        )

        sim_init_fun, sim_apply_fun = init_simulation_algorithm(
            make_model_calculate_fun,
            self._force_field,
            self._shift_fun,
            self._config,
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
        )
        return sim_init_fun, pure_simulation_step_fun

    def _validate_initialization(self):
        if self._pure_simulation_step_fun is None:
            raise SimulationIsNotInitializedError(
                "Simulation must be initialized before calling the run() function."
            )

    def _reallocate_neighbors(self) -> None:
        logger.debug("Neighbor lists require reallocation...")
        if self.is_npt_simulation:
            box = self._internal_state.jax_md_state.box
        else:
            box = self._initial_box

        def _allocate(n_fun, p, b):
            return n_fun.allocate(p, box=b)

        positions = self._internal_state.jax_md_state.position
        if self.is_batched_sim and not isinstance(box, list):
            box = [box] * len(positions)
        new_neighbors = jax.tree.map(
            _allocate,
            self._neighbor_fun,
            positions,
            box,
            is_leaf=is_neighbor_fun,
        )
        new_long_range_neighbors = None
        if self._has_long_range:
            new_long_range_neighbors = jax.tree.map(
                _allocate,
                self._long_range_neighbor_fun,
                positions,
                box,
                is_leaf=is_neighbor_fun,
            )
        if new_long_range_neighbors is None:
            new_system_state = jax.tree.map(
                lambda s, n: s.set(neighbors=n),
                self._internal_state.system_state,
                new_neighbors,
                is_leaf=lambda x: is_system_state(x) or is_neighbor_list(x),
            )
        else:
            new_system_state = jax.tree.map(
                lambda s, n, lr: s.set(neighbors=n, long_range_neighbors=lr),
                self._internal_state.system_state,
                new_neighbors,
                new_long_range_neighbors,
                is_leaf=lambda x: is_system_state(x) or is_neighbor_list(x),
            )
        self._internal_state = self._internal_state.set(system_state=new_system_state)
        self._update_base_graph_in_pure_sim_step_fun(
            new_neighbors, new_long_range_neighbors
        )
        logger.debug("Reallocation of neighbor lists completed.")

    def _init_box_and_displacement_fun(self) -> None:
        atoms_list = self._atoms if isinstance(self._atoms, list) else [self._atoms]
        boxes = []
        warned = False
        for atoms_i in atoms_list:
            cell = atoms_i.get_cell()
            if np.any(cell):
                if not np.all(np.diag(np.diag(cell)) == cell):
                    raise NotImplementedError(
                        "Currently can only run JAX-MD simulations with orthorhombic "
                        "(diagonal) cells. Replace `atoms.cell` with a suitable array."
                    )
                if not warned:
                    logger.warning(
                        "Ignoring `box` parameter as `atoms` already have cell."
                    )
                    warned = True
                boxes.append(np.diag(cell))
            else:
                box = self._config.box
                if box is not None:
                    if isinstance(box, float):
                        box = [box] * 3
                    if isinstance(self._config.box, list):
                        assert len(self._config.box) == 3
                    box = np.array(box)
                boxes.append(box)

        # Check we either all systems have cells or all systems have no cell
        none_mask = [b is None for b in boxes]
        if any(none_mask) and not all(none_mask):
            raise ValueError(
                "For batched simulations, either all systems must have cells "
                "or all systems must have no cell."
            )

        ref_box = boxes[0]
        self._displacement_fun, self._shift_fun, self._cell_to_box_fun = (
            init_displacement_fun(ref_box)
        )
        self._initial_box = boxes if isinstance(self._atoms, list) else boxes[0]

    @staticmethod
    def _has_simulation_exploded(internal_state: JaxMDSimulationState) -> bool:
        """Whether the simulation has exploded."""
        has_exploded = jax.tree.map(
            lambda log: has_simulation_exploded(log.temperature),
            internal_state.episode_log,
            is_leaf=is_episode_log,
        )
        return np.any(has_exploded)

    def _get_model_calculate_fun(
        self, graph: Graph, force_field_model: ForceField, is_energy_fun: bool
    ) -> ModelEnergyFun | ModelForcesFun:
        """This function returns the core force calculate function compatible with
        JAX-MD and also compatible with batched simulations if requested.
        """

        def calc_func(
            positions: np.ndarray,
            system_state: SystemState,
            base_graph: Graph,
            force_field: ForceField,
            is_batched_sim: bool,
            update_graph_in_sim_step_fun: UpdateGraphInSimStepFun,
            forces_split_idx: list[int] | None,
            is_energy_fun: bool,
            box: np.ndarray | list[np.ndarray] | None = self._initial_box,
        ) -> np.ndarray | list[np.ndarray]:
            updated_graph = update_graph_in_sim_step_fun(
                system_state, positions, base_graph, is_batched_sim, box=box
            )

            force_field_output = force_field(updated_graph)

            if is_energy_fun:
                energies = (
                    force_field_output.energy[:-1] * KCAL_PER_MOL_PER_ELECTRON_VOLT
                )
                return list(energies) if is_batched_sim else energies[0]

            forces = (
                jnp.delete(force_field_output.forces, -1, axis=0)
                * KCAL_PER_MOL_PER_ELECTRON_VOLT
            )
            if is_batched_sim:
                return jnp.split(forces, forces_split_idx, axis=0)
            return forces

        forces_split_idx = None
        if self.is_batched_sim:
            sizes = np.delete(graph.n_node, -1)
            forces_split_idx = [int(sum(sizes[:i])) for i in range(1, len(sizes))]

        return functools.partial(
            calc_func,
            base_graph=graph,
            force_field=force_field_model,
            is_batched_sim=self.is_batched_sim,
            update_graph_in_sim_step_fun=self._get_update_graph_in_sim_step_fun(),
            forces_split_idx=forces_split_idx,
            is_energy_fun=is_energy_fun,
        )

    @staticmethod
    def _get_update_graph_in_sim_step_fun() -> UpdateGraphInSimStepFun:
        """Returns the standard update function for a graph inside a simulation step.

        This is its own method so it can be overridden easily in custom JAX-MD engines
        that would like to modify the `Graph` object and hence need to use a different
        update function for the graph.
        """
        return update_graph_in_simulation_step

    def _get_initial_jax_md_state(
        self,
        atoms: ase.Atoms | list[ase.Atoms],
        system_state: SystemState | list[SystemState],
        sim_init_fun: Callable,
    ) -> jax_compatible_dataclass:
        """Initializing JAX-MD state either batched or non-batched."""
        random_key = jax.random.PRNGKey(SIMULATION_RANDOM_SEED)
        positions = jax.tree.map(lambda a: a.get_positions(), atoms)
        masses = jax.tree.map(get_masses, atoms)

        if self._config.simulation_type == SimulationType.MINIMIZATION:
            args = [positions, masses]

        elif self._config.simulation_type == SimulationType.MD:
            if self._config.md_integrator == MDIntegrator.NVT_LANGEVIN:
                args = [random_key, positions, masses]
            elif self._config.md_integrator == MDIntegrator.NPT_MC_LANGEVIN:
                args = [random_key, positions, self._initial_box, masses]
            else:
                raise ValueError(
                    f"MD integrator {self._config.md_integrator} not supported."
                )

        else:
            raise ValueError(
                f"Simulation type {self._config.simulation_type} not supported."
            )

        return sim_init_fun(*args, system_state=system_state)

    def _init_episode_log(self, num_atoms: int) -> EpisodeLog:
        is_md_simulation = self.is_md_simulation
        one_dimensional = jnp.zeros((self._steps_per_episode,))
        atoms_by_three_dimensional = jnp.zeros((self._steps_per_episode, num_atoms, 3))
        three_by_three_dimensional = jnp.zeros((self._steps_per_episode, 3, 3))

        return EpisodeLog(
            positions=atoms_by_three_dimensional,
            forces=atoms_by_three_dimensional,
            temperature=one_dimensional if is_md_simulation else jnp.empty(0),
            kinetic_energy=one_dimensional if is_md_simulation else jnp.empty(0),
            velocities=atoms_by_three_dimensional if is_md_simulation else jnp.empty(0),
            cell=three_by_three_dimensional if self.is_npt_simulation else jnp.empty(0),
        )

    @staticmethod
    def _simulation_step_fun(
        step_idx: int,
        internal_state: JaxMDSimulationState,
        apply_fun: Callable,
        temperature_schedule: Callable[[int], float],
        is_md_simulation: bool,
        is_npt_simulation: bool,
        initial_box: np.ndarray | None,
    ) -> JaxMDSimulationState:
        """This function is the implementation of the core simulation step.

        Needs to be wrapped around `functools.partial` with some arguments fixed so
        that it can be jitted later on.
        """
        log = internal_state.episode_log
        jax_md_state = internal_state.jax_md_state

        current_force = jax.tree.map(
            lambda f: f / KCAL_PER_MOL_PER_ELECTRON_VOLT, jax_md_state.force
        )
        new_log = jax.tree.map(
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
            current_temperature = jax.tree.map(
                lambda mom, mass: quantity.temperature(momentum=mom, mass=mass),
                jax_md_state.momentum,
                jax_md_state.mass,
            )
            current_temperature_kelvin = jax.tree.map(
                lambda t: t / TEMPERATURE_CONVERSION_FACTOR, current_temperature
            )

            current_kinetic_energy = jax.tree.map(
                lambda mom, mass: quantity.kinetic_energy(momentum=mom, mass=mass),
                jax_md_state.momentum,
                jax_md_state.mass,
            )
            current_kinetic_energy_ev = jax.tree.map(
                lambda kin: kin / KCAL_PER_MOL_PER_ELECTRON_VOLT, current_kinetic_energy
            )

            current_velocities = jax.tree.map(
                lambda v: v / VELOCITY_CONVERSION_FACTOR, jax_md_state.velocity
            )

            new_log = jax.tree.map(
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

        if is_npt_simulation:
            current_cells = jax.tree.map(box_to_cell, jax_md_state.box)
            new_log = jax.tree.map(
                lambda _log, c: _log.set(
                    cell=_log.cell.at[step_idx].set(c),
                ),
                new_log,
                current_cells,
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
        old_neighbors = jax.tree.map(
            lambda s: s.neighbors, internal_state.system_state, is_leaf=is_system_state
        )

        _box = new_jax_md_state.box if is_npt_simulation else initial_box
        if isinstance(jax_md_state.position, list) and not isinstance(_box, list):
            _box = [_box] * len(jax_md_state.position)

        new_neighbors = jax.tree.map(
            lambda n, p, b: n.update(p, box=b),
            old_neighbors,
            new_jax_md_state.position,
            _box,
            is_leaf=is_neighbor_list,
        )
        # Mirror the update for the long-range neighbor list when present.
        old_lr_neighbors = jax.tree.map(
            lambda s: s.long_range_neighbors,
            internal_state.system_state,
            is_leaf=is_system_state,
        )
        has_long_range = jax.tree_util.tree_leaves(old_lr_neighbors) != []
        if has_long_range:
            new_lr_neighbors = jax.tree.map(
                lambda n, p, b: n.update(p, box=b),
                old_lr_neighbors,
                new_jax_md_state.position,
                _box,
                is_leaf=is_neighbor_list,
            )
        else:
            if isinstance(new_neighbors, list):
                new_lr_neighbors = [None] * len(new_neighbors)
            else:
                new_lr_neighbors = None

        new_system_state = jax.tree.map(
            lambda s, n, lr: s.set(neighbors=n, long_range_neighbors=lr),
            internal_state.system_state,
            new_neighbors,
            new_lr_neighbors,
            is_leaf=lambda x: is_system_state(x) or is_neighbor_list(x),
        )

        steps_completed = internal_state.steps_completed + 1
        return internal_state.set(
            jax_md_state=new_jax_md_state,
            episode_log=new_log,
            system_state=new_system_state,
            steps_completed=steps_completed,
        )

    def _update_state(self, episode_idx: int, episode_duration: float) -> None:
        """Updates the simulation state that is publicly accessed by users of
        this class. This means taking the logged data from the episode log and
        concatenating it to the existing arrays in the state.
        """
        log_outputs = self._config.log_outputs

        if log_outputs.positions:
            self.state.positions = self._concat(
                self.state.positions, self._extract_from_log("positions")
            )
        if log_outputs.forces:
            self.state.forces = self._concat(
                self.state.forces, self._extract_from_log("forces")
            )

        self.state.step = (episode_idx + 1) * self._steps_per_episode
        self.state.compute_time_seconds += episode_duration

        if self.is_md_simulation:
            if log_outputs.temperature:
                self.state.temperature = self._concat(
                    self.state.temperature, self._extract_from_log("temperature")
                )
            if log_outputs.kinetic_energy:
                self.state.kinetic_energy = self._concat(
                    self.state.kinetic_energy, self._extract_from_log("kinetic_energy")
                )
            if log_outputs.velocities:
                self.state.velocities = self._concat(
                    self.state.velocities, self._extract_from_log("velocities")
                )

            if log_outputs.cell and self.is_npt_simulation:
                self.state.cell = self._concat(
                    self.state.cell, self._extract_from_log("cell")
                )

    @staticmethod
    def _system_state_from_neighbors(
        neighbors: jax_md.partition.NeighborList,
        long_range_neighbors: jax_md.partition.NeighborList | None = None,
    ) -> SystemState:
        if long_range_neighbors is None:
            return jax.tree.map(
                lambda n: SystemState(neighbors=n),
                neighbors,
                is_leaf=is_neighbor_list,
            )
        return jax.tree.map(
            lambda n, lr: SystemState(neighbors=n, long_range_neighbors=lr),
            neighbors,
            long_range_neighbors,
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

        Written so that it works with batched simulations, too. Also checks the
        long-range neighbor list when it is present.
        """
        did_buffer_overflow = jax.tree.map(
            lambda s: s.neighbors.did_buffer_overflow,
            internal_state.system_state,
            is_leaf=is_system_state,
        )
        long_range_overflow = jax.tree.map(
            lambda s: (
                s.long_range_neighbors.did_buffer_overflow
                if s.long_range_neighbors is not None
                else False
            ),
            internal_state.system_state,
            is_leaf=is_system_state,
        )
        # In batched simulations, we rerun an episode if any of the
        # systems overflowed its buffer
        return np.any(did_buffer_overflow) or np.any(long_range_overflow)

    def _init_base_graph(
        self,
        atoms: ase.Atoms | list[ase.Atoms],
        senders: np.ndarray | list[np.ndarray],
        receivers: np.ndarray | list[np.ndarray],
        senders_long_range: np.ndarray | list[np.ndarray] | None = None,
        receivers_long_range: np.ndarray | list[np.ndarray] | None = None,
    ) -> Graph:
        """Initiates the base graph (batched or unbatched) for the simulation.

        This graph will be given to the jitted calculate function, which will then
        replace relevant parts of it in each simulation step.

        Args:
            atoms: The atoms or list of atoms.
            senders: The sender indices of the edges.
            receivers: The receiver indices of the edges.
            senders_long_range: Optional sender indices of the long-range edges.
            receivers_long_range: Optional receiver indices of the long-range edges.

        Returns:
            The base graph (either batched or unbatched).
        """
        if senders_long_range is None:
            if isinstance(atoms, list):
                senders_long_range = [None] * len(atoms)
                receivers_long_range = [None] * len(atoms)

        graph = jax.tree.map(
            lambda a, s, r, slr, rlr: create_graph_from_atoms_and_edges(
                atoms=a,
                senders=s,
                receivers=r,
                displacement_fun=self._displacement_fun,
                cell_to_box_fun=self._cell_to_box_fun,
                senders_long_range=slr,
                receivers_long_range=rlr,
            ),
            atoms,
            senders,
            receivers,
            senders_long_range,
            receivers_long_range,
        )

        # Batched simulations
        if isinstance(atoms, list):
            assert isinstance(graph, list) and len(graph) == len(atoms)
            # displ_fun is a Callable that dynamically_batch cannot handle.
            # We strip it and replace afterwards with make_batched_displ_fun.
            saved_edges_long_range = graph[0].edges_long_range
            dummy_edges = GraphEdges(shifts=None, displ_fun=None)
            dummy_lr_edges = (
                GraphEdges(shifts=None, displ_fun=None)
                if saved_edges_long_range is not None
                else None
            )
            graph = [
                g.replace(edges=dummy_edges, edges_long_range=dummy_lr_edges)
                for g in graph
            ]
            n_edge_long_range = None
            if graph[0].n_edge_long_range is not None:
                n_edge_long_range = (
                    sum(int(g.n_edge_long_range.item(0)) for g in graph) + 1
                )
            batched_graph = next(
                dynamically_batch(
                    graph,
                    n_node=sum(g.n_node.item(0) for g in graph) + 1,
                    n_edge=sum(g.n_edge.item(0) for g in graph) + 1,
                    n_graph=len(graph) + 1,
                    n_edge_long_range=n_edge_long_range,
                )
            )
            batched_displ_fun = make_batched_displ_fun(
                self._displacement_fun, batched_graph.n_edge
            )
            edges_long_range = None
            if saved_edges_long_range is not None:
                batched_displ_fun_lr = make_batched_displ_fun(
                    self._displacement_fun, batched_graph.n_edge_long_range
                )
                edges_long_range = GraphEdges(
                    shifts=None, displ_fun=batched_displ_fun_lr
                )
            return batched_graph.replace(
                edges=GraphEdges(shifts=None, displ_fun=batched_displ_fun),
                edges_long_range=edges_long_range,
            )

        return graph

    def _update_base_graph_in_pure_sim_step_fun(
        self,
        neighbors: jax_md.partition.NeighborList,
        long_range_neighbors: jax_md.partition.NeighborList | None = None,
    ) -> None:
        """Update `self._pure_simulation_step_fun` after reallocation of neighbors.

        After reallocation of neighbors, the simulation step function needs to
        be updated because the `graph.n_edge` (and possibly `graph.n_edge_long_range`)
        attribute has changed.
        """
        senders = jax.tree.map(
            get_neighbor_list_senders, neighbors, is_leaf=is_neighbor_list
        )
        receivers = jax.tree.map(
            get_neighbor_list_receivers, neighbors, is_leaf=is_neighbor_list
        )
        senders_long_range = None
        receivers_long_range = None
        if long_range_neighbors is not None:
            senders_long_range = jax.tree.map(
                get_neighbor_list_senders,
                long_range_neighbors,
                is_leaf=is_neighbor_list,
            )
            receivers_long_range = jax.tree.map(
                get_neighbor_list_receivers,
                long_range_neighbors,
                is_leaf=is_neighbor_list,
            )
        new_base_graph = self._init_base_graph(
            self._atoms,
            senders,
            receivers,
            senders_long_range,
            receivers_long_range,
        )
        make_model_calculate_fun = functools.partial(
            self._get_model_calculate_fun,
            graph=new_base_graph,
        )
        _, sim_apply_fun = init_simulation_algorithm(
            make_model_calculate_fun,
            self._force_field,
            self._shift_fun,
            self._config,
        )

        self._pure_simulation_step_fun.keywords["apply_fun"] = sim_apply_fun

    def _concat(
        self,
        current: np.ndarray | list[np.ndarray] | None,
        new: np.ndarray | list[np.ndarray],
    ) -> np.ndarray | list[np.ndarray]:
        """Append the new information from the latest episode to the current state.

        Information from every `log_interval` snapshots is added to the state array.
        Uses `tree_map` to work for batched simulations, too.

        Args:
            current: The array representing the current state of one of
                     the state's attributes.
            new: The array representing one of the state's attributes in
                 the last episode.

        Returns:
            The updated array.
        """
        snapshot_interval = self._config.snapshot_interval
        if current is None:
            return jax.tree.map(lambda n: n[::snapshot_interval], new)
        return jax.tree.map(
            lambda c, n: np.concatenate([c, n[::snapshot_interval]], axis=0),
            current,
            new,
        )

    def _extract_from_log(self, attr_name: str) -> np.ndarray | list[np.ndarray]:
        """Small helper using `tree_map` to extract a property from the episode log."""
        episode_log = self._internal_state.episode_log

        return jax.tree.map(
            lambda _log: getattr(_log, attr_name),
            episode_log,
            is_leaf=is_episode_log,
        )
