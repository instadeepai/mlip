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

import dataclasses
import logging
from dataclasses import replace
from typing import Callable

import ase
import jax
import jax.numpy as jnp
import numpy as np

from mlip.graph import Graph
from mlip.models.force_field import ForceField
from mlip.simulation.jax_md.helpers import is_episode_log
from mlip.simulation.jax_md.jax_md_simulation_engine import (
    JaxMDSimulationEngine,
    ModelEnergyFun,
    ModelForcesFun,
)
from mlip.simulation.jax_md.states import JaxMDSimulationState
from mlip.simulation.metadynamics.config import (
    JaxMDMetadynamicsSimulationConfig,
    MetadynamicsConfig,
)
from mlip.simulation.metadynamics.helpers import (
    build_metadynamics_energy_head,
    update_graph_in_metadynamics_simulation_step,
    update_metadynamics_state,
)
from mlip.simulation.metadynamics.states import (
    MetadynamicsEpisodeLog,
    MetadynamicsSimulationState,
    MetadynamicsState,
    MetadynamicsSystemState,
)

logger = logging.getLogger("mlip")


class JaxMDMetadynamicsSimulationEngine(JaxMDSimulationEngine):
    """Simulation engine for well-tempered metadynamics using the JAX-MD backend.

    Extends `JaxMDSimulationEngine` to periodically deposit Gaussian hills along
    one or two collective variables (CVs), optionally applying well-tempered
    rescaling, wall potentials, and positional restraints.

    Only non-batched simulations are supported. The metadynamics state (hill centers
    and heights) is carried inside `MetadynamicsSystemState` and injected into the
    `Graph` globals at each step so bias potentials can access it.
    """

    Config = JaxMDMetadynamicsSimulationConfig
    simulation_state_class = MetadynamicsSimulationState

    def __init__(
        self,
        atoms: ase.Atoms | list[ase.Atoms],
        force_field: ForceField,
        config: JaxMDMetadynamicsSimulationConfig,
    ) -> None:
        super().__init__(atoms, force_field, config)

    def _initialize(
        self,
        atoms: ase.Atoms | list[ase.Atoms],
        force_field: ForceField,
        config: JaxMDMetadynamicsSimulationConfig,
    ) -> None:
        """Initialize the metadynamics engine.

        Resolves the metadynamics config, builds the initial hill state, calls the
        parent initializer, and wraps the resulting `SystemState` with a
        `MetadynamicsSystemState` carrying the hill buffer.

        Args:
            atoms: The atoms of the system to simulate. Batched input is not supported.
            force_field: The force field to use in the simulation.
            config: Simulation configuration.
        """
        if isinstance(atoms, list):
            raise NotImplementedError("Cannot run batched metadynamics simulations.")

        self.metadynamics_config: MetadynamicsConfig = (
            config.metadynamics_config.resolve(atoms, config.temperature_kelvin)
        )
        self._initial_metadynamics_state = self._init_metadynamics_state()

        super()._initialize(atoms, force_field, config)

        system_state = self._internal_state.system_state
        new_system_state = MetadynamicsSystemState(
            neighbors=system_state.neighbors,
            long_range_neighbors=system_state.long_range_neighbors,
            metadynamics_state=self._initial_metadynamics_state,
        )

        self._internal_state = self._internal_state.set(system_state=new_system_state)

    def _init_metadynamics_state(self) -> MetadynamicsState:
        """Create the initial (empty) `MetadynamicsState` hill buffer.

        Allocates zero-filled arrays of size `max_gaussians` for the hill centers
        and heights, and logs the metadynamics setup to the mlip logger.

        Returns:
            A `MetadynamicsState` with all hill slots zeroed and `num_gaussians=0`.
        """
        max_gaussians = self.metadynamics_config.max_gaussians
        num_cvs = len(self.metadynamics_config.bias_cvs)
        metadynamics_state = MetadynamicsState(
            gaussian_centers=jnp.zeros((max_gaussians, num_cvs)),
            gaussian_heights=jnp.zeros(max_gaussians),
            num_gaussians=0,
        )
        wt_str = (
            f"well-tempered (γ={self.metadynamics_config.bias_factor})"
            if self.metadynamics_config.bias_factor is not None
            else "plain"
        )
        cv_descs = [c.type for c in self.metadynamics_config.bias_cvs]
        logger.info(
            "Metadynamics enabled [%s]: cvs=[%s], h0=%.4f eV",
            wt_str,
            ", ".join(cv_descs),
            self.metadynamics_config.initial_height,
        )
        return metadynamics_state

    def _setup_sim_functions(self, base_graph: Graph) -> tuple[Callable, Callable]:
        """Set up simulation functions, injecting metadynamics config into the step fn.

        Calls the parent implementation and then binds `metadynamics_config` and
        `base_graph` as keyword arguments on the partial step function so they are
        available inside `jax.lax.fori_loop` without capturing them as closures.

        Args:
            base_graph: The base graph built from the input `ase.Atoms`.

        Returns:
            The tuple of `(sim_init_fun, pure_simulation_step_fun)`.
        """
        sim_init_fun, pure_simulation_step_fun = super()._setup_sim_functions(
            base_graph
        )
        pure_simulation_step_fun.keywords["metadynamics_config"] = (
            self.metadynamics_config
        )
        pure_simulation_step_fun.keywords["base_graph"] = base_graph
        return sim_init_fun, pure_simulation_step_fun

    def _get_model_calculate_fun(
        self,
        graph: Graph,
        force_field_model: ForceField,
        is_energy_fun: bool,
        return_aux_properties: bool,
    ) -> ModelEnergyFun | ModelForcesFun:
        """Return the JAX-MD-compatible model function, wrapping in metadynamics energy.

        When building the forces function (`is_energy_fun=False`), replaces the force
        field's energy head with one that also evaluates bias, wall, and restraint
        potentials. The energy function is left unchanged so that energy logging
        does not double-count the metadynamics bias.

        Args:
            graph: Base graph used to trace the calculation function.
            force_field_model: The force field to use.
            is_energy_fun: Whether to return the energy function (`True`) or the
                forces function (`False`).
            return_aux_properties: Whether the returned function should also return
                auxiliary properties alongside the forces/energy.

        Returns:
            A JAX-MD-compatible callable for energy or forces.
        """
        if self.metadynamics_config is not None and not is_energy_fun:
            force_field_model = self._create_metadynamics_force_field(force_field_model)
        return super()._get_model_calculate_fun(
            graph, force_field_model, is_energy_fun, return_aux_properties
        )

    def _create_metadynamics_force_field(
        self, force_field_model: ForceField
    ) -> ForceField:
        """Return a copy of the force field with the metadynamics energy head."""
        metadynamics_config = self.metadynamics_config
        bias_potential = metadynamics_config.build_bias_potential()
        wall_potentials = metadynamics_config.build_wall_potentials()
        restraint_potentials = metadynamics_config.build_restraint_potentials()
        metadynamics_potentials = (
            [bias_potential] + wall_potentials + restraint_potentials
        )

        base_energy_head = force_field_model.predictor.energy_head
        metadynamics_energy_head = build_metadynamics_energy_head(
            base_energy_head, metadynamics_potentials
        )
        new_predictor = dataclasses.replace(
            force_field_model.predictor, energy_head=metadynamics_energy_head
        )
        return replace(force_field_model, predictor=new_predictor)

    def _init_base_graph(self, *args, **kwargs) -> Graph:
        """Build the base graph and seed it with the initial metadynamics hill state.

        Calls the parent implementation and then writes the initial (empty) Gaussian
        centers, heights, and count into the graph's global features so they are
        available to bias potentials from the very first simulation step.

        Returns:
            The base graph with metadynamics global features populated.
        """
        graph = super()._init_base_graph(*args, **kwargs)
        s = self._initial_metadynamics_state
        return graph.update_global_features(
            gaussian_centers=s.gaussian_centers,
            gaussian_heights=s.gaussian_heights,
            num_gaussians=s.num_gaussians,
        )

    def _get_update_graph_in_sim_step_fun(self):
        """Override parent method to add metadynamics values to the global features."""
        return update_graph_in_metadynamics_simulation_step

    def _init_episode_log(self, num_atoms: int) -> MetadynamicsEpisodeLog:
        """Initialize the episode log with metadynamics CV and bias tracking arrays.

        Extends the parent's `EpisodeLog` into a `MetadynamicsEpisodeLog` by
        adding zero-filled arrays for per-step bias potential and CV values.

        Args:
            num_atoms: Number of atoms in the simulated system.

        Returns:
            A `MetadynamicsEpisodeLog` with all fields initialized to zero.
        """
        episode_log = super()._init_episode_log(num_atoms)
        num_cvs = len(self.metadynamics_config.bias_cvs)
        return MetadynamicsEpisodeLog(
            **{
                f.name: getattr(episode_log, f.name)
                for f in dataclasses.fields(episode_log)
            },
            bias_potential=jnp.zeros((self._steps_per_episode,)),
            bias_cv_values=jnp.zeros((self._steps_per_episode, num_cvs)),
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
        base_graph: Graph,
        metadynamics_config: MetadynamicsConfig,
    ) -> JaxMDSimulationState:
        """Execute one metadynamics step: base MD step + CV logging + hill deposition.

        Delegates the core integration step to the base `_simulation_step_fun`,
        then evaluates the bias potential and current CV values, writes them into the
        episode log, and conditionally deposits a new Gaussian hill according to
        `metadynamics_config.deposition_interval`.

        Args:
            step_idx: Index of the current step within the episode (0-based).
            internal_state: Current simulation state, including the metadynamics
                hill buffer in `system_state.metadynamics_state`.
            apply_fun: JAX-MD apply function from the simulation algorithm.
            temperature_schedule: Callable mapping global step to temperature (K).
            is_md_simulation: Whether this is an MD (vs. minimization) run.
            is_npt_simulation: Whether this is an NPT ensemble run.
            initial_box: Simulation box array, or `None` for non-periodic systems.
            base_graph: Static base graph used to construct the per-step graph.
            metadynamics_config: Resolved metadynamics settings including CVs,
                bias factor, and deposition interval.

        Returns:
            Updated `JaxMDSimulationState` with new hill state, logged CV values,
            and logged bias potential.
        """
        # Perform base step function first
        internal_state = JaxMDSimulationEngine._simulation_step_fun(
            step_idx,
            internal_state,
            apply_fun,
            temperature_schedule,
            is_md_simulation,
            is_npt_simulation,
            initial_box,
        )

        # === Metadynamics: log CVs, bias, and deposit Gaussian hill ===
        cfg = metadynamics_config
        new_log = internal_state.episode_log
        metadynamics_state = internal_state.system_state.metadynamics_state

        positions = internal_state.jax_md_state.position
        _box = internal_state.jax_md_state.box if is_npt_simulation else initial_box
        graph = update_graph_in_metadynamics_simulation_step(
            internal_state.system_state,
            positions,
            base_graph,
            is_batched=False,
            box=_box,
        )

        bias_potential = cfg.build_bias_potential()
        current_bias = bias_potential(graph)
        bias_cv_values = bias_potential.compute_cvs(graph)

        # Log CV values and bias
        new_log = jax.tree.map(
            lambda _log: _log.set(
                bias_cv_values=_log.bias_cv_values.at[step_idx].set(bias_cv_values),
                bias_potential=_log.bias_potential.at[step_idx].set(current_bias),
            ),
            new_log,
            is_leaf=is_episode_log,
        )

        # Well-tempered scaling: h_k = h0 · exp(-V_bias / (γ · kBT))
        bias_factor = cfg.bias_factor
        if bias_factor is not None:
            scaled_height = cfg.initial_height * jnp.exp(
                -current_bias / (bias_factor * cfg.thermal_energy_ev)
            )
        else:
            scaled_height = jnp.array(cfg.initial_height)

        # Deposit hill every N steps (skip step 0; guard against buffer overflow)
        global_step = internal_state.steps_completed - 1
        should_deposit = (global_step > 0) & (
            (global_step % cfg.deposition_interval) == 0
        )

        metadynamics_state = jax.lax.cond(
            should_deposit,
            lambda s: update_metadynamics_state(s, bias_cv_values, scaled_height),
            lambda s: s,
            metadynamics_state,
        )

        internal_state = internal_state.set(
            episode_log=new_log,
            system_state=internal_state.system_state.set(
                metadynamics_state=metadynamics_state,
            ),
        )

        return internal_state

    def _update_state(self, episode_idx: int, episode_duration: float) -> None:
        """Update the public simulation state after each episode.

        Calls the parent implementation to append positions, forces, temperature,
        etc., then additionally appends the per-step CV values and bias potential
        from the episode log, and refreshes the cumulative Gaussian hill arrays
        from the internal metadynamics state.

        Args:
            episode_idx: Zero-based index of the completed episode.
            episode_duration: Wall-clock duration of the episode in seconds.
        """
        super()._update_state(episode_idx, episode_duration)

        self.state.bias_potential = self._concat(
            self.state.bias_potential, self._extract_from_log("bias_potential")
        )
        self.state.bias_cv_values = self._concat(
            self.state.bias_cv_values, self._extract_from_log("bias_cv_values")
        )

        # gaussian_centers/heights are added cumulatively, not stored per-frame.
        metadynamics_state = self._internal_state.system_state.metadynamics_state
        num_gaussians = int(metadynamics_state.num_gaussians)
        self.state.gaussian_centers = np.array(
            metadynamics_state.gaussian_centers[:num_gaussians]
        )
        self.state.gaussian_heights = np.array(
            metadynamics_state.gaussian_heights[:num_gaussians]
        )
