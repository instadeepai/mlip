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

import logging
import time
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Callable

import ase
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from mlip.models.force_field import ForceField
from mlip.simulation.fep.enums import EpisodeStatus, FEPStage
from mlip.simulation.fep.models.alchemical_forcefield import AlchemicalForceField
from mlip.simulation.fep.models.alchemical_models import to_alchemical_mlip_network
from mlip.simulation.fep.sampler.alchemical_jaxmd_engine import (
    AlchemicalJaxMDSimulationEngine,
)
from mlip.simulation.fep.sampler.fep_sampler_config import FEPSimulationSamplerConfig
from mlip.simulation.fep.sampler.states import FEPSimulationState
from mlip.simulation.fep.sampler.utils import (
    extract_last_step_energies,
    pad_neighbor_lists,
    perform_replica_exchange,
    run_episode_all_engines_gpu,
    run_episode_all_engines_tpu,
)
from mlip.simulation.jax_md.helpers import is_neighbor_list, is_system_state
from mlip.simulation.jax_md.states import JaxMDSimulationState
from mlip.simulation.temperature_scheduling import get_temperature_schedule

logger = logging.getLogger("mlip")


class FEPSimulationSampler:
    """Orchestrator for running Free Energy Perturbation (FEP) simulations.

    Runs multiple simulation engines for different lambda values in parallel across
    local devices, and also supports multi-host deployments on TPU.

    Note that for multi-host deployment, `jax.distributed.initialize()` must be called
    on every host before any JAX operations and before constructing this sampler.
    """

    Config = FEPSimulationSamplerConfig

    def __init__(
        self,
        atoms: ase.Atoms | list[ase.Atoms],
        force_field: ForceField,
        config: FEPSimulationSamplerConfig,
        checkpoint_dir_upload_fun: Callable[[Path], None] | None = None,
    ) -> None:
        """Initialize the FEP sampler.

        Args:
            atoms: Starting configuration(s). Pass a list with one entry per lambda
                to use different starting configurations per lambda; pass a single
                Atoms object to use the same starting configuration for all lambdas.
            force_field: The force field to use for running each alchemical simulation.
                Depending on `config.use_alchemical_mlip`, the force field is either
                converted to an Alchemical MLIP, which scales alchemical edges in its
                update equations, or the alchemical potential is computed by summing
                predictions of the base model on the two endstates.
            config: Simulation configuration.
            checkpoint_dir_upload_fun: An optional function to upload the checkpoints
                directory to remote storage after each checkpoint save.

        Raises:
            NotImplementedError: If the input force field was trained with
                `use_coulomb_term=True`, which is not supported for FEP simulations.
        """
        mlip_config = force_field.predictor.mlip_network.config
        if getattr(mlip_config, "use_coulomb_term", False):
            raise NotImplementedError(
                "FEP simulations cannot be run using a force field with "
                "`use_coulomb_term=True`. Please use a different force field."
            )

        self._config = config
        self._simulation_config = config.simulation_config
        lambda_edge_values = jnp.asarray(config.lambda_edge_values)
        lambda_repulsion_values = jnp.asarray(config.lambda_repulsion_values)
        self._lambda_values = jnp.stack(
            [lambda_edge_values, lambda_repulsion_values], axis=1
        )
        self._using_tpu = jax.default_backend() == "tpu"

        if isinstance(atoms, list):
            if len(atoms) != len(self._lambda_values):
                raise ValueError(
                    "`atoms` list passed to `FEPSimulationSampler` must have the same "
                    "length as `lambda_values`."
                )
            if not np.all([len(a) == len(atoms[0]) for a in atoms]):
                raise ValueError(
                    "`atoms` list passed to `FEPSimulationSampler` must be the same "
                    "system, but may have different starting configurations."
                )

        self.loggers: list[Callable[[FEPSimulationState, int | None], None]] = []
        self._episode_offset = 0

        num_lambdas = len(self._lambda_values)
        atoms_list = atoms if isinstance(atoms, list) else [atoms] * num_lambdas
        self._n_atoms_per_system = len(atoms_list[0].numbers)

        alchemical_force_field = self._build_alchemical_force_field(force_field, config)

        self._engines = self._build_engines(
            atoms_list, alchemical_force_field, config, self._lambda_values
        )
        self._share_neighbor_funs_across_engines()

        # Reallocate neighbors and build per-device step_fns
        self._per_device_step_funs = None
        self._global_reallocate_neighbors()

        self._global_state = FEPSimulationState(replica_exchange_log=[])
        self._latest_repex_energies = {}
        self._repex_key: Array | None = None

        self._checkpoint_dir = (
            config.checkpoint_dir.resolve() if config.checkpoint_dir else None
        )
        self._checkpoint_dir_upload_fun = checkpoint_dir_upload_fun

        if self._using_tpu:
            self._run_compiled_episode_pmap = jax.pmap(
                AlchemicalJaxMDSimulationEngine.run_compiled_episode,
                static_broadcasted_argnums=(0, 1),
                devices=jax.devices(),
            )
        else:
            self._run_compiled_episode = jax.jit(
                AlchemicalJaxMDSimulationEngine.run_compiled_episode,
                static_argnums=(0, 1),
            )

    @property
    def engine_states(self) -> tuple[FEPSimulationState, ...]:
        """The state of each lambda window's engine, in schedule order.

        Used to extract per-engine outputs after running an FEP simulation. Each
        engine state contains a `per_lambda_energies` array, which is required
        for computing a free energy estimate with BAR or MBAR.
        """
        return tuple(engine.state for engine in self._engines)

    @property
    def replica_exchange_log(self) -> list[dict[str, float | None]] | None:
        """The log of attempted replica exchange swaps throughout the simulation.

        Used to evaluate swap frequency and replica exchange paths after running
        an FEP simulation.
        """
        return self._global_state.replica_exchange_log

    @staticmethod
    def _build_alchemical_force_field(
        force_field: ForceField,
        config: FEPSimulationSamplerConfig,
    ) -> AlchemicalForceField:
        """Convert a standard `ForceField` into an `AlchemicalForceField`.

        If `config.use_alchemical_mlip` is True, creates an Alchemical MLIP, which
        scales alchemical edges in its update equations. Otherwise, computes the
        alchemical potential by summing end-state predictions of the base force field.
        """
        if config.use_alchemical_mlip:
            alchemical_mlip_network = to_alchemical_mlip_network(
                force_field.predictor.mlip_network
            )
            fep_stage = None
        else:
            alchemical_mlip_network = force_field.predictor.mlip_network
            fep_stage = FEPStage.A

        alchemical_ff = AlchemicalForceField.from_mlip_network(
            alchemical_mlip_network,
            fep_stage=fep_stage,
            required_properties=force_field.predictor.required_properties,
            repulsive_potential=config.repulsive_potential,
            inference_context=force_field.inference_context,
        )
        return AlchemicalForceField(alchemical_ff.predictor, force_field.params)

    @staticmethod
    def _build_engines(
        atoms_list: list[ase.Atoms],
        force_field: AlchemicalForceField,
        config: FEPSimulationSamplerConfig,
        lambda_values: Array,
    ) -> list[AlchemicalJaxMDSimulationEngine]:
        """Build an alchemical engine for running each lambda value."""
        is_stage_dependent = hasattr(force_field.predictor, "fep_stage")
        engines = []
        for i in range(len(lambda_values)):
            stage_force_field = force_field
            if is_stage_dependent:
                stage = FEPStage.from_edge_weight(float(lambda_values[i][0]))
                stage_force_field = dataclass_replace(
                    force_field,
                    predictor=force_field.predictor.copy(fep_stage=stage),
                )
            engine = AlchemicalJaxMDSimulationEngine(
                atoms_list[i],
                stage_force_field,
                config.simulation_config,
                config.alchemical_atom_indices,
                lambda_values[i],
                lambda_values,
                config.alchemical_energy_batch_size,
            )
            engines.append(engine)
        return engines

    def _get_engine_stage(
        self, engine: AlchemicalJaxMDSimulationEngine
    ) -> FEPStage | None:
        """Returns the FEP stage for the engine, for mapping to a shared step_fun.

        On TPU, merges `FEPStage.A` into `FEPStage.AR`, to avoid running the `A`
        group alone under pmap (see module docstring for the TPU dispatch model).
        """
        if self._using_tpu:
            return FEPStage.AR if engine.fep_stage == FEPStage.A else engine.fep_stage
        return engine.fep_stage

    def _share_neighbor_funs_across_engines(self) -> None:
        """Share `NeighborListFns` between engines to ensure shared compilation."""
        shared_neighbor_fun = self._engines[0]._neighbor_fun
        shared_long_range_neighbor_fun = self._engines[0]._long_range_neighbor_fun
        for engine in self._engines[1:]:
            engine._neighbor_fun = shared_neighbor_fun
            engine._long_range_neighbor_fun = shared_long_range_neighbor_fun

    def attach_logger(
        self, logger: Callable[[FEPSimulationState, int | None], None]
    ) -> None:
        """Adds a logger to the list of loggers of the sampler.

        The logger must take two arguments: an `FEPSimulationState` and an integer
        referencing which engine the state refers to (or None if global state).

        Args:
            logger: The logger to add.
        """
        self.loggers.append(logger)

    def _call_engine_loggers(self) -> None:
        """Run all loggers for every engine."""
        if jax.process_index() != 0:
            return
        for engine_index, engine in enumerate(self._engines):
            for _logger in self.loggers:
                _logger(engine.state, engine_index)

    def _call_global_loggers(self) -> None:
        """Run all loggers for global state."""
        if jax.process_index() != 0:
            return
        for _logger in self.loggers:
            _logger(self._global_state, None)

    def _run_episode_all_engines(
        self,
    ) -> tuple[EpisodeStatus, dict[int, JaxMDSimulationState]]:
        """Run one episode for all lambda windows across available devices."""
        if self._using_tpu:
            return run_episode_all_engines_tpu(
                self._engines,
                [self._get_engine_stage(e) for e in self._engines],
                self._per_device_step_funs[jax.local_devices()[0]],
                self._run_compiled_episode_pmap,
            )
        return run_episode_all_engines_gpu(
            self._engines, self._per_device_step_funs, self._run_compiled_episode
        )

    def _perform_replica_exchange(
        self, episode_idx: int, temperature_schedule: Callable
    ) -> None:
        """Perform one round of Hamiltonian Replica Exchange between all lambdas."""
        states = [engine.get_internal_state() for engine in self._engines]
        (
            states,
            self._global_state.replica_exchange_log,
            self._repex_key,
        ) = perform_replica_exchange(
            self._latest_repex_energies,
            episode_idx,
            temperature_schedule,
            states,
            self._lambda_values,
            self._repex_key,
            self._global_state.replica_exchange_log,
        )
        for engine, state in zip(self._engines, states):
            engine.set_internal_state(state)

    def _pad_all_neighbor_lists(self) -> None:
        """Pad neighbor lists across all engines to the same capacity."""
        ref_engine = self._engines[0]
        padding_idx = (
            ref_engine._num_atoms
            if isinstance(ref_engine._num_atoms, int)
            else ref_engine._num_atoms[0]
        )
        has_long_range_edges = ref_engine.has_long_range_edges

        internal_states = [engine.get_internal_state() for engine in self._engines]
        neighbors_list = [s.system_state.neighbors for s in internal_states]
        padded_neighbors = pad_neighbor_lists(neighbors_list, padding_idx)

        if has_long_range_edges:
            lr_list = [s.system_state.long_range_neighbors for s in internal_states]
            padded_lr = pad_neighbor_lists(lr_list, padding_idx)

        for i, (engine, state) in enumerate(zip(self._engines, internal_states)):
            new_sys_state = jax.tree.map(
                lambda s, n: s.set(neighbors=n),
                state.system_state,
                padded_neighbors[i],
                is_leaf=lambda x: is_system_state(x) or is_neighbor_list(x),
            )
            if has_long_range_edges:
                new_sys_state = jax.tree.map(
                    lambda s, lr: s.set(long_range_neighbors=lr),
                    new_sys_state,
                    padded_lr[i],
                    is_leaf=lambda x: is_system_state(x) or is_neighbor_list(x),
                )
            engine.set_internal_state(state.set(system_state=new_sys_state))

    def _build_per_device_step_funs(self) -> dict[jax.Device, Callable]:
        """Build {device: {stage: step_fun}} by calling one engine per stage.

        On TPU, builds a single canonical step_fun per stage, rather than one
        per local device, since pmap replicates a single compiled program.
        """
        grouped_engines: dict[FEPStage | None, AlchemicalJaxMDSimulationEngine] = {}
        for engine in self._engines:
            stage = self._get_engine_stage(engine)
            current = grouped_engines.get(stage)
            # Required as we merge A and AR stages for TPU to store correct engine:
            if current is None or (
                current.fep_stage != stage and engine.fep_stage == stage
            ):
                grouped_engines[stage] = engine

        devices = [jax.local_devices()[0]] if self._using_tpu else jax.local_devices()
        device_step_funs: dict = {device: {} for device in devices}
        for key, engine in grouped_engines.items():
            per_device = engine.build_per_device_step_fun()
            for device in devices:
                device_step_funs[device][key] = per_device[device]

        return device_step_funs

    def _global_reallocate_neighbors(self) -> None:
        """Reallocate neighbors for all engines, pad to common size, and recompile."""
        logger.info("Running global neighbor reallocation...")
        for engine in self._engines:
            engine._reallocate_neighbors()
        self._pad_all_neighbor_lists()
        self._per_device_step_funs = self._build_per_device_step_funs()
        logger.info("Global reallocation completed.")

    def run(self) -> None:
        """Run all lambda windows for FEP with multi-device parallelism."""
        if self._config.use_replica_exchange and self._repex_key is None:
            self._repex_key = jax.random.PRNGKey(self._simulation_config.random_seed)

        logger.info("Starting simulation...")
        episode_idx = 0
        temperature_schedule = get_temperature_schedule(
            self._simulation_config.temperature_schedule_config,
            self._simulation_config.num_steps,
        )
        logger.info("Dispatching across %d device(s).", len(jax.local_devices()))
        abs_episode_idx = self._episode_offset - 1

        while episode_idx < self._simulation_config.num_episodes:
            start_time = time.perf_counter()
            episode_status, completed_states = self._run_episode_all_engines()

            if episode_status == EpisodeStatus.OVERFLOW:
                logger.info(
                    "Overflow in neighbour list. Reallocating for all lambdas "
                    "and rerunning episode."
                )
                self._global_reallocate_neighbors()
                continue

            episode_duration = time.perf_counter() - start_time
            abs_episode_idx = self._episode_offset + episode_idx
            logger.info(
                "Episode %s completed for all lambdas in %.2f s.",
                abs_episode_idx + 1,
                episode_duration,
            )

            # Commit completed states and update per-lambda simulation states.
            for engine_index, completed_state in completed_states.items():
                engine = self._engines[engine_index]
                engine.set_internal_state(completed_state)
                engine._update_state(episode_idx, episode_duration)
                if self._config.use_replica_exchange:
                    self._latest_repex_energies[engine_index] = (
                        extract_last_step_energies(completed_state.episode_log)
                    )

            self._call_engine_loggers()
            if (episode_idx + 1) % self._config.checkpoint_interval_episodes == 0:
                self._save_checkpoint(abs_episode_idx + 1)

            if episode_status == EpisodeStatus.EXPLODED:
                logger.error("Simulation exploded. Exiting simulation early.")
                break

            if self._config.use_replica_exchange:
                if abs_episode_idx + 1 > self._config.num_equilibration_episodes:
                    logger.info(
                        "Performing replica exchange (episode %s).", abs_episode_idx
                    )
                    self._perform_replica_exchange(
                        abs_episode_idx, temperature_schedule
                    )
                else:
                    logger.info("Skipping replica exchange (equilibration phase).")

            self._call_global_loggers()
            episode_idx += 1

        self._call_engine_loggers()
        self._save_checkpoint(abs_episode_idx + 1)
        logger.info("Simulation completed.")

    def _save_checkpoint(self, completed_episodes: int) -> None:
        """Save and upload a complete checkpoint to checkpoint_dir.

        Saves per-lambda jax_md_state leaves, the replica exchange PRNG key, and the
        completed episode count, to enable restoring a run if interrupted.

        Args:
            completed_episodes: Total number of episodes completed so far (absolute).
        """
        if jax.process_index() != 0 or self._checkpoint_dir is None:
            return

        logger.info("Saving checkpoint at episode %s...", completed_episodes)

        for i, engine in enumerate(self._engines):
            lambda_dir = self._checkpoint_dir / f"lambda_{i}"
            lambda_dir.mkdir(parents=True, exist_ok=True)
            leaves = jax.tree_util.tree_leaves(engine.get_internal_state().jax_md_state)
            for j, leaf in enumerate(leaves):
                np.save(lambda_dir / f"jax_md_state_leaf_{j}.npy", np.array(leaf))

        episode_count_path = self._checkpoint_dir / "episode_count.npy"
        np.save(episode_count_path, np.array(completed_episodes))
        if self._config.use_replica_exchange and self._repex_key is not None:
            np.save(self._checkpoint_dir / "repex_key.npy", np.array(self._repex_key))

        if self._checkpoint_dir_upload_fun is not None:
            self._checkpoint_dir_upload_fun(self._checkpoint_dir)

    def restore_checkpoint(self, checkpoint_dir: Path) -> None:
        """Restore complete sampler state from a local checkpoint directory.

        Restores per-lambda jax_md_state leaves, the replica exchange PRNG key, and
        infers `episode_offset` from the saved episode count.

        Note that this method must read from a local directory. If the checkpoint
        directory is in remote storage, download it before calling this method.

        Args:
            checkpoint_dir: Local directory written by `_save_checkpoint`.
        """
        checkpoint_dir = Path(checkpoint_dir)
        for i, engine in enumerate(self._engines):
            internal_state = engine.get_internal_state()
            ref_jax_md_state = internal_state.jax_md_state
            leaves, treedef = jax.tree_util.tree_flatten(ref_jax_md_state)
            lambda_dir = checkpoint_dir / f"lambda_{i}"
            loaded_leaves = [
                jnp.array(np.load(lambda_dir / f"jax_md_state_leaf_{j}.npy"))
                for j in range(len(leaves))
            ]
            restored = treedef.unflatten(loaded_leaves)
            engine.set_internal_state(internal_state.set(jax_md_state=restored))

        episode_count = int(np.load(checkpoint_dir / "episode_count.npy"))
        self._episode_offset = episode_count

        repex_key_path = checkpoint_dir / "repex_key.npy"
        if self._config.use_replica_exchange and repex_key_path.exists():
            self._repex_key = jnp.array(np.load(repex_key_path))

        logger.info(
            "Restored checkpoint: %d lambda windows, episode_offset=%d.",
            len(self._engines),
            episode_count,
        )
