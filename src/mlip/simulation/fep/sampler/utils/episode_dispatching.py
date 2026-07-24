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

import concurrent.futures
import logging
import queue
import time
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.experimental.multihost_utils import process_allgather, sync_global_devices

from mlip.simulation.fep.enums import EpisodeStatus, FEPStage
from mlip.simulation.fep.sampler.alchemical_jaxmd_engine import (
    AlchemicalJaxMDSimulationEngine,
)
from mlip.simulation.jax_md.states import JaxMDSimulationState

logger = logging.getLogger("mlip")


def _detect_episode_status(
    completed_states: dict[int, JaxMDSimulationState],
) -> EpisodeStatus:
    """Classify overall episode status from per-engine completed states."""
    episode_status = EpisodeStatus.SUCCESS
    for engine_index, state in completed_states.items():
        if AlchemicalJaxMDSimulationEngine._has_simulation_exploded(state):
            logger.error("Simulation exploded for engine %s.", engine_index)
            episode_status = EpisodeStatus.EXPLODED
        elif AlchemicalJaxMDSimulationEngine._did_neighbor_buffer_overflow(state):
            logger.warning("Neighbor list overflow in engine %s.", engine_index)
            if episode_status != EpisodeStatus.EXPLODED:
                episode_status = EpisodeStatus.OVERFLOW
    return episode_status


def _launch_episode_single_engine(
    engine: AlchemicalJaxMDSimulationEngine,
    device: jax.Device,
    per_device_step_funs: dict[jax.Device, dict[FEPStage, Callable]],
    run_compiled_episode: Callable,
) -> JaxMDSimulationState:
    """Dispatch a single-episode simulation to `device` without blocking."""
    step_fun = per_device_step_funs[device][engine.fep_stage]
    state = jax.device_put(engine.get_internal_state(), device)
    return run_compiled_episode(engine._steps_per_episode, step_fun, state)


def run_episode_all_engines_gpu(
    engines: Sequence[AlchemicalJaxMDSimulationEngine],
    per_device_step_funs: dict[jax.Device, dict[FEPStage, Callable]],
    run_compiled_episode: Callable,
) -> tuple[EpisodeStatus, dict[int, JaxMDSimulationState]]:
    """Dispatch one episode per engine independently, via a thread pool.

    Each engine is compiled and launched separately; CUDA handles these
    concurrently via separate streams.

    Args:
        engines: All simulation engines.
        per_device_step_funs: Compiled step_fun per device per FEP stage, as
            built by `FEPSimulationSampler._build_per_device_step_funs`.
        run_compiled_episode: jitted `run_compiled_episode`. Must be built once
            and reused across calls to prevent recompilation.

    Returns:
        The episode status for all engines and a mapping from engine index to its
        completed `JaxMDSimulationState`.
    """
    completed_states: dict[int, JaxMDSimulationState] = {}

    device_queue: queue.Queue = queue.Queue()
    for device in jax.local_devices():
        device_queue.put(device)

    def _dispatch(engine_index: int):
        target_device = device_queue.get()
        try:
            lambda_start = time.perf_counter()
            future_state = _launch_episode_single_engine(
                engines[engine_index],
                target_device,
                per_device_step_funs,
                run_compiled_episode,
            )
            completed_state = jax.block_until_ready(future_state)
            runtime = time.perf_counter() - lambda_start
            return engine_index, completed_state, target_device, runtime
        finally:
            device_queue.put(target_device)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines)) as executor:
        futures = {executor.submit(_dispatch, i): i for i in range(len(engines))}
        for future in concurrent.futures.as_completed(futures):
            engine_index, completed_state, used_device, runtime = future.result()
            logger.info(
                "Engine %s: Episode completed in %.2f s (device %s).",
                engine_index,
                runtime,
                used_device,
            )
            completed_states[engine_index] = completed_state

    return _detect_episode_status(completed_states), completed_states


def _run_pmap_group(
    engines: Sequence[AlchemicalJaxMDSimulationEngine],
    engine_indices: list[int],
    step_fun: Callable,
    run_compiled_episode_pmap: Callable,
) -> dict[int, JaxMDSimulationState]:
    """Run one episode for a group of engines sharing `step_fun`, via pmap.

    Engines are batched into chunks of `n_pmap_devices` along a leading axis.
    Results are synchronized before returning so all hosts can access all states.

    Args:
        engines: All simulation engines.
        engine_indices: Indices of all engines sharing a `step_fun`.
        step_fun: The compiled step function shared by all input engines.
        run_compiled_episode_pmap: pmapped `run_compiled_episode`. Must be
            built once and reused across calls.

    Returns:
        Mapping from engine index to the completed `JaxMDSimulationState`.
        Identical on every host after the allgather.
    """
    n_pmap_devices = len(jax.devices())
    n_local_devices = len(jax.local_devices())
    n_engines = len(engine_indices)
    pad = (-n_engines) % n_pmap_devices
    padded_indices = engine_indices + [engine_indices[-1]] * pad
    steps = engines[engine_indices[0]]._steps_per_episode

    results: dict[int, JaxMDSimulationState] = {}

    # Ensure all hosts enter the pmap together.
    sync_global_devices("pmap_group_start")

    for chunk_start in range(0, len(padded_indices), n_pmap_devices):
        chunk_indices = padded_indices[chunk_start : chunk_start + n_pmap_devices]
        real_count = min(n_pmap_devices, n_engines - chunk_start)

        # Each host stacks only its contiguous local slice of the chunk.
        process_id = jax.process_index()
        local_indices = chunk_indices[
            process_id * n_local_devices : (process_id + 1) * n_local_devices
        ]
        local_states = [engines[i].get_internal_state() for i in local_indices]
        local_batched = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *local_states)

        # Single SPMD dispatch: one XLA executable across all hosts/chips.
        start_time = time.perf_counter()
        local_result = run_compiled_episode_pmap(steps, step_fun, local_batched)
        jax.block_until_ready(local_result)
        elapsed = time.perf_counter() - start_time

        # All-gather results across hosts, returned as [n_pmap_devices, ...]
        global_result = process_allgather(local_result, tiled=True)

        if jax.process_index() == 0:
            logger.info(
                "pmap chunk [engines %s, +%d pad]: %.2f s across %d chip(s).",
                engine_indices[chunk_start : chunk_start + real_count],
                n_pmap_devices - real_count,
                elapsed,
                n_pmap_devices,
            )

        # Unstack per-engine results; discard padded slots.
        for slot in range(n_pmap_devices):
            real_idx = chunk_start + slot
            if real_idx >= n_engines:
                break
            results[engine_indices[real_idx]] = jax.tree.map(
                lambda x: x[slot], global_result
            )

    return results


def run_episode_all_engines_tpu(
    engines: Sequence[AlchemicalJaxMDSimulationEngine],
    engine_stages: list[FEPStage],
    per_stage_step_funs: dict[FEPStage, Callable],
    run_compiled_episode_pmap: Callable,
) -> tuple[EpisodeStatus, dict[int, JaxMDSimulationState]]:
    """Run one episode for all engines via pmap, grouped by shared step_fun.

    Args:
        engines: All lambda engines, indexed like `engine_stages`.
        engine_stages: The stage for each engine, by index.
        per_stage_step_funs: Compiled step_fun per stage, to be shared across each pmap.
        run_compiled_episode_pmap: pmapped `run_compiled_episode`. Must be built
            once by `FEPSimulationSampler`, then reused across calls.

    Returns:
        The episode status for all engines and a mapping from engine index to its
        completed `JaxMDSimulationState`.
    """
    grouped_indices: dict[FEPStage, list[int]] = {}
    for i, stage in enumerate(engine_stages):
        grouped_indices.setdefault(stage, []).append(i)

    completed_states: dict[int, JaxMDSimulationState] = {}
    for stage, indices in grouped_indices.items():
        completed_states.update(
            _run_pmap_group(
                engines, indices, per_stage_step_funs[stage], run_compiled_episode_pmap
            )
        )

    return _detect_episode_status(completed_states), completed_states
