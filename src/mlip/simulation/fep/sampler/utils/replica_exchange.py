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
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from mlip.simulation.fep.sampler.states import FEPEpisodeLog
from mlip.simulation.jax_md.helpers import (
    KCAL_PER_MOL_PER_ELECTRON_VOLT,
    TEMPERATURE_CONVERSION_FACTOR,
)
from mlip.simulation.jax_md.states import JaxMDSimulationState

logger = logging.getLogger("mlip")


def extract_last_step_energies(episode_log: FEPEpisodeLog) -> Array:
    """Extract the final-step per-lambda energies from an episode log."""
    return jax.device_put(
        jnp.asarray(episode_log.per_lambda_energies[-1]), jax.local_devices()[0]
    )


def _metropolis_criterion(
    cycle_energies_i: list[dict[str, Array | None]],
    cycle_energies_j: list[dict[str, Array | None]],
    temperature_kelvin: float,
) -> tuple[Array, Array]:
    """Compute the Metropolis criterion for two adjacent replicas (j = i + 1).

    Computes the delta as
        delta_energy = (E_i_at_j + E_j_at_i) - (E_i_at_i + E_j_at_j)
    where:
        E_i_at_j is 'up' for replica i at lambda j
        E_j_at_i is 'down' for replica j at lambda i

    The probability is computed as:
        probability = min(1.0, exp(-delta_energy / kT))

    Args:
        cycle_energies_i: The cycle energies for one replica.
        cycle_energies_j: The cycle energies for the next replica.
        temperature_kelvin: The temperature in Kelvin.

    Returns:
        Tuple containing the delta energy and the acceptance probability.
    """
    u_i_i = cycle_energies_i["current"]
    u_j_j = cycle_energies_j["current"]
    u_i_j = cycle_energies_i["up"]
    u_j_i = cycle_energies_j["down"]

    delta_energy = (u_i_j + u_j_i) - (u_i_i + u_j_j)
    if delta_energy <= 0:
        acceptance_probability = 1.0
    else:
        temperature_kt = temperature_kelvin * TEMPERATURE_CONVERSION_FACTOR
        acceptance_probability = jnp.exp(-(1.0 / temperature_kt) * delta_energy)
    return delta_energy, acceptance_probability


def _extract_lambda_energy(per_lambda_energies: Array, lambda_index: int) -> Array:
    """Return the total alchemical energy at a given lambda index in kcal/mol."""
    return per_lambda_energies[lambda_index] * KCAL_PER_MOL_PER_ELECTRON_VOLT


def _compute_all_cycle_energies(
    latest_repex_energies: dict[int, Array],
    num_lambdas: int,
) -> list[dict[str, Array | None]]:
    """Compute current/up/down cycle energies for every lambda window."""
    cycle_energies = []
    for i in range(num_lambdas):
        energies = latest_repex_energies[i]
        e_current = _extract_lambda_energy(energies, i)
        e_up = _extract_lambda_energy(energies, i + 1) if i < num_lambdas - 1 else None
        e_down = _extract_lambda_energy(energies, i - 1) if i > 0 else None
        cycle_energies.append({"current": e_current, "up": e_up, "down": e_down})
    return cycle_energies


def _swap_lambdas(
    i: int, j: int, lambda_internal_states: list[JaxMDSimulationState]
) -> list[JaxMDSimulationState]:
    """Return a new list with configurations at slots i and j swapped."""

    def _build_state(
        config_state: JaxMDSimulationState, lambda_vals: jnp.ndarray
    ) -> JaxMDSimulationState:
        return config_state.set(
            system_state=config_state.system_state.set(
                lambda_values=lambda_vals,
            )
        )

    state_i = lambda_internal_states[i]
    state_j = lambda_internal_states[j]
    new_states = list(lambda_internal_states)
    new_states[i] = _build_state(state_j, state_i.system_state.lambda_values)
    new_states[j] = _build_state(state_i, state_j.system_state.lambda_values)
    return new_states


def _run_repex_round(
    cycle_energies: list[dict[str, Array | None]],
    episode_idx: int,
    temperature_schedule: Callable[[int], float],
    lambda_internal_states: list[JaxMDSimulationState],
    lambda_values: Array,
    repex_rng: Array,
    replica_exchange_log: list[dict[str, Any]],
) -> tuple[list[JaxMDSimulationState], list[dict[str, Any]], Array]:
    """Apply one round of Metropolis-criterion swaps given pre-computed cycle energies.

    Returns updated (lambda_internal_states, replica_exchange_log, repex_rng).
    """
    replica_exchange_log = list(replica_exchange_log)
    offset = episode_idx % 2
    for i in range(offset, len(lambda_values) - 1, 2):
        j = i + 1
        temperature_kelvin = temperature_schedule(
            lambda_internal_states[i].steps_completed
        )
        delta_energy, acceptance_probability = _metropolis_criterion(
            cycle_energies[i], cycle_energies[j], temperature_kelvin
        )
        repex_rng, accept_rng = jax.random.split(repex_rng)
        accept = jax.random.bernoulli(accept_rng, acceptance_probability)

        replica_exchange_log.append({
            "episode_idx": episode_idx,
            "lambda_i": i,
            "lambda_j": j,
            "delta_energy": delta_energy,
            "acceptance_probability": acceptance_probability,
            "accept": accept,
        })

        # Note that due to if-else here this method can't be jitted.
        if accept:
            lambda_internal_states = _swap_lambdas(i, j, lambda_internal_states)
            logger.info(
                f"Swap ({i}, {j}): Accepted (prob = {acceptance_probability:.2f})"
            )
        else:
            logger.info(
                f"Swap ({i}, {j}): Rejected (prob = {acceptance_probability:.2f})"
            )

    return lambda_internal_states, replica_exchange_log, repex_rng


def perform_replica_exchange(
    latest_repex_energies: dict[int, Array],
    episode_idx: int,
    temperature_schedule: Callable[[int], float],
    lambda_internal_states: list[JaxMDSimulationState],
    lambda_values: Array,
    repex_rng: Array,
    replica_exchange_log: list[dict[str, Any]],
) -> tuple[list[JaxMDSimulationState], list[dict[str, Any]], Array]:
    """Run one round of Hamiltonian Replica Exchange.

    Computes cycle energies from the last-step energies, then applies
    Metropolis-criterion swaps between adjacent lambda windows.

    Returns updated (lambda_internal_states, replica_exchange_log, repex_rng).
    """
    cycle_energies = _compute_all_cycle_energies(
        latest_repex_energies, len(lambda_values)
    )
    return _run_repex_round(
        cycle_energies,
        episode_idx,
        temperature_schedule,
        lambda_internal_states,
        lambda_values,
        repex_rng,
        replica_exchange_log,
    )
