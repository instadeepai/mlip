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

import jax.numpy as jnp
from jax_md.dataclasses import dataclass as jax_compatible_dataclass
from jax_md.partition import NeighborList


@jax_compatible_dataclass
class SystemState:
    """Holds the state of the system that is simulated."""

    neighbors: NeighborList


@jax_compatible_dataclass
class EpisodeLog:
    """Holds the logging information for the currently processed episode."""

    positions: jnp.ndarray
    forces: jnp.ndarray
    velocities: jnp.ndarray
    temperature: jnp.ndarray
    kinetic_energy: jnp.ndarray


@jax_compatible_dataclass
class JaxMDSimulationState:
    """Holds the main information of the simulation in a jit-compatible way.

    The three components are the JAX-MD internal state object, the state of the system,
    the logging information for the current episode and the number of steps completed
    (across all episodes).
    """

    jax_md_state: jax_compatible_dataclass
    system_state: jax_compatible_dataclass
    episode_log: jax_compatible_dataclass
    steps_completed: int
