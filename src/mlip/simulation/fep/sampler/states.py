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

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax_md.dataclasses import dataclass as jax_compatible_dataclass

from mlip.simulation.jax_md.states import EpisodeLog, SystemState
from mlip.simulation.state import SimulationState


@dataclass
class FEPSimulationState(SimulationState):
    """State for a single simulation engine in a multi-engine FEP simulation.

    Inherits all the fields from :class:`~mlip.simulation.state.SimulationState` and
    adds the ones below.

    Attributes:
        per_lambda_energies: The per-lambda energies along the simulation.
        replica_exchange_log: The log of the replica exchange attempts. This is
            populated only for a shared global state stored by the sampler.
        final_positions: The final positions of the simulation.
        final_velocities: The final velocities of the simulation.
        final_cell: The final cell of the simulation.
    """

    per_lambda_energies: np.ndarray | None = None
    replica_exchange_log: list[dict[str, float | None]] | None = None

    # State information required for restoring.
    final_positions: np.ndarray | None = None
    final_velocities: np.ndarray | None = None
    final_cell: np.ndarray | None = None


@jax_compatible_dataclass
class FEPSystemState(SystemState):
    """State for an FEP simulation.

    Inherits all the fields from
    :class:`~mlip.simulation.jax_md.states.SystemState` and adds the ones below.

    Attributes:
        lambda_values: The lambda values for the simulation. This is fixed throughout
            the simulation, even if the replicas are swapped during replica exchange.
    """

    lambda_values: jnp.ndarray | None = None


@jax_compatible_dataclass
class FEPEpisodeLog(EpisodeLog):
    """Holds the logging information for the currently processed episode.

    Inherits all the fields from
    :class:`~mlip.simulation.jax_md.states.EpisodeLog` and adds the one below.

    Attributes:
        per_lambda_energies: The per-lambda energies along the episode.
    """

    per_lambda_energies: jnp.ndarray
