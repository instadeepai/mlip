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
class MetadynamicsSimulationState(SimulationState):
    """Public simulation state returned to the user after a metadynamics run.

    Extends `SimulationState` with per-step collective variable (CV) trajectories,
    bias potential values, and the full history of deposited Gaussian hills.

    Attributes:
        bias_cv_values: Values of the collective variables used by the bias potential
            along the trajectory. Has shape `(num_snapshots, num_cvs)`.
        bias_potential: Bias potential values along the trajectory in eV.
            Has shape `(num_snapshots,)`.
        gaussian_centers: Center of each deposited hill in CV space.
            Has shape `(num_hills, num_cvs)`.
        gaussian_heights: Height of each deposited hill in eV. Has shape `(n_hills,)`.
    """

    bias_cv_values: np.ndarray | None = None
    bias_potential: np.ndarray | None = None
    gaussian_centers: np.ndarray | None = None
    gaussian_heights: np.ndarray | None = None


@jax_compatible_dataclass
class MetadynamicsState:
    """Buffer of deposited Gaussian hills, carried in the simulation state.

    Attributes:
        gaussian_centers: Hill centers in CV space, shape `(max_gaussians, num_cvs)`.
            Entries beyond `num_gaussians` are zero-padded and ignored.
        gaussian_heights: Hill heights in eV, shape `(max_gaussians,)`.
            Entries beyond `num_gaussians` are zero-padded and ignored.
        num_gaussians: Number of hills deposited so far.
    """

    gaussian_centers: jnp.ndarray
    gaussian_heights: jnp.ndarray
    num_gaussians: int


@jax_compatible_dataclass
class MetadynamicsSystemState(SystemState):
    """Extends `SystemState` to carry the metadynamics hill buffer across steps.

    Attributes:
        metadynamics_state: Current hill buffer. `None` before initialisation.
    """

    metadynamics_state: MetadynamicsState | None = None


@jax_compatible_dataclass
class MetadynamicsEpisodeLog(EpisodeLog):
    """Extends `EpisodeLog` to record bias potential values at each step.

    Attributes:
        bias_cv_values: Values of the collective variables used by the bias potential
            at each step within the episode, shape `(num_steps, num_cvs)`.
        bias_potential: Bias potential energy at each step in eV, shape `(n_steps,)`.
    """

    bias_cv_values: jnp.ndarray
    bias_potential: jnp.ndarray
