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
from typing import Optional

import numpy as np


@dataclass
class SimulationState:
    """Holds all the information of the current state of a simulation.

    This object is populated during a simulation and is processed by the
    loggers of a simulation.

    When running batched simulations, some of the fields may be of type
    `list[numpy.ndarray]` instead of just `numpy.ndarray`.

    Attributes:
        atomic_numbers: The atomic numbers of the system.
        positions: The positions along the trajectory. Has shape M x N x 3, with M
                   being the number of steps divided by the logging interval, and N
                   being the number of atoms of the system. The unit is Angstrom.
        forces: The forces along the trajectory with the same shape as the positions.
                The unit is eV / Angstrom.
        velocities: The velocities along the trajectory with the same shape as the
                    positions, in units of :math:`\\sqrt{eV/u}`.
        temperature: The temperatures along the trajectory in Kelvin.
        kinetic_energy: The total kinetic energy along the trajectory in eV.
        step: The current number of steps performed.
        compute_time_seconds: The compute time in seconds used so far for the run
                              (not including logging times).
    """

    atomic_numbers: Optional[np.ndarray | list[np.ndarray]] = None
    positions: Optional[np.ndarray | list[np.ndarray]] = None
    forces: Optional[np.ndarray | list[np.ndarray]] = None
    velocities: Optional[np.ndarray | list[np.ndarray]] = None
    temperature: Optional[np.ndarray | list[np.ndarray]] = None
    kinetic_energy: Optional[np.ndarray | list[np.ndarray]] = None
    step: int = 0
    compute_time_seconds: float = 0.0
