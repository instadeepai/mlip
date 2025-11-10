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

from enum import Enum


class SimulationType(Enum):
    """Enum for the type of simulation.

    Attributes:
        MD: Molecular Dynamics.
        MINIMIZATION: Energy minimization.
    """

    MD = "md"
    MINIMIZATION = "minimization"


class SimulationBackend(Enum):
    """Enum for the simulation backend.

    Attributes:
        JAX_MD: Simulations with the JAX-MD backend.
        ASE: Simulations with the ASE backend.
    """

    JAX_MD = "jaxmd"
    ASE = "ase"


class TemperatureScheduleMethod(Enum):
    """Enum for the type of temperature schedule.

    Attributes:
        CONSTANT: Constant temperature schedule.
        LINEAR: Linear temperature schedule.
        TRIANGLE: Triangle temperature schedule.
    """

    CONSTANT = "constant"
    LINEAR = "linear"
    TRIANGLE = "triangle"
