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
        TS_SEARCH: Transition state search.
    """

    MD = "md"
    MINIMIZATION = "minimization"
    TS_SEARCH = "ts_search"


class MDIntegrator(Enum):
    """Enum for the type of MD integrator.

    Attributes:
        NVT_LANGEVIN: Integrates using Langevin dynamics in the NVT ensemble.
        NPT_MC_LANGEVIN: Integrates using Langevin dynamics in the NPT
            ensemble using a Monte-Carlo Barostat.
        NVE_VELOCITY_VERLET: Integrates using velocity-Verlet dynamics in the
            NVE ensemble.
    """

    NVT_LANGEVIN = "nvt_langevin"
    NPT_MC_LANGEVIN = "npt_mc_langevin"
    NVE_VELOCITY_VERLET = "nve_velocity_verlet"

    @property
    def ensemble(self) -> str:
        """Returns the ensemble type for the integrator."""
        return self.value.split("_")[0]


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


class StructureOptimizationMethod(Enum):
    """Enum for the optimizer type for transition state search.

    Attributes:
        BFGS: Optimizer using the Broyden Fletcher Goldfarb Shanno algorithm.
        FIRE: Optimizer using the FIRE descend algorithm.
    """

    BFGS = "bfgs"
    FIRE = "fire"


class NEBMethod(Enum):
    """Enum for the NEB formulation passed to `ase.mep.NEB`.

    Attributes:
        ASENEB: Legacy ASE NEB implementation.
        IMPROVED_TANGENT: Henkelman & Jonsson, J. Chem. Phys. 113, 9978 (2000).
                          ASE's effective default.
        EB: Full spring-force formulation, Kolsbjerg et al., J. Chem. Phys. 145,
            094107 (2016).
        SPLINE: Spline interpolation, Makri et al., J. Chem. Phys. 150, 094109
                (2019). Supports preconditioning.
        STRING: String method, Makri et al. (2019).
    """

    ASENEB = "aseneb"
    IMPROVED_TANGENT = "improvedtangent"
    EB = "eb"
    SPLINE = "spline"
    STRING = "string"
