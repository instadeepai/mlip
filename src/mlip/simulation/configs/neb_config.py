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

from pydantic import field_validator

from mlip.simulation.configs.ase_config import ASESimulationConfig
from mlip.simulation.enums import NEBMethod, SimulationType, StructureOptimizationMethod


class NEBSimulationConfig(ASESimulationConfig):
    """Configuration for the NEB simulations.

    Also includes all the attributes of the
    :class:`~mlip.simulation.configs.ase_config.ASESimulationConfig`.

    Attributes:
        simulation_type: The type of simulation to run. Only ts_search is supported.
        optimizer: Optimizer algorithm for the NEB force optimization.
        num_images: Number of images along the elastic band.
        neb_k: Force constant between the images.
        max_force_convergence_threshold: Max. atom force threshold for NEB simulation.
        continue_from_previous_run: Whether the NEB simulation restarts from a
                                    previous run. This will skip interpolation between
                                    images.
        climb: Whether to apply the climbing image method.
        neb_method: Tangent / spring formulation passed to `ase.mep.NEB`.
        bfgs_alpha: Alpha parameter for the BFGS optimizer.
        bfgs_maxstep: Maxstep parameter for the BFGS optimizer.
        fire_timestep: Timestep parameter for the FIRE optimizer. Note that this
                       timestep is in ASE units.
    """

    simulation_type: SimulationType = SimulationType.TS_SEARCH
    optimizer: StructureOptimizationMethod = StructureOptimizationMethod.BFGS
    num_images: int = 7
    neb_k: float | None = 10.0
    max_force_convergence_threshold: float | None = 0.1
    continue_from_previous_run: bool = False
    climb: bool = False
    neb_method: NEBMethod = NEBMethod.IMPROVED_TANGENT
    bfgs_alpha: float = 70.0
    bfgs_maxstep: float = 0.03

    fire_timestep: float = 0.1

    @field_validator("simulation_type")
    @classmethod
    def validate_ts_search_simulation_type(cls, v: SimulationType) -> SimulationType:
        if v != SimulationType.TS_SEARCH:
            raise ValueError("The NEB simulation type must be 'TS_SEARCH'.")
        return v
