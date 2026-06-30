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
import dataclasses

import pydantic
from pydantic import Field
from typing_extensions import Annotated

from mlip.simulation.enums import (
    MDIntegrator,
    SimulationType,
    TemperatureScheduleMethod,
)
from mlip.typing import PositiveFloat, PositiveInt

DEFAULT_EDGE_CAPACITY_MULT = 1.25
DEFAULT_SIMULATION_RANDOM_SEED = 42


ThreeDimensionalListWithPositiveFloats = Annotated[
    list[PositiveFloat], pydantic.Field(min_length=3, max_length=3)
]
FloatLargerThanOrEqualToOne = Annotated[float, pydantic.Field(ge=1)]


class SimulationConfig(pydantic.BaseModel):
    """The base configuration that all simulations share.

    It only contains fields that are independent of backend and simulation type.

    Attributes:
        simulation_type: The type of simulation to run, either MD or minimization.
            Defaults to MD.
        md_integrator: The type of MD integrator to use if running MD simulations.
            Ignored if `simulation_type` is not MD. Defaults to NVT-Langevin.
        num_steps: The number of total steps to run. For energy minimizations,
            this is the maximum number of steps if no convergence reached
            earlier.
        snapshot_interval: The interval (in steps) between snapshots of the simulation
            state. This means information about every N-th
            snapshot is stored in the simulation state available to the
            loggers (N being the snapshot interval). Defaults to 1.
        box: The optional simulation box. If `None`, no periodic boundary conditions
            (PBCs) are applied. It can be set to either a float or a list three floats,
            to enforce orthorhombic PBCs. Note that the `ASESimulationEngine` supports
            generic PBCs by reading the `cell` attribute of the `ase.Atoms` to
            simulate, in which case the `box` parameter should not be used.
        edge_capacity_multiplier: Factor to multiply the number of edges
            (and long-range edges, when present) by to obtain the edge capacity
            including padding. Defaults to 1.25.
        set_none_charge_to_zero: Whether to set a missing total charge on the
            simulated system to zero. Default is `True`.
        random_seed: Random seed to use throughout the simulation.
        independent_seeds_batched: When `True`, in batched simulations,  assign
            each sub-simulation an independent random key (derived by splitting the
            base key produced by the seed), resulting in decorrelated trajectories.
            Has no effect for non-batched simulations. Defaults to `True`.
    """

    simulation_type: SimulationType = SimulationType.MD
    md_integrator: MDIntegrator = MDIntegrator.NVT_LANGEVIN
    num_steps: PositiveInt
    snapshot_interval: PositiveInt = 1
    box: PositiveFloat | ThreeDimensionalListWithPositiveFloats | None = None
    edge_capacity_multiplier: FloatLargerThanOrEqualToOne = DEFAULT_EDGE_CAPACITY_MULT
    set_none_charge_to_zero: bool = True
    random_seed: int = DEFAULT_SIMULATION_RANDOM_SEED
    independent_seeds_batched: bool = True


class TemperatureScheduleConfig(pydantic.BaseModel):
    """The base configuration containing all the possible parameters for the
    temperature schedules.

    Attributes:
        method: The type of temperature schedule to use. Default is constant.
        temperature: The temperature to use for the constant schedule in Kelvin.
        start_temperature: The starting temperature in Kelvin.
            Used for the linear schedule.
        end_temperature: The ending temperature in Kelvin.
            Used for the linear schedule.
        max_temperature: The maximum temperature in Kelvin.
            Used for the triangle schedule.
        min_temperature: The minimum temperature in Kelvin.
            Used for the triangle schedule.
        heating_period: The period for heating the system.
            Measured in number of simulation steps. Used for the triangle schedule.

    """

    method: TemperatureScheduleMethod = Field(
        default=TemperatureScheduleMethod.CONSTANT
    )

    # Constant schedule
    temperature: PositiveFloat | None = None

    # Linear schedule
    start_temperature: PositiveFloat | None = None
    end_temperature: PositiveFloat | None = None

    # Triangle schedule
    max_temperature: PositiveFloat | None = None
    min_temperature: PositiveFloat | None = None
    heating_period: PositiveInt | None = None


@dataclasses.dataclass
class SimulationLogOutputs:
    """Dataclass that specifies which of the fields in the simulation state get
    updated/logged during the simulation.

    By default, all of them will. A user can hence disable specific ones to save
    memory, e.g., if one is only interested in saving positions.

    Attributes:
        positions: Whether to save positions.
        velocities: Whether to save velocities.
        forces: Whether to save forces.
        kinetic_energy: Whether to save kinetic energies.
        temperature: Whether to save temperatures.
        cell: Whether to save cell vectors.
        potential_energy: Whether to save potential energies.
        partial_charges: Whether to save partial charges if predicted by the model.
    """

    positions: bool = True  # [T, N, 3]
    velocities: bool = True  # [T, N, 3]
    forces: bool = True  # [T, N, 3]
    kinetic_energy: bool = True  # [T]
    temperature: bool = True  # [T]
    cell: bool = True  # [T, 3, 3]
    potential_energy: bool = True  # [T]
    partial_charges: bool = True  # [T, N]
