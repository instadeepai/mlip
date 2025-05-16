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

from pydantic import Field, model_validator
from typing_extensions import Self

from mlip.simulation.configs.simulation_config import (
    SimulationConfig,
    TemperatureScheduleConfig,
)
from mlip.typing import PositiveFloat, PositiveInt

NUM_STEPS_LOGGING_THRESHOLD = 1_000
MIN_LOG_FREQ = 10
MAX_LOG_FREQ = 1_000

logger = logging.getLogger("mlip")


class ASESimulationConfig(SimulationConfig):
    """Configuration for the ASE-based simulations.

    Also includes the attributes of the parent class
    :class:`~mlip.simulation.configs.simulation_config.SimulationConfig`.

    The config is separated into three blocks: values that
    are used for both MD and minimization, and then the ones used exclusively for MD
    and for minimization, respectively.

    Attributes:
        log_interval: The interval in ``num_steps`` at which the loggers
                      will be called. If not set, an appropriate value will
                      be attempted to be selected. For fewer than 1000 steps,
                      it will default to 10. For more than 1000 steps, it will
                      default to 1000.
        timestep_fs: The simulation timestep in femtoseconds. The default is
                     1.0.
        temperature_kelvin: The temperature in Kelvin, set to 300 by default. Must be
                            set to ``None`` for energy minimizations.
        friction: Friction coefficient for the simulation. Default is 0.1.
        temperature_schedule_config: The temperature schedule config to use for the
                                   simulation. Default is the constant schedule in
                                   which case ``temperature_kelvin`` will be applied.
        max_force_convergence_threshold: The convergence threshold for minimizations
                                         w.r.t. the sum of the force norms. See the
                                         ASE docs for more information. If not set,
                                         the ASE default will be used.
    """

    log_interval: PositiveInt | None = None

    # MD Only
    timestep_fs: PositiveFloat | None = 1.0
    temperature_kelvin: PositiveFloat | None = 300.0
    friction: PositiveFloat | None = 0.1

    # Temperature scheduling for MD
    temperature_schedule_config: TemperatureScheduleConfig = Field(
        default=TemperatureScheduleConfig(temperature=temperature_kelvin)
    )

    # Minimization only
    max_force_convergence_threshold: PositiveFloat | None = None

    @model_validator(mode="after")
    def validate_log_interval(self) -> Self:
        if not self.log_interval:
            if self.num_steps < NUM_STEPS_LOGGING_THRESHOLD:
                self.log_interval = MIN_LOG_FREQ
            else:
                self.log_interval = MAX_LOG_FREQ
        return self

    @model_validator(mode="after")
    def validate_snapshot_interval(self) -> Self:
        if self.num_steps % self.snapshot_interval > 0:
            logger.warning(
                "It is best for snapshot interval to divide"
                " the number of steps otherwise the final state of"
                " the simulation will not be saved.",
            )
        return self
