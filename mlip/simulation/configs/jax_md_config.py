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

from typing import Optional

from pydantic import Field, model_validator
from typing_extensions import Self

from mlip.simulation.configs.simulation_config import (
    SimulationConfig,
    TemperatureScheduleConfig,
)
from mlip.typing import PositiveFloat, PositiveInt

NUM_STEPS_PER_EP_THRESHOLD = 1_000
MIN_STEPS_PER_EP = 10
MAX_STEPS_PER_EP = 1_000


class JaxMDSimulationConfig(SimulationConfig):
    """Configuration for the JAX-MD-based simulations.

    Also includes the attributes of the parent class
    :class:`~mlip.simulation.configs.simulation_config.SimulationConfig`.

    The config is separated into three blocks: values that
    are used for both MD and minimization, and then the ones used exclusively for MD
    and for minimization, respectively.

    Attributes:
        num_episodes: Number of episodes to divide the simulation into. Each episode
                      runs in a fully jitted way, and the loggers are only
                      called after each episode. If not set, an appropriate value will
                      be attempted to be select but it is possible that it may have to
                      be manually set. For fewer than 1000 steps, ``num_episodes`` will
                      be set so that the number of steps per episode will be 10. For
                      more than 1000 steps, ``num_episodes`` will be set so that the
                      number of steps per episode will be 1000. Therefore, if
                      ``num_episodes`` is not set, it requires that ``num_steps`` be
                      divisible by 1000 if greater than 1000 otherwise divisible by 10.
        timestep_fs: The simulation timestep in femtoseconds. This is also used as the
                     initial timestep in the FIRE minimization algorithm. The default is
                     1.0.
        temperature_kelvin: The temperature in Kelvin, set to 300 by default. Must be
                            set to ``None`` for energy minimizations.
        temperature_schedule_config: The temperature schedule config to use for the
                                 simulation. Default is the constant schedule in
                                 which case ``temperature_kelvin`` will be applied.
    """

    num_episodes: PositiveInt | None = None
    timestep_fs: Optional[PositiveFloat] = 1.0

    # MD only
    temperature_kelvin: Optional[PositiveFloat] = 300.0
    temperature_schedule_config: TemperatureScheduleConfig = Field(
        default=TemperatureScheduleConfig(temperature=temperature_kelvin)
    )

    @model_validator(mode="after")
    def validate_num_episodes(self) -> Self:
        if not self.num_episodes:
            if self.num_steps < NUM_STEPS_PER_EP_THRESHOLD:
                self.num_episodes = max(self.num_steps // MIN_STEPS_PER_EP, 1)
            else:
                self.num_episodes = self.num_steps // MAX_STEPS_PER_EP
        if self.num_steps % self.num_episodes > 0:
            raise ValueError("Number of episodes must evenly divide total steps.")
        return self

    @model_validator(mode="after")
    def validate_snapshot_interval(self) -> Self:
        steps_per_episode = self.num_steps // self.num_episodes
        if steps_per_episode % self.snapshot_interval > 0:
            raise ValueError("Snapshot interval must evenly divide steps per episode.")
        return self
