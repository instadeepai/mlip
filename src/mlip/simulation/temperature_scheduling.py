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

from typing import Callable

import jax.numpy as jnp

from mlip.simulation.configs.ase_config import TemperatureScheduleConfig
from mlip.simulation.enums import TemperatureScheduleMethod


def get_temperature_schedule(
    temperature_schedule_config: TemperatureScheduleConfig, num_steps: int
) -> Callable[[int], float]:
    """Get the temperature schedule for the simulation. The schedule is a function that
    takes as input the current step and returns the temperature at that step.

    Arguments:
        temperature_schedule_config: The temperature schedule to use.
        num_steps: The total duration of the simulation
    """
    if temperature_schedule_config.method == TemperatureScheduleMethod.CONSTANT:
        return constant_schedule(temperature=temperature_schedule_config.temperature)
    elif temperature_schedule_config.method == TemperatureScheduleMethod.LINEAR:
        return linear_schedule(
            start_temperature=temperature_schedule_config.start_temperature,
            end_temperature=temperature_schedule_config.end_temperature,
            duration=num_steps,
        )
    elif temperature_schedule_config.method == TemperatureScheduleMethod.TRIANGLE:
        return triangle_schedule(
            max_temperature=temperature_schedule_config.max_temperature,
            min_temperature=temperature_schedule_config.min_temperature,
            heating_period=temperature_schedule_config.heating_period,
        )
    else:
        raise NotImplementedError(
            f"Temperature schedule type {temperature_schedule_config.method}"
            f" not implemented."
        )


def constant_schedule(temperature: float) -> Callable[[int], float]:
    """Returns a constant temperature schedule that returns the same temperature.

    Arguments:
        temperature: The temperature in Kelvin.
    """
    return lambda step: temperature


def linear_schedule(
    start_temperature: float, end_temperature: float, duration: int
) -> Callable[[int], float]:
    """Returns a linear temperature schedule that starts at ``start_temperature``
    and ends at ``end_temperature`` after ``duration`` steps. ``duration`` will
    automatically be set to the total number of steps in the simulation. Only the
    ``start_temperature`` and ``end_temperature`` need to be set.

    Arguments:
        start_temperature: The starting temperature in Kelvin.
        end_temperature: The ending temperature in Kelvin.
        duration: The duration for heating the system.
    """
    slope = (end_temperature - start_temperature) / duration
    return lambda step: (
        jnp.where(step > duration, end_temperature, slope * step + start_temperature)
    )


def triangle_schedule(
    max_temperature: float, min_temperature: float, heating_period: int
) -> Callable[[int], float]:
    """Returns a triangle wave with period ``heating_period`` that starts at
    ``min_temperature``, rises to ``max_temperature`` after ``heating_period / 2`` steps
    before returning to ``min_temperature`` after another ``heating_period / 2`` steps.

    Arguments:
        max_temperature: The maximum temperature in Kelvin.
        min_temperature: The minimum temperature in Kelvin.
        heating_period: The period for heating the system.
    """
    amplitude = (max_temperature - min_temperature) / 2
    return (
        lambda step: (4 * amplitude)
        / heating_period
        * jnp.abs(((step - heating_period / 2) % heating_period) - (heating_period / 2))
        + min_temperature
    )
