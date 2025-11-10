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

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from mlip.typing import ModelParameters


class EMAParameterTransformation(NamedTuple):
    """Container for parameter transformation functions.

    Attributes:
        init: Function to initialize state.
        update: Function to update state.
    """

    init: Callable
    update: Callable


class EMAState(NamedTuple):
    """Container for Exponentially Weighted Average state.

    Attributes:
        params_ema: Exponentially Weighted Average of the parameters.
        step: The current step.
    """

    params_ema: ModelParameters
    step: int


def exponentially_moving_average(decay: float = 0.99) -> EMAParameterTransformation:
    """Creates an exponentially moving average (EMA) transformation.

    Args:
        decay: The decay factor for the EMA. Defaults to 0.99.

    Returns:
        A named tuple containing init and update functions for EMA.
    """

    def init_fn(params: ModelParameters) -> EMAState:
        """Initializes the EMAState with zeros and step 0.

        Args:
            params: The initial parameters.

        Returns:
            he initialized state.
        """
        return EMAState(params_ema=jax.tree.map(jnp.zeros_like, params), step=0)

    def update_fn(state: EMAState, params: ModelParameters) -> EMAState:
        """Updates the EMAState by computing the exponentially moving average
        of the parameters.

        Args:
            state: The current state.
            params: The current parameters.

        Returns:
            EMAState: The updated state.
        """
        params_ema = jax.tree.map(
            lambda old, new: decay * old + (1 - decay) * new, state.params_ema, params
        )

        state = EMAState(params_ema=params_ema, step=state.step + 1)

        return state

    return EMAParameterTransformation(init=init_fn, update=update_fn)


def get_debiased_params(state: EMAState, decay: float) -> ModelParameters:
    """Gets the debiased parameters from the EMAState.

    Args:
        state: The current state.
        decay: The decay factor for the EMA.

    Returns:
        The debiased parameters.
    """
    denom = 1 - decay**state.step
    debiased_params = jax.tree.map(lambda x: x / denom, state.params_ema)
    return debiased_params
