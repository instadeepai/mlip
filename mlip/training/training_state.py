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
import time
from dataclasses import field
from typing import Optional

import chex
import jax
import optax
from flax.struct import dataclass as flax_dataclass

from mlip.training.ema import EMAParameterTransformation, EMAState
from mlip.typing import ModelParameters

logger = logging.getLogger("mlip")


@flax_dataclass
class TrainingState:
    """
    Represents the state of training.

    Attributes:
        params: Model parameters.
        optimizer_state: State of the optimizer.
        ema_state: Exponentially weighted average state.
        num_steps: The number of training steps taken.
        acc_steps: The number of gradient accumulation steps taken; resets to 0 after
                   each optimizer step.
        key: Pseudo-random number generator key.
        extras: Additional auxiliary information in form of a dictionary.
    """

    params: ModelParameters
    optimizer_state: optax.OptState
    ema_state: EMAState
    num_steps: jax.Array
    acc_steps: jax.Array
    key: chex.PRNGKey
    extras: Optional[dict] = field(default_factory=dict)


def _count_parameters(params: ModelParameters) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def init_training_state(
    initial_params: ModelParameters,
    random_key: chex.PRNGKey,
    optimizer: optax.GradientTransformation,
    ema_fun: EMAParameterTransformation,
) -> TrainingState:
    """Initializes the training state.

    Args:
        initial_params: The initial parameters.
        random_key: A jax-compatible random key.
        optimizer: The optimizer.
        ema_fun: The EMA parameter transformation function.

    Returns:
        The initialized training state.
    """
    key, gnn_key = jax.random.split(random_key, 2)
    cpu_device = jax.devices("cpu")[0]
    start_time = time.perf_counter()

    with jax.default_device(cpu_device):
        opt_state = optimizer.init(initial_params)
        ema_state = ema_fun.init(initial_params)

        training_state = TrainingState(
            params=initial_params,
            optimizer_state=opt_state,
            ema_state=ema_state,
            num_steps=0,
            acc_steps=0,
            key=key,
            extras={},
        )

        logger.debug(
            "Prepared training state on CPU in %.2f sec.",
            time.perf_counter() - start_time,
        )
        logger.info("Number of parameters: %s", _count_parameters(initial_params))
        logger.info(
            "Number of parameters in optimizer: %s", _count_parameters(opt_state)
        )

    return training_state
