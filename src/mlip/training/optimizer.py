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

import optax

from mlip.training.optimizer_config import OptimizerConfig
from mlip.utils.dict_flatten import flatten_dict, unflatten_dict


# This following function is needed for MACE
def _weight_decay_mask(params: optax.Params) -> optax.Params:
    params = flatten_dict(params)

    linear_down_exists = any(any(("linear_down" in ki) for ki in k) for k in params)
    symmetric_contraction_exists = any(
        any(("SymmetricContraction" in ki) for ki in k) for k in params
    )

    # Only apply the mask if one of them exists
    if linear_down_exists or symmetric_contraction_exists:
        mask = {
            k: any(("linear_down" in ki) or ("SymmetricContraction" in ki) for ki in k)
            for k in params
        }
        return unflatten_dict(mask)

    return unflatten_dict(dict.fromkeys(params, True))


def get_mlip_optimizer_chain_with_flexible_base_optimizer(
    base_optimizer_factory_fun: Callable[[float], optax.GradientTransformation],
    config: OptimizerConfig,
) -> optax.GradientTransformation:
    """Initializes an optimizer (based on optax) as a chain that is derived from
    a base optimizer class, e.g., optax.amsgrad.

    The initialization happens from a base optimizer function, for example,
    `optax.adam`. This base optimizer function must be able to take in the learning rate
    as a single parameter.

    The return value of this is a full optimizer pipeline consisting of
    gradient clipping, warm-up, etc.

    Args:
        base_optimizer_factory_fun: The base optimizer function which must be able to
                                    take in the learning rate as a single parameter.
        config: The optimizer pydantic config.

    Returns:
        The full optimizer pipeline constructed based on the provided
        base optimizer function.
    """
    if config.apply_weight_decay_mask:
        weight_decay_transform = optax.add_decayed_weights(
            config.weight_decay, _weight_decay_mask
        )
    else:
        weight_decay_transform = optax.add_decayed_weights(config.weight_decay)

    return optax.inject_hyperparams(
        lambda lr: optax.MultiSteps(
            optax.chain(
                weight_decay_transform,
                optax.clip_by_global_norm(config.grad_norm),
                base_optimizer_factory_fun(lr),
            ),
            every_k_schedule=config.num_gradient_accumulation_steps,
        )
    )(
        lr=optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=config.init_learning_rate,
                    end_value=config.peak_learning_rate,
                    transition_steps=config.warmup_steps,
                ),
                optax.linear_schedule(
                    init_value=config.peak_learning_rate,
                    end_value=config.final_learning_rate,
                    transition_steps=config.transition_steps,
                ),
            ],
            boundaries=[config.warmup_steps],
        ),
    )


def get_default_mlip_optimizer(
    config: OptimizerConfig | None = None,
) -> optax.GradientTransformation:
    """Get a default optimizer for training MLIP models.

    This is a specialized optimizer setup that originated in the MACE torch
    repo: https://github.com/ACEsuit/mace. It is customizable to an extent via the
    `OptimizerConfig`.

    This optimizer is based on the `optax.amsgrad` base optimizer and adds a weight
    decay transform to it optionally (only should be done for MACE), a gradient clipping
    step, and a scheduling of the learning rate with possible warm up period.
    Furthermore, we allow for gradient accumulation if requested.

    The learning rate schedule works as follows:
    First there is a period of warmup steps where the learning rate linearly increases
    from the "initial" to the "peak" learning rate. After that,
    we have a linearly increasing learning rate from "peak" to "final" learning rate.

    See the optimizer config's documentation for how
    to customize this default MLIP optimizer.

    This function internally uses
    :meth:`~mlip.training.optimizer.get_mlip_optimizer_chain_with_flexible_base_optimizer`
    which can also be used directly to build an analogous optimizer chain but with a
    different base optimizer.

    Args:
        config: The optimizer config. Default is ``None`` which leads to the
                pure default config being used.

    Returns:
        The default optimizer.
    """
    if config is None:
        config = OptimizerConfig()

    default_amsgrad_kwargs = {"b1": 0.9, "b2": 0.999, "eps": 1e-8, "eps_root": 0.0}

    def base_optimizer_factory_fun(
        learning_rate: float,
    ) -> optax.GradientTransformation:
        return optax.amsgrad(learning_rate=learning_rate, **default_amsgrad_kwargs)

    return get_mlip_optimizer_chain_with_flexible_base_optimizer(
        base_optimizer_factory_fun, config
    )
