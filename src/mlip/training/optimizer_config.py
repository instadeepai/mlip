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

import pydantic
from typing_extensions import Annotated

PositiveFloat = Annotated[float, pydantic.Field(gt=0.0)]
NonNegativeFloat = Annotated[float, pydantic.Field(ge=0.0)]
PositiveInt = Annotated[int, pydantic.Field(gt=0)]
NonNegativeInt = Annotated[int, pydantic.Field(ge=0)]


class OptimizerConfig(pydantic.BaseModel):
    """Pydantic config holding all settings that are relevant for the optimizer.

    Attributes:
        apply_weight_decay_mask: Whether to apply a weight decay mask. If set to
                                 ``False``, a weight decay is applied to all parameters.
                                 If set to ``True`` (default), only the parameters of
                                 model blocks "linear_down" and "SymmetricContraction"
                                 are assigned a weight decay. These blocks only exist
                                 for MACE models, and it is recommended for MACE to
                                 set this setting to ``True``. If it is set to
                                 ``True`` but neither of these blocks exist in the
                                 model (like for ViSNet or NequIP),
                                 we apply weight decay to all parameters.
        weight_decay: The weight decay with a default of zero.
        grad_norm: Gradient norm used for gradient clipping.
        num_gradient_accumulation_steps: Number of gradient steps to accumulate before
                                         taking an optimizer step. Default is 1.
        init_learning_rate: Initial learning rate (default is 0.01).
        peak_learning_rate: Peak learning rate (default is 0.01).
        final_learning_rate: Final learning rate (default is 0.01).
        warmup_steps: Number of optimizer warm-up steps (default is 4000).
                      Check optax's ``linear_schedule()`` function for more info.
        transition_steps: Number of optimizer transition steps (default is 360000).
                          Check optax's ``linear_schedule()`` function for more info.
    """

    apply_weight_decay_mask: bool = True

    weight_decay: NonNegativeFloat = 0.0
    grad_norm: NonNegativeFloat = 500
    num_gradient_accumulation_steps: PositiveInt = 1

    init_learning_rate: PositiveFloat = 0.01
    peak_learning_rate: PositiveFloat = 0.01
    final_learning_rate: PositiveFloat = 0.01
    warmup_steps: NonNegativeInt = 4000
    transition_steps: NonNegativeInt = 360000
