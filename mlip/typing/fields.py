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

import e3nn_jax as e3nn
from pydantic import AfterValidator, Field
from typing_extensions import Annotated

PositiveFloat = Annotated[float, Field(gt=0)]

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


def _check_irreps(irreps: str) -> str:
    """Check that a string can be interpreted as `e3nn.Irreps`.

    This is useful to stop at the validation step.
    We can't return `e3nn.Irreps` for now as `model_dump` would break.
    """
    _ = e3nn.Irreps(irreps)
    return irreps


Irreps = Annotated[str, AfterValidator(_check_irreps)]
