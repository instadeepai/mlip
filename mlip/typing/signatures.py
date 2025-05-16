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

from typing import Callable, TypeAlias

from jax import Array
from jraph import GraphsTuple

from mlip.typing.prediction import Prediction

# ParameterDict from flax.linen
ModelParameters: TypeAlias = dict[str, dict[str, Array | dict]]

# ForceFieldPredictor.apply
ModelPredictorFun: TypeAlias = Callable[[ModelParameters, GraphsTuple], Prediction]

# LossFunction : (predictions, graph, epoch, eval_metrics) -> (loss, metrics)
LossFunction: TypeAlias = Callable[
    [Prediction, GraphsTuple, int, bool],
    tuple[Array, dict[str, Array]],
]
