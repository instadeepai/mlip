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

import functools
from typing import Callable, Optional, TypeAlias

import flax
import jax
import jraph
import numpy as np

from mlip.data.helpers.data_prefetching import PrefetchIterator
from mlip.data.helpers.graph_dataset import GraphDataset
from mlip.models.predictor import ForceFieldPredictor
from mlip.training.training_io_handler import LogCategory, TrainingIOHandler
from mlip.typing import LossFunction, ModelParameters

EvaluationStepFun: TypeAlias = Callable[
    [ModelParameters, jraph.GraphsTuple, int], dict[str, np.ndarray]
]


def _evaluation_step(
    params: ModelParameters,
    graph: jraph.GraphsTuple,
    training_epoch: int,
    predictor: ForceFieldPredictor,
    eval_loss_fun: LossFunction,
    should_parallelize: bool,
) -> dict[str, np.ndarray]:

    predictions = predictor.apply(params, graph)
    _, metrics = eval_loss_fun(predictions, graph, training_epoch)

    if should_parallelize:
        metrics = jax.lax.pmean(metrics, axis_name="device")
    return metrics


def make_evaluation_step(
    predictor: ForceFieldPredictor,
    eval_loss_fun: LossFunction,
    should_parallelize: bool = True,
) -> EvaluationStepFun:
    """Creates the evaluation step function.

    Args:
        predictor: The predictor to use.
        eval_loss_fun: The loss function for the evaluation.
        should_parallelize: Whether to apply data parallelization across
                            multiple devices.

    Returns:
        The evaluation step function.
    """
    evaluation_step = functools.partial(
        _evaluation_step,
        predictor=predictor,
        eval_loss_fun=eval_loss_fun,
        should_parallelize=should_parallelize,
    )

    if should_parallelize:
        return jax.pmap(
            evaluation_step, axis_name="device", static_broadcasted_argnums=2
        )
    return jax.jit(evaluation_step)


def run_evaluation(
    evaluation_step: EvaluationStepFun,
    eval_dataset: GraphDataset | PrefetchIterator,
    params: ModelParameters,
    epoch_number: int,
    io_handler: TrainingIOHandler,
    devices: Optional[list[jax.Device]] = None,
    is_test_set: bool = False,
) -> float:
    """Runs a model evaluation on a given dataset.

    Args:
        evaluation_step: The evaluation step function.
        eval_dataset: The dataset on which to evaluate the model.
        params: The parameters to use for the evaluation.
        epoch_number: The current epoch number.
        io_handler: The IO handler class that handles the logging of the result.
        devices: The jax devices. It can be None if not run in parallel (default).
        is_test_set: Whether the evaluation is done on the test set, i.e.,
                     not during a training run. By default, this is false.

    Returns:
        The mean loss.
    """
    should_unreplicate_batches = devices is None and isinstance(
        eval_dataset, PrefetchIterator
    )

    metrics = []
    for batch in eval_dataset:
        if should_unreplicate_batches:
            batch = flax.jax_utils.unreplicate(batch)
        _metrics = evaluation_step(params, batch, epoch_number)
        metrics.append(jax.device_get(_metrics))

    to_log = {}
    for metric_name in metrics[0].keys():
        metrics_values = [m[metric_name] for m in metrics]
        if not any(val is None for val in metrics_values):
            to_log[metric_name] = np.mean(metrics_values)

    mean_eval_loss = float(to_log["loss"])

    if is_test_set:
        io_handler.log(LogCategory.TEST_METRICS, to_log, epoch_number)
    else:
        io_handler.log(LogCategory.EVAL_METRICS, to_log, epoch_number)

    return mean_eval_loss
