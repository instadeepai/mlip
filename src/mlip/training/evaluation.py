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
from typing import Callable, TypeAlias, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding

from mlip.data.helpers.type_aliases import GraphDatasetLike
from mlip.graph import Graph
from mlip.models.predictors import ForceFieldPredictor
from mlip.training.metrics_reweighting import reweight_metrics_by_number_of_graphs
from mlip.training.training_io_handler import LogCategory, TrainingIOHandler
from mlip.typing import LossFunction, ModelParameters
from mlip.utils.multihost import DATA_PARALLELISM_AXIS_NAME

EvaluationStepFun: TypeAlias = Callable[
    [ModelParameters, Graph, int], dict[str, np.ndarray]
]


def _evaluation_step(
    params: ModelParameters,
    graph: Graph,
    training_epoch: int,
    predictor: ForceFieldPredictor,
    eval_loss_fun: LossFunction,
    avg_n_graphs_per_batch: float,
    should_parallelize: bool = False,
) -> dict[str, np.ndarray]:

    def _single_eval(
        params: ModelParameters,
        single_input_graph: Graph,
        epoch: int,
    ) -> dict[str, jax.Array]:
        single_pred_graph = predictor.apply(params, single_input_graph)
        _, metrics = eval_loss_fun(single_pred_graph, single_input_graph, epoch)
        metrics = reweight_metrics_by_number_of_graphs(
            metrics, single_input_graph, avg_n_graphs_per_batch
        )
        return metrics

    if should_parallelize:
        _eval = jax.vmap(
            _single_eval,
            in_axes=(None, 0, None),
            spmd_axis_name=DATA_PARALLELISM_AXIS_NAME,
        )
        all_metrics = _eval(params, graph, training_epoch)
        return jax.tree.map(lambda m: jnp.mean(m, axis=0), all_metrics)

    return _single_eval(params, graph, training_epoch)


def make_evaluation_step(
    predictor: ForceFieldPredictor,
    eval_loss_fun: LossFunction,
    avg_n_graphs_per_batch: float,
    should_parallelize: bool = False,
    in_shardings: Union[NamedSharding, tuple[NamedSharding, ...]] | None = None,
    out_shardings: Union[NamedSharding, tuple[NamedSharding, ...]] | None = None,
) -> EvaluationStepFun:
    """Creates the evaluation step function.

    Args:
        predictor: The predictor to use.
        eval_loss_fun: The loss function for the evaluation.
        avg_n_graphs_per_batch: Average number of graphs per batch used for
                                reweighting of metrics.
        in_shardings: Optional in_shardings for `jax.jit`.
        out_shardings: Optional out_shardings for `jax.jit`.

    Returns:
        The evaluation step function.
    """
    evaluation_step = functools.partial(
        _evaluation_step,
        predictor=predictor,
        eval_loss_fun=eval_loss_fun,
        avg_n_graphs_per_batch=avg_n_graphs_per_batch,
        should_parallelize=should_parallelize,
    )
    jit_kwargs = {}
    if in_shardings is not None:
        jit_kwargs["in_shardings"] = in_shardings
    if out_shardings is not None:
        jit_kwargs["out_shardings"] = out_shardings
    return jax.jit(evaluation_step, **jit_kwargs)


def run_evaluation(
    evaluation_step: EvaluationStepFun,
    eval_dataset: GraphDatasetLike,
    params: ModelParameters,
    epoch_number: int,
    io_handler: TrainingIOHandler,
    is_test_set: bool = False,
    subset_name: str | None = None,
) -> float:
    """Runs a model evaluation on a given dataset.

    Args:
        evaluation_step: The evaluation step function.
        eval_dataset: The dataset on which to evaluate the model.
        params: The parameters to use for the evaluation.
        epoch_number: The current epoch number.
        io_handler: The IO handler class that handles the logging of the result.
        is_test_set: Whether the evaluation is done on the test set, i.e.,
                     not during a training run. By default, this is false.
        subset_name: Subset name for that dataset. If given, then the metric names
                     will be prefixed by it.

    Returns:
        The mean loss.
    """
    metrics = []
    for batch in eval_dataset:
        _metrics = evaluation_step(params, batch, epoch_number)
        metrics.append(_metrics)

    to_log = {}
    for metric_name in metrics[0].keys():
        metrics_values = np.array([m[metric_name] for m in metrics])
        if np.any(metrics_values != 0.0) and not any(
            jnp.isnan(val).any() for val in metrics_values
        ):
            to_log[metric_name] = np.mean(metrics_values[metrics_values != 0.0])

    mean_eval_loss = float(to_log.get("loss", jnp.nan))

    # Prefix by subset name; skips if None or empty string
    if subset_name:
        to_log = {f"{subset_name}_{metric}": value for metric, value in to_log.items()}

    if is_test_set:
        io_handler.log(LogCategory.TEST_METRICS, to_log, epoch_number)
    else:
        io_handler.log(LogCategory.EVAL_METRICS, to_log, epoch_number)

    return mean_eval_loss
