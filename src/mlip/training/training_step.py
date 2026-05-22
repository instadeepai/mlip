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
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import optax
from jax import Array
from jax.sharding import NamedSharding
from jraph import GraphsTuple

from mlip.models.predictor import ForceFieldPredictor
from mlip.training.ema import EMAParameterTransformation
from mlip.training.metrics_reweighting import reweight_metrics_by_number_of_graphs
from mlip.training.training_state import TrainingState
from mlip.typing import LossFunction, ModelParameters
from mlip.utils.multihost import DATA_PARALLELISM_AXIS_NAME


def _training_step(
    training_state: TrainingState,
    graph: GraphsTuple,
    epoch_number: int,
    model_loss_fun: Callable[[ModelParameters, GraphsTuple, int], Array],
    optimizer: optax.GradientTransformation,
    ema_fun: EMAParameterTransformation,
    avg_n_graphs_per_batch: float,
    num_gradient_accumulation_steps: Optional[int],
) -> tuple[TrainingState, dict]:
    # Fetch params and optimizer state from training state.
    params = training_state.params
    optimizer_state = training_state.optimizer_state
    ema_state = training_state.ema_state
    num_steps = training_state.num_steps
    acc_steps = training_state.acc_steps

    def single_loss_with_reweight(params, single_graph, epoch):
        loss, metrics = model_loss_fun(params, single_graph, epoch)
        metrics = reweight_metrics_by_number_of_graphs(
            metrics, single_graph, avg_n_graphs_per_batch
        )
        return loss, metrics

    def mean_loss_fn(params, stacked_graph, epoch):
        losses, metrics = jax.vmap(
            single_loss_with_reweight,
            in_axes=(None, 0, None),
            spmd_axis_name=DATA_PARALLELISM_AXIS_NAME,
        )(params, stacked_graph, epoch)

        return jnp.mean(losses), jax.tree.map(lambda m: jnp.mean(m, axis=0), metrics)

    grad_fn = jax.grad(mean_loss_fn, argnums=0, has_aux=True)
    grads, metrics = grad_fn(params, graph, epoch_number)

    # Gradient step on params.
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params=params)
    params = optax.apply_updates(params, updates)

    # Add batch-level metrics to the dictionary.
    metrics["gradient_norm"] = optax.global_norm(grads)
    metrics["param_update_norm"] = optax.global_norm(updates)

    # Update per-step variables.
    acc_steps = (acc_steps + 1) % num_gradient_accumulation_steps
    ema_state = jax.lax.cond(
        acc_steps == 0, lambda x: ema_fun.update(x, params), lambda x: x, ema_state
    )
    num_steps = jax.lax.cond(acc_steps == 0, lambda x: x + 1, lambda x: x, num_steps)

    # Prepare new training state.
    training_state = TrainingState(
        params=params,
        optimizer_state=optimizer_state,
        ema_state=ema_state,
        num_steps=num_steps,
        acc_steps=acc_steps,
        extras=training_state.extras,
    )

    return training_state, metrics


def make_train_step(
    predictor: ForceFieldPredictor,
    loss_fun: LossFunction,
    optimizer: optax.GradientTransformation,
    ema_fun: EMAParameterTransformation,
    avg_n_graphs_per_batch: float,
    num_gradient_accumulation_steps: Optional[int] = 1,
    in_shardings: Optional[Union[NamedSharding, tuple[NamedSharding, ...]]] = None,
    out_shardings: Optional[Union[NamedSharding, tuple[NamedSharding, ...]]] = None,
) -> Callable:
    """Create a training step function to optimize model params using gradients.

    Args:
        predictor: The force field predictor, instance of `nn.Module`.
        loss_fun: A function that computes the loss from predictions, a reference
                  labelled graph, and the epoch number.
        optimizer: An optimizer for updating model params based on computed gradients.
        ema_fun: A function for updating the exponential moving average (EMA) of
                 the model params.
        avg_n_graphs_per_batch: Average number of graphs per batch used for
                                reweighting of metrics.
        num_gradient_accumulation_steps: The number of gradient accumulation
                                         steps before a parameter update is performed.
                                         Defaults to 1, implying immediate updates.
        in_shardings: Optional in_shardings for `jax.jit`.
        out_shardings: Optional out_shardings for `jax.jit`.

    Returns:
        A function that takes the current training state and a batch of data as
        input, and returns the updated training state along with training metrics.
    """

    def model_loss(
        params: ModelParameters, ref_graph: GraphsTuple, epoch: int
    ) -> tuple[Array, dict[str, Array]]:
        predictions = predictor.apply(params, ref_graph)
        return loss_fun(predictions, ref_graph, epoch)

    training_step = functools.partial(
        _training_step,
        model_loss_fun=model_loss,
        optimizer=optimizer,
        ema_fun=ema_fun,
        avg_n_graphs_per_batch=avg_n_graphs_per_batch,
        num_gradient_accumulation_steps=num_gradient_accumulation_steps,
    )
    jit_kwargs = {}
    if in_shardings is not None:
        jit_kwargs["in_shardings"] = in_shardings
    if out_shardings is not None:
        jit_kwargs["out_shardings"] = out_shardings
    return jax.jit(training_step, **jit_kwargs)
