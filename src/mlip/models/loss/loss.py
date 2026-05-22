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

import jax.numpy as jnp

from mlip.graph import Graph
from mlip.models.loss.eval_metrics import Metrics, compute_eval_metrics
from mlip.models.loss.loss_term import LossTerm, LossValue

EpochNumber: TypeAlias = int
Weight: TypeAlias = float
WeightSchedule: TypeAlias = Callable[[EpochNumber], Weight]


class Loss:
    """Loss combining multiple loss terms and schedules to form a new weighted loss."""

    def __init__(
        self,
        losses: list[LossTerm],
        schedules: list[WeightSchedule],
        extended_metrics: bool = False,
    ):
        """Constructor.

        Args:
            losses: A list of loss terms.
            schedules: A list of schedules. Has to be same order as losses.
            extended_metrics: Whether to include an extended list of metrics.
                              Defaults to `False`.

        """
        if len(losses) != len(schedules):
            raise ValueError("There must be as many schedules as there are losses.")

        self.losses = losses
        self.schedules = schedules
        self.extended_metrics = extended_metrics

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
        epoch: int,
        eval_metrics: bool = False,
    ) -> tuple[LossValue, Metrics]:
        """The call function that outputs the loss and metrics (auxiliary data).

        Args:
            pred_graph: Graph holding the predicted data.
            ref_graph: The reference graph holding the ground truth data.
            epoch: The epoch number passed to the schedules to get the correct weights
                   for each sub-loss.
            eval_metrics: Switch deciding whether to include additional
                          evaluation metrics to the returned dictionary.
                          Default is `False`.

        Returns:
            The loss and the auxiliary metrics dictionary.
        """

        # Sum terms. For schedules whose weight is compile-time zero (e.g.
        # `optax.piecewise_constant_schedule(0.0)`), XLA constant-folds the
        # `weight * loss(...)` product and its VJP away — so there's no
        # runtime cost for inactive losses in that common case.
        _loss_current_sum = 0.0
        for loss, schedule in zip(self.losses, self.schedules):
            weight = schedule(epoch)
            _loss_current_sum += weight * loss(pred_graph, ref_graph)

        # Average losses over graphs
        graph_mask = ref_graph.graph_mask()  # [n_graphs,]
        n_graphs = jnp.sum(graph_mask)
        total_loss = jnp.sum(jnp.where(graph_mask, _loss_current_sum, 0.0))
        avg_loss = total_loss / n_graphs

        metrics: Metrics = {"loss": avg_loss}

        # Optionally append loss weights, but as training metrics only
        if self.extended_metrics and not eval_metrics:
            metrics.update({
                f"{_l.property_name}_weight": _s(epoch)
                for _l, _s in zip(self.losses, self.schedules)
            })

        if eval_metrics:
            metrics |= self._compute_eval_metrics(pred_graph, ref_graph)

        return avg_loss, metrics

    def _compute_eval_metrics(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> Metrics:
        """Compute additional evaluation metrics.

        This is its own method of this class so it can easily be overridden if desired.
        """
        return compute_eval_metrics(
            pred_graph,
            ref_graph,
            self.extended_metrics,
        )
