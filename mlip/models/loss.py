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

import abc
from typing import Callable

import jax.numpy as jnp
import jraph
from jax import Array
from jraph import GraphsTuple

from mlip.models import loss_helpers
from mlip.typing import Prediction


class Loss(abc.ABC):
    """Very simple loss base class that defines the signature of the call function."""

    @abc.abstractmethod
    def __call__(
        self,
        prediction: Prediction,
        ref_graph: GraphsTuple,
        epoch: int,
        eval_metrics: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """The call function that outputs the loss and metrics (auxiliary data).

        Args:
            prediction: The force field predictor's outputs.
            ref_graph: The reference graph holding the ground truth data.
            epoch: The epoch number.
            eval_metrics: Switch deciding whether to include additional
                          evaluation metrics to the returned dictionary.
                          Default is `False`.

        Returns:
            The loss and the auxiliary metrics dictionary.
        """
        pass


class WeightedEFSLoss(Loss, abc.ABC):
    """Loss base class for scheduled average of energy, forces and stress errors."""

    def __init__(
        self,
        energy_weight_schedule: Callable[[int], float] = lambda _: 1.0,
        forces_weight_schedule: Callable[[int], float] = lambda _: 25.0,
        stress_weight_schedule: Callable[[int], float] = lambda _: 0.0,
        extended_metrics: bool = False,
    ) -> None:
        """
        Loss averaging energy, forces and stress errors with epoch-dependent weights.

        Args:
            energy_weight_schedule: The schedule function for the energy weight.
                                    Default is a constant weight of 1.
            forces_weight_schedule: The schedule function for the energy weight.
                                    Default is a constant weight of 25.
            stress_weight_schedule: The schedule function for the energy weight.
                                    Default is a constant weight of 0.
            extended_metrics: Whether to include an extended list of metrics.
                              Defaults to `False`.
        """
        self.energy_weight_schedule = energy_weight_schedule
        self.forces_weight_schedule = forces_weight_schedule
        self.stress_weight_schedule = stress_weight_schedule
        self.extended_metrics = extended_metrics

    @abc.abstractmethod
    def _energy_term(self, graph: GraphsTuple, energy: Array) -> Array:
        pass

    @abc.abstractmethod
    def _forces_term(self, graph: GraphsTuple, forces: Array) -> Array:
        pass

    @abc.abstractmethod
    def _stress_term(self, graph: GraphsTuple, stress: Array) -> Array:
        pass

    def __call__(
        self,
        prediction: Prediction,
        ref_graph: GraphsTuple,
        epoch: int,
        eval_metrics: bool = False,
    ) -> tuple[float, dict[str, float]]:
        # Get weights
        energy_weight = self.energy_weight_schedule(epoch)
        forces_weight = self.forces_weight_schedule(epoch)
        stress_weight = self.stress_weight_schedule(epoch)

        # Sum terms
        loss = 0.0
        loss += energy_weight * self._energy_term(ref_graph, prediction.energy)
        loss += forces_weight * self._forces_term(ref_graph, prediction.forces)
        if prediction.stress is not None:
            stress = prediction.stress
            loss += stress_weight * self._stress_term(ref_graph, stress)

        # Average losses over graphs
        graph_mask = jraph.get_graph_padding_mask(ref_graph)  # [n_graphs,]
        n_graphs = jnp.sum(graph_mask)
        total_loss = jnp.sum(jnp.where(graph_mask, loss, 0.0))
        avg_loss = total_loss / n_graphs

        metrics = {"loss": avg_loss}

        # Optionally append loss weights, but as training metrics only
        if self.extended_metrics and not eval_metrics:
            metrics.update(
                {
                    "energy_weight": self.energy_weight_schedule(epoch),
                    "forces_weight": self.forces_weight_schedule(epoch),
                    "stress_weight": self.stress_weight_schedule(epoch),
                }
            )

        if eval_metrics:
            metrics |= self._compute_eval_metrics(prediction, ref_graph)

        return avg_loss, metrics

    def _compute_eval_metrics(
        self,
        prediction: Prediction,
        ref_graph: GraphsTuple,
    ) -> dict[str, Array]:
        """Compute additional metrics."""
        return loss_helpers.compute_eval_metrics(
            prediction,
            ref_graph,
            self.extended_metrics,
        )


class MSELoss(WeightedEFSLoss):
    """Mean squared-error loss for scheduled average of energy, forces
    and stress errors.
    """

    def _energy_term(self, ref_graph: GraphsTuple, energy: Array) -> Array:
        return loss_helpers.mean_squared_error_energy(ref_graph, energy)

    def _forces_term(self, ref_graph: GraphsTuple, forces: Array) -> Array:
        return loss_helpers.mean_squared_error_forces(ref_graph, forces)

    def _stress_term(self, ref_graph: GraphsTuple, stress: Array) -> Array:
        return loss_helpers.mean_squared_error_stress(ref_graph, stress)


class HuberLoss(WeightedEFSLoss):
    """Huber loss for scheduled average of energy, forces and stress errors."""

    def _energy_term(self, ref_graph: GraphsTuple, energy: Array) -> Array:
        return loss_helpers.huber_loss_energy(ref_graph, energy)

    def _forces_term(self, ref_graph: GraphsTuple, forces: Array) -> Array:
        return loss_helpers.adaptive_huber_loss_forces(ref_graph, forces)

    def _stress_term(self, ref_graph: GraphsTuple, stress: Array) -> Array:
        return loss_helpers.huber_loss_stress(ref_graph, stress)
