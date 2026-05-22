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
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import optax

from mlip.graph import Graph
from mlip.models.loss.loss_helpers import (
    HUBER_LOSS_DEFAULT_DELTA,
    compute_adaptive_huber_loss_forces,
    sum_nodes_of_the_same_graph,
)
from mlip.utils.safe_norm import safe_divide

# Per-graph loss array of shape [n_graphs,]. Losses may also return the literal
# `0` in the None-guard branch; it broadcasts fine against the per-graph sums
# in `Loss`.
LossValue: TypeAlias = jax.Array


class LossTerm(abc.ABC):
    """Abstract loss base class that defines the signature of the call function.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Needs to be overridden by subclasses. Mostly meant for use in
                       metrics logging.
    """

    # Should be updated by subclass
    property_name: str | None = None

    @abc.abstractmethod
    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        """The call function that outputs the loss.

        Args:
            pred_graph: Graph holding the predicted data.
            ref_graph: The reference graph holding the ground truth data.

        Returns:
            The loss.
        """
        pass

    def __init_subclass__(cls, **kwargs: Any):
        """This enforces that child classes will need to override
        the `property_name` attribute.
        """
        super().__init_subclass__(**kwargs)
        if cls.property_name is None:
            raise NotImplementedError(
                f"{cls.__name__} must override the `property_name` attribute."
            )


class MSEEnergyLoss(LossTerm):
    """Loss for the mean-squared error of the energy.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "energy".
    """

    property_name: str | None = "energy"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        energy_ref = ref_graph.globals.energy  # [n_graphs, ]
        energy_pred = pred_graph.globals.energy  # [n_graphs, ]

        if energy_ref is None or energy_pred is None:
            # We null out the loss if the reference energy is not provided
            return 0

        # Per-graph validity: NaN in ref marks samples from datasets that did
        # not provide this field. `safe_ref` keeps NaN out of the arithmetic
        # so gradients don't get poisoned (0 * NaN = NaN in JAX).
        valid = ~jnp.isnan(energy_ref)
        safe_ref = jnp.where(valid, energy_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * jnp.square(
            safe_divide(safe_ref - energy_pred, ref_graph.n_node)
        )
        return jnp.where(valid, loss_per_graph, 0.0)  # [n_graphs, ]


class HuberEnergyLoss(LossTerm):
    """Loss for the Huber loss of the energy.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "energy".
    """

    property_name: str | None = "energy"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        energy_ref = ref_graph.globals.energy  # [n_graphs, ]
        energy_pred = pred_graph.globals.energy  # [n_graphs, ]

        if energy_ref is None or energy_pred is None:
            # We null out the loss if the reference energy is not provided
            return 0

        valid = ~jnp.isnan(energy_ref)
        safe_ref = jnp.where(valid, energy_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * optax.losses.huber_loss(
            safe_divide(energy_pred, ref_graph.n_node),
            safe_divide(safe_ref, ref_graph.n_node),
            delta=HUBER_LOSS_DEFAULT_DELTA,
        )
        return jnp.where(valid, loss_per_graph, 0.0)  # [n_graphs, ]


class MSEForcesLoss(LossTerm):
    """Loss for the mean-squared error of the forces.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "forces".
    """

    property_name: str | None = "forces"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        forces_ref = ref_graph.nodes.forces  # [n_nodes, 3]
        forces_pred = pred_graph.nodes.forces  # [n_nodes, 3]

        if forces_ref is None or forces_pred is None:
            # We null out the loss if the reference forces are not provided
            return 0

        # NaN-valued ref forces come from datasets that did not provide this
        # field. Mask both the arithmetic (to keep NaN out of gradients) and
        # the final per-graph loss.
        node_valid = ~jnp.isnan(forces_ref).any(axis=1)  # [n_nodes]
        safe_ref = jnp.where(node_valid[:, None], forces_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * safe_divide(
            sum_nodes_of_the_same_graph(
                ref_graph, jnp.mean(jnp.square(safe_ref - forces_pred), axis=1)
            ),
            ref_graph.n_node,
        )
        graph_valid = (
            sum_nodes_of_the_same_graph(ref_graph, node_valid.astype(jnp.float32)) > 0
        )
        return jnp.where(graph_valid, loss_per_graph, 0.0)  # [n_graphs, ]


class HuberForcesLoss(LossTerm):
    """Loss for the adaptive Huber loss of the forces.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "forces".
    """

    property_name: str | None = "forces"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        forces_ref = ref_graph.nodes.forces  # [n_nodes, 3]
        forces_pred = pred_graph.nodes.forces  # [n_nodes, 3]

        if forces_ref is None or forces_pred is None:
            # We null out the loss if the reference forces are not provided
            return 0

        node_valid = ~jnp.isnan(forces_ref).any(axis=1)  # [n_nodes]
        safe_ref = jnp.where(node_valid[:, None], forces_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * safe_divide(
            sum_nodes_of_the_same_graph(
                ref_graph,
                jnp.mean(
                    compute_adaptive_huber_loss_forces(forces_pred, safe_ref),
                    axis=1,
                ),
            ),
            ref_graph.n_node,
        )
        graph_valid = (
            sum_nodes_of_the_same_graph(ref_graph, node_valid.astype(jnp.float32)) > 0
        )
        return jnp.where(graph_valid, loss_per_graph, 0.0)  # [n_graphs, ]


class MSEStressLoss(LossTerm):
    """Loss for the mean-squared error of the stress.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "stress".
    """

    property_name: str | None = "stress"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        stress_ref = ref_graph.globals.stress  # [n_graphs, 3, 3]
        stress_pred = pred_graph.globals.stress  # [n_graphs, 3, 3]

        if stress_ref is None or stress_pred is None:
            # We null out the loss if the reference stress is not provided
            return 0

        valid = ~jnp.isnan(stress_ref).any(axis=(1, 2))  # [n_graphs]
        safe_ref = jnp.where(valid[:, None, None], stress_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * jnp.mean(
            jnp.square(safe_ref - stress_pred), axis=(1, 2)
        )
        return jnp.where(valid, loss_per_graph, 0.0)  # [n_graphs, ]


class HuberStressLoss(LossTerm):
    """Loss for the Huber loss of the stress.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "stress".
    """

    property_name: str | None = "stress"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        stress_ref = ref_graph.globals.stress  # [n_graphs, 3, 3]
        stress_pred = pred_graph.globals.stress  # [n_graphs, 3, 3]

        if stress_ref is None or stress_pred is None:
            # We null out the loss if the reference stress is not provided
            return 0

        valid = ~jnp.isnan(stress_ref).any(axis=(1, 2))  # [n_graphs]
        safe_ref = jnp.where(valid[:, None, None], stress_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * jnp.mean(
            optax.losses.huber_loss(
                stress_pred,
                safe_ref,
                delta=HUBER_LOSS_DEFAULT_DELTA,
            ),
            axis=(1, 2),
        )
        return jnp.where(valid, loss_per_graph, 0.0)  # [n_graphs, ]


class MSEHessianLoss(LossTerm):
    """Loss for the mean-squared error of the Hessian.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "Hessian".
    """

    property_name: str | None = "Hessian"

    def __call__(self, pred_graph: Graph, ref_graph: Graph) -> LossValue:
        hessian_ref = ref_graph.nodes.hessian  # [n, R, 3]
        hessian_pred = pred_graph.nodes.hessian  # [n, R, 3]
        if hessian_ref is None or hessian_pred is None:
            # We null out the loss if the reference Hessian is not provided
            return 0

        return ref_graph.globals.weight * safe_divide(
            sum_nodes_of_the_same_graph(
                ref_graph,
                jnp.append(  # append 0 to match length (n_node + 1)
                    jnp.mean(jnp.square(hessian_ref - hessian_pred), axis=(0, -1)), 0
                ),
            ),
            ref_graph.n_node,
        )  # [n_graphs, ]


class HuberHessianLoss(LossTerm):
    """Loss for the Huber loss of the Hessian.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "Hessian".
    """

    property_name: str | None = "Hessian"

    def __call__(self, pred_graph: Graph, ref_graph: Graph) -> LossValue:
        hessian_ref = ref_graph.nodes.hessian  # [n, R, 3]
        hessian_pred = pred_graph.nodes.hessian  # [n, R, 3]
        if hessian_ref is None or hessian_pred is None:
            # We null out the loss if the reference Hessian is not provided
            return 0

        hessian_error = optax.losses.huber_loss(
            hessian_pred,
            hessian_ref,
            delta=HUBER_LOSS_DEFAULT_DELTA,
        )
        # mean huber Hessian error
        # append 0 to match length (n_node + 1)
        hessian_error = jnp.append(jnp.mean(hessian_error, axis=(-1, -2)), 0)

        return ref_graph.globals.weight * safe_divide(
            sum_nodes_of_the_same_graph(ref_graph, hessian_error),
            ref_graph.n_node,
        )


class MSEPartialChargesLoss(LossTerm):
    """Loss for the mean-squared error of the partial charges.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "partial_charges".
    """

    property_name: str | None = "partial_charges"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        partial_charges_ref = ref_graph.nodes.partial_charges  # [n_nodes, ]
        partial_charges_pred = pred_graph.nodes.partial_charges  # [n_nodes, ]

        if partial_charges_ref is None or partial_charges_pred is None:
            # We null out the loss if the reference partial charges are not provided
            return 0

        node_valid = ~jnp.isnan(partial_charges_ref)  # [n_nodes]
        safe_ref = jnp.where(node_valid, partial_charges_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * safe_divide(
            sum_nodes_of_the_same_graph(
                ref_graph, jnp.square(safe_ref - partial_charges_pred)
            ),
            ref_graph.n_node,
        )
        graph_valid = (
            sum_nodes_of_the_same_graph(ref_graph, node_valid.astype(jnp.float32)) > 0
        )
        return jnp.where(graph_valid, loss_per_graph, 0.0)  # [n_graphs, ]


class HuberPartialChargesLoss(LossTerm):
    """Loss for the adaptive Huber loss of the partial charges.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "partial_charges".
    """

    property_name: str | None = "partial_charges"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        partial_charges_ref = ref_graph.nodes.partial_charges  # [n_nodes, ]
        partial_charges_pred = pred_graph.nodes.partial_charges  # [n_nodes, ]

        if partial_charges_ref is None or partial_charges_pred is None:
            # We null out the loss if the reference partial charges are not provided
            return 0

        node_valid = ~jnp.isnan(partial_charges_ref)  # [n_nodes]
        safe_ref = jnp.where(node_valid, partial_charges_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * safe_divide(
            sum_nodes_of_the_same_graph(
                ref_graph,
                optax.losses.huber_loss(
                    partial_charges_pred,
                    safe_ref,
                    delta=HUBER_LOSS_DEFAULT_DELTA,
                ),
            ),
            ref_graph.n_node,
        )
        graph_valid = (
            sum_nodes_of_the_same_graph(ref_graph, node_valid.astype(jnp.float32)) > 0
        )
        return jnp.where(graph_valid, loss_per_graph, 0.0)  # [n_graphs, ]


class MSEChargeLoss(LossTerm):
    """Loss for the mean-squared error of the total charge.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "charge".
    """

    property_name: str | None = "charge"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        charge_ref = ref_graph.globals.charge  # [n_graphs, ]
        charge_pred = pred_graph.globals.non_corrected_charge  # [n_graphs, ]

        if charge_ref is None or charge_pred is None:
            # We null out the loss if the reference charge is not provided
            return 0

        valid = ~jnp.isnan(charge_ref)  # [n_graphs]
        safe_ref = jnp.where(valid, charge_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * safe_divide(
            jnp.square(safe_ref - charge_pred),
            ref_graph.n_node,
        )
        return jnp.where(valid, loss_per_graph, 0.0)  # [n_graphs, ]


class HuberChargeLoss(LossTerm):
    """Loss for the Huber loss of the total charge.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "charge".
    """

    property_name: str | None = "charge"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        charge_ref = ref_graph.globals.charge  # [n_graphs, ]
        charge_pred = pred_graph.globals.non_corrected_charge  # [n_graphs, ]

        if charge_ref is None or charge_pred is None:
            # We null out the loss if the reference charge is not provided
            return 0

        valid = ~jnp.isnan(charge_ref)  # [n_graphs]
        safe_ref = jnp.where(valid, charge_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * safe_divide(
            optax.losses.huber_loss(
                charge_pred,
                safe_ref,
                delta=HUBER_LOSS_DEFAULT_DELTA,
            ),
            ref_graph.n_node,
        )
        return jnp.where(valid, loss_per_graph, 0.0)  # [n_graphs, ]


class MSEDipoleMomentLoss(LossTerm):
    """Loss for the mean-squared error of the dipole moment.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "dipole_moment".
    """

    property_name: str | None = "dipole_moment"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        dipole_moment_ref = ref_graph.globals.dipole_moment  # [n_graphs, 3]
        dipole_moment_pred = pred_graph.globals.dipole_moment  # [n_graphs, 3]

        if dipole_moment_ref is None or dipole_moment_pred is None:
            # We null out the loss if the reference dipole moment is not provided
            return 0

        valid = ~jnp.isnan(dipole_moment_ref).any(axis=1)  # [n_graphs]
        safe_ref = jnp.where(valid[:, None], dipole_moment_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * safe_divide(
            jnp.mean(
                jnp.square(safe_ref - dipole_moment_pred),
                axis=1,
            ),
            ref_graph.n_node,
        )
        return jnp.where(valid, loss_per_graph, 0.0)  # [n_graphs, ]


class HuberDipoleMomentLoss(LossTerm):
    """Loss for the Huber loss of the dipole moment.

    Attributes:
        property_name: The name of the property that is being penalized in this loss.
                       Mostly meant for use in metrics logging. For this class, it's
                       "dipole_moment".
    """

    property_name: str | None = "dipole_moment"

    def __call__(
        self,
        pred_graph: Graph,
        ref_graph: Graph,
    ) -> LossValue:
        dipole_moment_ref = ref_graph.globals.dipole_moment  # [n_graphs, 3]
        dipole_moment_pred = pred_graph.globals.dipole_moment  # [n_graphs, 3]

        if dipole_moment_ref is None or dipole_moment_pred is None:
            # We null out the loss if the reference dipole moment is not provided
            return 0

        valid = ~jnp.isnan(dipole_moment_ref).any(axis=1)  # [n_graphs]
        safe_ref = jnp.where(valid[:, None], dipole_moment_ref, 0.0)
        loss_per_graph = ref_graph.globals.weight * safe_divide(
            jnp.mean(
                optax.losses.huber_loss(
                    dipole_moment_pred,
                    safe_ref,
                    delta=HUBER_LOSS_DEFAULT_DELTA,
                ),
                axis=1,
            ),
            ref_graph.n_node,
        )
        return jnp.where(valid, loss_per_graph, 0.0)  # [n_graphs, ]
