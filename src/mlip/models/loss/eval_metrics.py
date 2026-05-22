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
import dataclasses
from typing import TypeAlias

import jax
import jax.numpy as jnp

from mlip.graph import Graph
from mlip.utils.safe_norm import safe_divide

Metrics: TypeAlias = dict[str, float | jax.Array]


def _masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    return safe_divide(jnp.sum(jnp.dot(mask, x)), jnp.sum(mask))


def _compute_mae(delta: jax.Array, mask: jax.Array) -> jax.Array:
    return _masked_mean(jnp.abs(delta), mask)


def _masked_mean_f(x: jax.Array, mask: jax.Array) -> jax.Array:
    return safe_divide(jnp.sum(mask[..., jnp.newaxis] * x), jnp.sum(mask) * 3)


def _compute_mae_f(delta: jax.Array, mask: jax.Array) -> jax.Array:
    return _masked_mean_f(jnp.abs(delta), mask)


def _masked_mean_stress(x: jax.Array, mask: jax.Array) -> jax.Array:
    return safe_divide(
        jnp.sum(mask[..., jnp.newaxis, jnp.newaxis] * x), jnp.sum(mask) * 9
    )


def _compute_mae_stress(delta: jax.Array, mask: jax.Array) -> jax.Array:
    return _masked_mean_stress(jnp.abs(delta), mask)


def _compute_mse(delta: jax.Array, mask: jax.Array) -> jax.Array:
    return _masked_mean(jnp.square(delta), mask)


def _compute_mse_f(delta: jax.Array, mask: jax.Array) -> jax.Array:
    return _masked_mean_f(jnp.square(delta), mask)


def _compute_mse_stress(delta: jax.Array, mask: jax.Array) -> jax.Array:
    return _masked_mean_stress(jnp.square(delta), mask)


PropertyName: TypeAlias = str
PropertyDeltasTotal: TypeAlias = list[jax.Array]
PropertyDeltasPerAtom: TypeAlias = list[jax.Array]


@dataclasses.dataclass
class _PropertyDeltas:
    """Small helper dataclass to combine delta lists for return
    type of `_compute_deltas_lists`.

    `mask` is the effective graph/node mask for this property (padding mask
    AND NaN-validity mask). When the property is not present at all, the
    lists are empty and `mask` is None.
    """

    total: PropertyDeltasTotal
    per_atom: PropertyDeltasPerAtom | None
    mask: jax.Array | None = None


@dataclasses.dataclass
class _PropertyDeltasCollection:
    """Small helper dataclass to provide proper typing to delta lists."""

    energy: _PropertyDeltas
    forces: _PropertyDeltas
    stress: _PropertyDeltas
    hessian: _PropertyDeltas
    partial_charges: _PropertyDeltas
    charge: _PropertyDeltas
    dipole_moment: _PropertyDeltas


def _compute_delta_lists(
    pred_graph: Graph,
    ref_graph: Graph,
    graph_mask: jax.Array,
    node_mask: jax.Array,
) -> _PropertyDeltasCollection:
    """Computes for energies, forces, and stress the deltas and deltas per atom.

    Returns this as a collection with one entry per property. Each entry holds
    the total and per-atom delta lists and an effective mask (padding mask AND
    NaN-validity mask — NaN targets come from datasets that did not provide the
    field and must be excluded from metric averages).

    Note that for forces, we don't compute a per-atom metric, and that one will be
    returned as None.
    """
    delta_es_list = []
    delta_es_per_atom_list = []
    energy_mask = None

    delta_fs_list = []
    forces_mask = None

    delta_stress_list = []
    delta_stress_per_atom_list = []
    stress_mask = None

    delta_hessian_list = []

    delta_partial_charges_list = []
    partial_charges_mask = None

    delta_charge_list = []
    delta_charge_per_atom_list = []
    charge_mask = None

    delta_dipole_moment_list = []
    delta_dipole_moment_per_atom_list = []
    dipole_moment_mask = None

    if ref_graph.globals.energy is not None and pred_graph.globals.energy is not None:
        energy_valid = ~jnp.isnan(ref_graph.globals.energy)
        energy_mask = graph_mask & energy_valid
        safe_ref = jnp.where(energy_valid, ref_graph.globals.energy, 0.0)
        delta = safe_ref - pred_graph.globals.energy
        delta = jnp.where(energy_valid, delta, 0.0)
        delta_es_list.append(delta)
        delta_es_per_atom_list.append(safe_divide(delta, ref_graph.n_node))

    if ref_graph.nodes.forces is not None and pred_graph.nodes.forces is not None:
        node_forces_valid = ~jnp.isnan(ref_graph.nodes.forces).any(axis=1)
        forces_mask = node_mask & node_forces_valid
        safe_ref = jnp.where(node_forces_valid[:, None], ref_graph.nodes.forces, 0.0)
        delta = safe_ref - pred_graph.nodes.forces
        delta = jnp.where(node_forces_valid[:, None], delta, 0.0)
        delta_fs_list.append(delta)

    if ref_graph.globals.stress is not None and pred_graph.globals.stress is not None:
        stress_valid = ~jnp.isnan(ref_graph.globals.stress).any(axis=(1, 2))
        stress_mask = graph_mask & stress_valid
        safe_ref = jnp.where(stress_valid[:, None, None], ref_graph.globals.stress, 0.0)
        delta = safe_ref - pred_graph.globals.stress
        delta = jnp.where(stress_valid[:, None, None], delta, 0.0)
        delta_stress_list.append(delta)
        delta_stress_per_atom_list.append(
            safe_divide(delta, ref_graph.n_node[:, None, None])
        )

    if (
        pred_graph.nodes.hessian is not None and ref_graph.nodes.hessian is not None
        # ref Hessian could be null during inference
    ):
        # mean over rows axis
        delta_hs = jnp.mean(
            ref_graph.nodes.hessian - pred_graph.nodes.hessian,
            axis=1,
        )
        delta_hessian_list.append(delta_hs)
    else:
        delta_hessian_list.append(jnp.zeros(ref_graph.num_graphs))

    if (
        ref_graph.nodes.partial_charges is not None
        and pred_graph.nodes.partial_charges is not None
    ):
        partial_charges_valid = ~jnp.isnan(ref_graph.nodes.partial_charges)
        partial_charges_mask = node_mask & partial_charges_valid
        safe_ref_pc = jnp.where(
            partial_charges_valid, ref_graph.nodes.partial_charges, 0.0
        )
        delta_pc = safe_ref_pc - pred_graph.nodes.partial_charges
        delta_pc = jnp.where(partial_charges_valid, delta_pc, 0.0)
        delta_partial_charges_list.append(delta_pc)

    if (
        ref_graph.globals.charge is not None
        and pred_graph.globals.non_corrected_charge is not None
    ):
        charge_valid = ~jnp.isnan(ref_graph.globals.charge)
        charge_mask = graph_mask & charge_valid
        safe_ref_charge = jnp.where(charge_valid, ref_graph.globals.charge, 0.0)
        delta_charge = safe_ref_charge - pred_graph.globals.non_corrected_charge
        delta_charge = jnp.where(charge_valid, delta_charge, 0.0)
        delta_charge_list.append(delta_charge)
        delta_charge_per_atom_list.append(safe_divide(delta_charge, ref_graph.n_node))

    if (
        ref_graph.globals.dipole_moment is not None
        and pred_graph.globals.dipole_moment is not None
    ):
        dipole_moment_valid = ~jnp.isnan(ref_graph.globals.dipole_moment).any(axis=-1)
        dipole_moment_mask = graph_mask & dipole_moment_valid
        safe_ref_dm = jnp.where(
            dipole_moment_valid[:, None], ref_graph.globals.dipole_moment, 0.0
        )
        delta_dm = safe_ref_dm - pred_graph.globals.dipole_moment
        delta_dm = jnp.where(dipole_moment_valid[:, None], delta_dm, 0.0)
        delta_dipole_moment_list.append(delta_dm)
        delta_dipole_moment_per_atom_list.append(
            safe_divide(delta_dm, ref_graph.n_node[:, None])
        )

    return _PropertyDeltasCollection(
        energy=_PropertyDeltas(delta_es_list, delta_es_per_atom_list, energy_mask),
        forces=_PropertyDeltas(delta_fs_list, None, forces_mask),
        stress=_PropertyDeltas(
            delta_stress_list, delta_stress_per_atom_list, stress_mask
        ),
        hessian=_PropertyDeltas(delta_hessian_list, None, node_mask),
        partial_charges=_PropertyDeltas(
            delta_partial_charges_list, None, partial_charges_mask
        ),
        charge=_PropertyDeltas(
            delta_charge_list, delta_charge_per_atom_list, charge_mask
        ),
        dipole_moment=_PropertyDeltas(
            delta_dipole_moment_list,
            delta_dipole_moment_per_atom_list,
            dipole_moment_mask,
        ),
    )


def _compute_metrics_from_deltas(
    deltas: _PropertyDeltasCollection,
    extended_metrics: bool,
):
    """Computes the metrics dictionary from a given set of delta lists containing the
    deviations between prediction and ground truth for multiple properties.
    """

    metrics: Metrics = {
        "mae_e": jnp.nan,
        "mae_e_per_atom": jnp.nan,
        "mse_e": jnp.nan,
        "mse_e_per_atom": jnp.nan,
        "mae_f": jnp.nan,
        "mse_f": jnp.nan,
        "mae_stress": jnp.nan,
        "mae_stress_per_atom": jnp.nan,
        "mse_stress": jnp.nan,
        "mse_stress_per_atom": jnp.nan,
        "mae_hes": jnp.nan,
        "mse_hes": jnp.nan,
        "mae_partial_charges": jnp.nan,
        "mse_partial_charges": jnp.nan,
        "mae_charge": jnp.nan,
        "mse_charge": jnp.nan,
        "mae_charge_per_atom": jnp.nan,
        "mse_charge_per_atom": jnp.nan,
        "mae_dipole_moment": jnp.nan,
        "mse_dipole_moment": jnp.nan,
        "mae_dipole_moment_per_atom": jnp.nan,
        "mse_dipole_moment_per_atom": jnp.nan,
    }

    if len(deltas.energy.total) > 0:
        delta_es = jnp.concatenate(deltas.energy.total, axis=0)
        delta_es_per_atom = jnp.concatenate(deltas.energy.per_atom, axis=0)

        metrics.update({
            # Mean absolute error
            "mae_e": _compute_mae(delta_es, deltas.energy.mask),
            # Mean-square error
            "mse_e": _compute_mse(delta_es, deltas.energy.mask),
        })
        if extended_metrics:
            metrics.update({
                # Mean absolute error
                "mae_e_per_atom": _compute_mae(delta_es_per_atom, deltas.energy.mask),
                # Mean-square error
                "mse_e_per_atom": _compute_mse(delta_es_per_atom, deltas.energy.mask),
            })

    if len(deltas.forces.total) > 0:
        delta_fs = jnp.concatenate(deltas.forces.total, axis=0)
        metrics.update({
            # Mean absolute error
            "mae_f": _compute_mae_f(delta_fs, deltas.forces.mask),
            # Mean-square error
            "mse_f": _compute_mse_f(delta_fs, deltas.forces.mask),
        })

    if len(deltas.stress.total) > 0 and extended_metrics:
        delta_stress = jnp.concatenate(deltas.stress.total, axis=0)
        delta_stress_per_atom = jnp.concatenate(deltas.stress.per_atom, axis=0)
        metrics.update({
            # Mean absolute error
            "mae_stress": _compute_mae_stress(delta_stress, deltas.stress.mask),
            "mae_stress_per_atom": _compute_mae_stress(
                delta_stress_per_atom, deltas.stress.mask
            ),
            # Mean-square error
            "mse_stress": _compute_mse_stress(delta_stress, deltas.stress.mask),
            "mse_stress_per_atom": _compute_mse_stress(
                delta_stress_per_atom, deltas.stress.mask
            ),
        })

    if len(deltas.hessian.total) > 0:
        delta_hessians = jnp.concatenate(deltas.hessian.total, axis=0)

        metrics.update({
            # Mean absolute error
            "mae_hes": _compute_mae_f(delta_hessians, deltas.hessian.mask),
            # Mean-square error
            "mse_hes": _compute_mse_f(delta_hessians, deltas.hessian.mask),
        })

    if len(deltas.partial_charges.total) > 0:
        delta_partial_charges = jnp.concatenate(deltas.partial_charges.total, axis=0)
        metrics.update({
            # Mean absolute error
            "mae_partial_charges": _compute_mae(
                delta_partial_charges, deltas.partial_charges.mask
            ),
            # Mean-square error
            "mse_partial_charges": _compute_mse(
                delta_partial_charges, deltas.partial_charges.mask
            ),
        })

    if len(deltas.charge.total) > 0:
        delta_charge = jnp.concatenate(deltas.charge.total, axis=0)
        delta_charge_per_atom = jnp.concatenate(deltas.charge.per_atom, axis=0)
        metrics.update({
            # Mean absolute error
            "mae_charge": _compute_mae(delta_charge, deltas.charge.mask),
            # Mean-square error
            "mse_charge": _compute_mse(delta_charge, deltas.charge.mask),
        })
        if extended_metrics:
            metrics.update({
                "mae_charge_per_atom": _compute_mae(
                    delta_charge_per_atom, deltas.charge.mask
                ),
                "mse_charge_per_atom": _compute_mse(
                    delta_charge_per_atom, deltas.charge.mask
                ),
            })

    if len(deltas.dipole_moment.total) > 0:
        delta_dipole_moment = jnp.concatenate(deltas.dipole_moment.total, axis=0)
        delta_dipole_moment_per_atom = jnp.concatenate(
            deltas.dipole_moment.per_atom, axis=0
        )
        metrics.update({
            # Mean absolute error
            "mae_dipole_moment": _compute_mae_f(
                delta_dipole_moment, deltas.dipole_moment.mask
            ),
            # Mean-square error
            "mse_dipole_moment": _compute_mse_f(
                delta_dipole_moment, deltas.dipole_moment.mask
            ),
        })
        if extended_metrics:
            metrics.update({
                "mae_dipole_moment_per_atom": _compute_mae_f(
                    delta_dipole_moment_per_atom, deltas.dipole_moment.mask
                ),
                "mse_dipole_moment_per_atom": _compute_mse_f(
                    delta_dipole_moment_per_atom, deltas.dipole_moment.mask
                ),
            })

    return metrics


def compute_eval_metrics(
    pred_graph: Graph,
    ref_graph: Graph,
    extended_metrics: bool = False,
) -> Metrics:
    """Computes the evaluation metrics for a given prediction and reference graph.

    Args:
        pred_graph: The graph containing the predictions.
        ref_graph: The reference graph with the ground truth values.
        extended_metrics: Whether to compute extended metrics. Default is `False`.
                          Without extended metrics means that "per-atom" metrics and
                          stress metrics are omitted. The smaller set just includes
                          MSE and MAE metrics for energies and forces.

    Returns:
        The metrics as a dictionary. These metrics are averaged over the given batch.
    """
    graph_mask = ref_graph.graph_mask()  # [n_graphs,]
    node_mask = ref_graph.node_mask()  # [n_nodes,]

    deltas = _compute_delta_lists(pred_graph, ref_graph, graph_mask, node_mask)

    return _compute_metrics_from_deltas(deltas, extended_metrics)
