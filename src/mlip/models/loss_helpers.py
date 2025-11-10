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
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from jraph import GraphsTuple

from mlip.typing import Prediction

HUBER_LOSS_DEFAULT_DELTA = 0.01


def safe_divide(x, y):
    return jnp.where(y == 0.0, 0.0, x / y)


def _masked_mean(x, mask):
    return jnp.sum(jnp.dot(mask, x)) / jnp.sum(mask)


def compute_mae(delta: jnp.ndarray, mask) -> float:
    return _masked_mean(jnp.abs(delta), mask)


def _masked_mean_f(x, mask):
    return jnp.sum(mask[..., jnp.newaxis] * x) / (jnp.sum(mask) * 3)


def compute_mae_f(delta: jnp.ndarray, mask) -> float:
    return _masked_mean_f(jnp.abs(delta), mask)


def _masked_mean_stress(x, mask):
    return jnp.sum(mask[..., jnp.newaxis, jnp.newaxis] * x) / (jnp.sum(mask) * 9)


def compute_mae_stress(delta: jnp.ndarray, mask) -> float:
    return _masked_mean_stress(jnp.abs(delta), mask)


def compute_mse(delta: jnp.ndarray, mask) -> float:
    return _masked_mean(jnp.square(delta), mask)


def compute_mse_f(delta: jnp.ndarray, mask) -> float:
    return _masked_mean_f(jnp.square(delta), mask)


def compute_mse_stress(delta: jnp.ndarray, mask) -> float:
    return _masked_mean_stress(jnp.square(delta), mask)


def _sum_nodes_of_the_same_graph(
    graph: GraphsTuple, node_quantities: jnp.ndarray
) -> jnp.ndarray:
    return e3nn.scatter_sum(node_quantities, nel=graph.n_node)  # [ n_graphs,]


def _compute_adaptive_huber_loss_forces(
    pred: np.ndarray, ref: np.ndarray
) -> np.ndarray:
    deltas = HUBER_LOSS_DEFAULT_DELTA * np.array([1.0, 0.7, 0.4, 0.1])

    cond_1 = jnp.linalg.norm(ref, axis=-1) < 100
    cond_2 = (jnp.linalg.norm(ref, axis=-1) > 100) & (
        jnp.linalg.norm(ref, axis=-1) < 200
    )
    cond_3 = (jnp.linalg.norm(ref, axis=-1) > 200) & (
        jnp.linalg.norm(ref, axis=-1) < 300
    )
    cond_4 = ~(cond_1 | cond_2 | cond_3)

    cond_1 = jnp.stack([cond_1] * 3, axis=1)
    cond_2 = jnp.stack([cond_2] * 3, axis=1)
    cond_3 = jnp.stack([cond_3] * 3, axis=1)
    cond_4 = jnp.stack([cond_4] * 3, axis=1)

    output = jnp.zeros_like(pred)
    output = jnp.where(
        cond_1, optax.losses.huber_loss(pred, ref, delta=deltas[0]), output
    )
    output = jnp.where(
        cond_2, optax.losses.huber_loss(pred, ref, delta=deltas[1]), output
    )
    output = jnp.where(
        cond_3, optax.losses.huber_loss(pred, ref, delta=deltas[2]), output
    )
    output = jnp.where(
        cond_4, optax.losses.huber_loss(pred, ref, delta=deltas[3]), output
    )

    return output


def mean_squared_error_energy(
    graph: GraphsTuple, energy_pred: np.ndarray
) -> np.ndarray:
    energy_ref = graph.globals.energy  # [n_graphs, ]
    if energy_ref is None:
        # We null out the loss if the reference energy is not provided
        energy_ref = jnp.zeros_like(energy_pred)
        energy_pred = jnp.zeros_like(energy_pred)
    return graph.globals.weight * jnp.square(
        safe_divide(energy_ref - energy_pred, graph.n_node)
    )  # [n_graphs, ]


def huber_loss_energy(graph: GraphsTuple, energy_pred: np.ndarray) -> np.ndarray:
    energy_ref = graph.globals.energy  # [n_graphs, ]
    if energy_ref is None:
        # We null out the loss if the reference energy is not provided
        energy_ref = jnp.zeros_like(energy_pred)
        energy_pred = jnp.zeros_like(energy_pred)
    return graph.globals.weight * optax.losses.huber_loss(
        safe_divide(energy_pred, graph.n_node),
        safe_divide(energy_ref, graph.n_node),
        delta=HUBER_LOSS_DEFAULT_DELTA,
    )  # [n_graphs, ]


def mean_squared_error_forces(
    graph: GraphsTuple, forces_pred: np.ndarray
) -> np.ndarray:
    forces_ref = graph.nodes.forces  # [n_nodes, 3]
    if forces_ref is None:
        # We null out the loss if the reference forces are not provided
        forces_ref = jnp.zeros_like(forces_pred)
        forces_pred = jnp.zeros_like(forces_pred)
    return graph.globals.weight * safe_divide(
        _sum_nodes_of_the_same_graph(
            graph, jnp.mean(jnp.square(forces_ref - forces_pred), axis=1)
        ),
        graph.n_node,
    )  # [n_graphs, ]


def adaptive_huber_loss_forces(
    graph: GraphsTuple, forces_pred: np.ndarray
) -> np.ndarray:
    forces_ref = graph.nodes.forces  # [n_nodes, 3]
    if forces_ref is None:
        # We null out the loss if the reference forces are not provided
        forces_ref = jnp.zeros_like(forces_pred)
        forces_pred = jnp.zeros_like(forces_pred)
    return graph.globals.weight * safe_divide(
        _sum_nodes_of_the_same_graph(
            graph,
            jnp.mean(
                _compute_adaptive_huber_loss_forces(forces_pred, forces_ref),
                axis=1,
            ),
        ),
        graph.n_node,
    )  # [n_graphs, ]


def mean_squared_error_stress(
    graph: GraphsTuple, stress_pred: np.ndarray
) -> np.ndarray:
    stress_ref = graph.globals.stress  # [n_graphs, 3, 3]
    if stress_ref is None:
        # We null out the loss if the reference stress is not provided
        stress_ref = jnp.zeros_like(stress_pred)
        stress_pred = jnp.zeros_like(stress_pred)
    return graph.globals.weight * jnp.mean(
        jnp.square(stress_ref - stress_pred), axis=(1, 2)
    )  # [n_graphs, ]


def huber_loss_stress(graph: GraphsTuple, stress_pred: np.ndarray) -> np.ndarray:
    stress_ref = graph.globals.stress  # [n_graphs, 3, 3]
    if stress_ref is None:
        # We null out the loss if the reference stress is not provided
        stress_ref = jnp.zeros_like(stress_pred)
        stress_pred = jnp.zeros_like(stress_pred)
    return graph.globals.weight * jnp.mean(
        optax.losses.huber_loss(
            stress_pred,
            stress_ref,
            delta=HUBER_LOSS_DEFAULT_DELTA,
        ),
        axis=(1, 2),
    )  # [n_graphs, ]


def compute_eval_metrics(
    prediction: Prediction,
    ref_graph: GraphsTuple,
    extended_metrics: bool = False,
) -> dict[str, jnp.ndarray]:
    """Compute (extended) evaluation metrics dictionary."""

    graph_mask = jraph.get_graph_padding_mask(ref_graph)  # [n_graphs,]
    node_mask = jraph.get_node_padding_mask(ref_graph)  # [n_nodes,]

    delta_es_list = []
    es_list = []

    delta_es_per_atom_list = []
    es_per_atom_list = []

    delta_fs_list = []
    fs_list = []

    delta_stress_list = []
    stress_list = []

    delta_stress_per_atom_list = []
    stress_per_atom_list = []

    pred_graph = ref_graph._replace(
        nodes=ref_graph.nodes._replace(forces=prediction.forces),
        globals=ref_graph.globals._replace(
            energy=prediction.energy,
            stress=prediction.stress,
        ),
    )

    if ref_graph.globals.energy is not None:
        delta_es_list.append(ref_graph.globals.energy - pred_graph.globals.energy)
        es_list.append(ref_graph.globals.energy)

        delta_es_per_atom_list.append(
            safe_divide(
                (ref_graph.globals.energy - pred_graph.globals.energy),
                ref_graph.n_node,
            )
        )
        es_per_atom_list.append(ref_graph.globals.energy / jnp.sum(node_mask))

    if ref_graph.nodes.forces is not None:
        delta_fs_list.append(ref_graph.nodes.forces - pred_graph.nodes.forces)
        fs_list.append(ref_graph.nodes.forces)

    if ref_graph.globals.stress is not None:
        delta_stress_list.append(ref_graph.globals.stress - pred_graph.globals.stress)
        stress_list.append(ref_graph.globals.stress)

        delta_stress_per_atom_list.append(
            safe_divide(
                (ref_graph.globals.stress - pred_graph.globals.stress),
                ref_graph.n_node[:, None, None],
            )
        )
        stress_per_atom_list.append(ref_graph.globals.stress / jnp.sum(node_mask))

    metrics = {
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
    }

    if len(delta_es_list) > 0:
        delta_es = jnp.concatenate(delta_es_list, axis=0)
        delta_es_per_atom = jnp.concatenate(delta_es_per_atom_list, axis=0)

        metrics.update(
            {
                # Mean absolute error
                "mae_e": compute_mae(delta_es, graph_mask),
                # Mean-square error
                "mse_e": compute_mse(delta_es, graph_mask),
            }
        )
        if extended_metrics:
            metrics.update(
                {
                    # Mean absolute error
                    "mae_e_per_atom": compute_mae(delta_es_per_atom, graph_mask),
                    # Mean-square error
                    "mse_e_per_atom": compute_mse(delta_es_per_atom, graph_mask),
                }
            )

    if len(delta_fs_list) > 0:
        delta_fs = jnp.concatenate(delta_fs_list, axis=0)
        metrics.update(
            {
                # Mean absolute error
                "mae_f": compute_mae_f(delta_fs, node_mask),
                # Mean-square error
                "mse_f": compute_mse_f(delta_fs, node_mask),
            }
        )

    if len(delta_stress_list) > 0 and extended_metrics:
        delta_stress = jnp.concatenate(delta_stress_list, axis=0)
        delta_stress_per_atom = jnp.concatenate(delta_stress_per_atom_list, axis=0)
        metrics.update(
            {
                # Mean absolute error
                "mae_stress": compute_mae_stress(delta_stress, graph_mask),
                "mae_stress_per_atom": compute_mae_stress(
                    delta_stress_per_atom, graph_mask
                ),
                # Mean-square error
                "mse_stress": compute_mse_stress(delta_stress, graph_mask),
                "mse_stress_per_atom": compute_mse_stress(
                    delta_stress_per_atom, graph_mask
                ),
            }
        )

    return metrics
