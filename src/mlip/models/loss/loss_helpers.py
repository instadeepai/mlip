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
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mlip.graph import Graph

HUBER_LOSS_DEFAULT_DELTA = 0.01
HUBER_DELTAS_SCALING_FORCES = [1.0, 0.7, 0.4, 0.1]
HUBER_THRESHOLDS_FORCES = [100, 200, 300]


def safe_divide(num: jax.Array, denom: jax.Array) -> jax.Array:
    """Divide two arrays but return zero where the denominator is zero."""
    return jnp.where(denom == 0.0, 0.0, num / denom)


def sum_nodes_of_the_same_graph(graph: Graph, node_quantities: jax.Array) -> jax.Array:
    """Sums the quantities of the nodes for each graph"""
    return e3nn.scatter_sum(node_quantities, nel=graph.n_node)  # [ n_graphs,]


def compute_adaptive_huber_loss_forces(pred: jax.Array, ref: jax.Array) -> jax.Array:
    """Computes the adaptive Huber loss between two arrays. This is the adaptive
    Huber loss that is only used for the `HuberForcesLoss`.
    """
    deltas = HUBER_LOSS_DEFAULT_DELTA * np.array(HUBER_DELTAS_SCALING_FORCES)

    cond_1 = jnp.linalg.norm(ref, axis=-1) < HUBER_THRESHOLDS_FORCES[0]
    cond_2 = (jnp.linalg.norm(ref, axis=-1) > HUBER_THRESHOLDS_FORCES[0]) & (
        jnp.linalg.norm(ref, axis=-1) < HUBER_THRESHOLDS_FORCES[1]
    )
    cond_3 = (jnp.linalg.norm(ref, axis=-1) > HUBER_THRESHOLDS_FORCES[1]) & (
        jnp.linalg.norm(ref, axis=-1) < HUBER_THRESHOLDS_FORCES[2]
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
