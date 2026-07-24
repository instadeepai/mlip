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

import logging

import jax.numpy as jnp
from jax import Array

from mlip.simulation.fep.alchemical_graph import AlchemicalGraph

logger = logging.getLogger("mlip")


def get_alchemical_edge_mask(graph: AlchemicalGraph) -> tuple[Array, Array | None]:
    """Compute a mask for whether an edge crosses the alchemical boundary.

    Args:
        graph: AlchemicalGraph with globals.alchemical_atom_indices set.

    Returns:
        Tuple of (short_range_mask, long_range_mask). Each is a boolean array,
        True if the edge crosses the boundary, False otherwise. long_range_mask
        is None if the graph has no long-range edges.
    """
    alchemical_atom_indices = graph.globals.alchemical_atom_indices
    if alchemical_atom_indices is None or len(alchemical_atom_indices) == 0:
        raise ValueError("alchemical_atom_indices must be provided.")

    def _xor_mask(senders: Array, receivers: Array) -> Array:
        return jnp.logical_xor(
            jnp.isin(senders, alchemical_atom_indices),
            jnp.isin(receivers, alchemical_atom_indices),
        )

    short_range_mask = _xor_mask(graph.senders, graph.receivers)
    long_range_mask = (
        _xor_mask(graph.senders_long_range, graph.receivers_long_range)
        if graph.senders_long_range is not None
        else None
    )
    return short_range_mask, long_range_mask


def _pack_edges(
    senders: Array,
    receivers: Array,
    n_edge: Array,
    shifts: Array | None,
    mask: Array,
    n_nodes: int,
) -> tuple[Array, Array, Array, Array | None]:
    """Pack kept edges to the front, replacing dropped edges with OOB padding.

    Only supports running on a single real subgraph, as all kept edges are
    assigned to the subgraph in slot 0. Not suitable for batched simulation.

    Uses the same logic as `jax_md.partition.neighbor_list` masking: a cumsum
    scatter moves kept edges to contiguous front positions, and the last slot
    is always set to the OOB index so it is never mistaken for a real edge.

    Args:
        senders: Sender indices, shape (E,).
        receivers: Receiver indices, shape (E,).
        n_edge: Per-subgraph edge counts, shape (n_subgraphs,), n_subgraphs <= 2.
        shifts: Periodic shift vectors, shape (E, 3), or None.
        mask: Boolean keep-mask, shape (E,). True = keep.
        n_nodes: OOB padding index (total real nodes).

    Returns:
        Tuple of (senders, receivers, n_edge, shifts) after packing.
        n_edge[0] is updated to the number of kept edges.
    """
    senders = jnp.reshape(senders, (-1,))
    receivers = jnp.reshape(receivers, (-1,))
    assert mask.shape[0] == senders.shape[0] == receivers.shape[0]
    if n_edge.shape[0] > 2:
        raise ValueError(
            "_pack_edges only supports a single real subgraph; got "
            f"n_edge.shape={n_edge.shape}."
        )

    cumsum = jnp.cumsum(mask)
    index = jnp.where(mask, cumsum - 1, len(receivers) - 1)

    oob = n_nodes * jnp.ones(receivers.shape, jnp.int32)
    senders_out = oob.at[index].set(senders).at[-1].set(n_nodes)
    receivers_out = oob.at[index].set(receivers).at[-1].set(n_nodes)

    if shifts is not None:
        shifts_out = (
            jnp.zeros(shifts.shape, shifts.dtype).at[index].set(shifts).at[-1].set(0)
        )
    else:
        shifts_out = None

    # Padding graph will have incorrect n_edge_out, but this is not used downstream.
    n_edge_out = jnp.ones(n_edge.shape, jnp.int32).at[0].set(cumsum[-1])

    return senders_out, receivers_out, n_edge_out, shifts_out


def prune_edges_with_mask(
    graph: AlchemicalGraph, mask: Array, long_range_mask: Array | None = None
) -> AlchemicalGraph:
    """
    Prune edges from a graph based on a mask (True to keep, False to remove).

    Uses the same logic as the masking function in jax_md.partition.neighbor_list:
    moves all masked edges to the end of the sender/receiver arrays,
    and replaces value with OOB value.

    Args:
        graph: The graph to prune, with senders/receivers of shape (E,).
        mask: The mask to use to prune the edges, of shape (E,).
        long_range_mask: Optional mask for long-range edges, shape (E_lr,).

    Returns:
        The pruned graph.
    """
    n_nodes = jnp.sum(graph.n_node[:-1]) if len(graph.n_node) > 1 else graph.n_node[0]

    senders, receivers, n_edge, shifts = _pack_edges(
        graph.senders, graph.receivers, graph.n_edge, graph.edges.shifts, mask, n_nodes
    )
    result = graph.replace(
        n_node=graph.n_node,
        n_edge=n_edge,
        senders=senders,
        receivers=receivers,
        edges=graph.edges.replace(shifts=shifts),
    )

    if graph.edges_long_range is not None and long_range_mask is not None:
        lr_senders, lr_receivers, lr_n_edge, lr_shifts = _pack_edges(
            graph.senders_long_range,
            graph.receivers_long_range,
            graph.n_edge_long_range,
            graph.edges_long_range.shifts,
            long_range_mask,
            n_nodes,
        )
        result = result.replace(
            senders_long_range=lr_senders,
            receivers_long_range=lr_receivers,
            n_edge_long_range=lr_n_edge,
            edges_long_range=graph.edges_long_range.replace(shifts=lr_shifts),
        )

    return result
