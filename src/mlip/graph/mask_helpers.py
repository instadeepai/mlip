# Copyright 2020 DeepMind Technologies Limited.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
"""Implementations in this file adapted from `jraph.utils`."""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from mlip.graph import Graph


def _get_mask(padding_length: int, full_length: int) -> jax.Array:
    """Creates simple boolean mask based on full length of array and padding length.

    Adapted from `jraph.utils._get_mask`.
    """
    valid_length = full_length - padding_length
    return jnp.arange(full_length, dtype=jnp.int32) < valid_length


def _get_number_of_padding_with_graphs_graphs(padded_graph: "Graph") -> int:
    """Returns number of padding graphs in given padded graph.

    Adapted from `jraph.utils.get_number_of_padding_with_graphs_graphs`.

    Warning: This method only gives results for graphs that have been padded with
    the `pad_with_graphs` function.

    Args:
      padded_graph: A graph that has been padded with `pad_with_graphs`.

    Returns:
      The number of padding graphs.
    """
    # The first padding graph always has at least one padding node, and
    # all padding graphs that follow have 0 nodes. We can count how many
    # trailing graphs have 0 nodes, and add one.
    n_trailing_empty_padding_graphs = jnp.argmin(padded_graph.n_node[::-1] == 0)
    return n_trailing_empty_padding_graphs + 1


def _get_number_of_padding_with_graphs_nodes(padded_graph: "Graph") -> int:
    """Returns number of padding nodes in given padded graph.

    Adapted from `jraph.utils.get_number_of_padding_with_graphs_nodes`.

    Warning: This method only gives results for graphs that have been padded with the
    `pad_with_graphs` function.

    Args:
      padded_graph: A graph that has been padded with `pad_with_graphs`.

    Returns:
      The number of padding nodes.
    """
    return padded_graph.n_node[-_get_number_of_padding_with_graphs_graphs(padded_graph)]


def get_graph_padding_mask(padded_graph: "Graph") -> jax.Array:
    """Returns a mask for the graphs of a padded graph.

    Adapted from `jraph.utils.get_graph_padding_mask`.

    Args:
        padded_graph: Graph padded using `pad_with_graphs` function.

    Returns:
        Boolean array of shape [total_num_graphs] containing True for real graphs,
        and False for padding graphs.
    """
    if len(padded_graph.n_node.shape) != 1:
        raise ValueError(
            "Graph has an extra leading dimension. "
            "Cannot compute graph padding mask on stacked graph."
        )

    if padded_graph.num_graphs == 1:
        return jnp.ones(1, dtype=bool)

    # NOTE: If > 1 graph, always assumes at least 1 padding graph
    n_padding_graph = _get_number_of_padding_with_graphs_graphs(padded_graph)
    return _get_mask(
        padding_length=n_padding_graph, full_length=padded_graph.num_graphs
    )


def get_node_padding_mask(padded_graph: "Graph") -> jax.Array:
    """Returns a mask for the nodes of a padded graph.

    Adapted from `jraph.utils.get_node_padding_mask`.

    NOTE:
    If only 1 graph, we assume it's a real graph (jraph instead always assumes at least
    1 padding graph). If more than 1 graph, we align with the jraph mask: the last graph
    is a padding graph, and all empty graphs are also not real graphs.

    Args:
        padded_graph: Graph padded using `pad_with_graphs` function.

    Returns:
        Boolean array of shape [total_num_nodes] containing True for real nodes,
        and False for padding nodes.
    """
    if len(padded_graph.n_node.shape) != 1:
        raise ValueError(
            "Graph has an extra leading dimension. "
            "Cannot compute graph padding mask on stacked graph."
        )

    if padded_graph.num_graphs == 1:
        return jnp.ones(padded_graph.nodes.positions.shape[0], dtype=bool)

    n_padding_node = _get_number_of_padding_with_graphs_nodes(padded_graph)
    flat_node_features = jax.tree_util.tree_leaves(padded_graph.nodes)

    if not flat_node_features:
        raise ValueError("Padded graph must have at least one array of node features")
    total_num_nodes = flat_node_features[0].shape[0]
    return _get_mask(padding_length=n_padding_node, full_length=total_num_nodes)
