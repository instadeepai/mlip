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

# Copyright 2022 Mario Geiger.

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
"""Modified version of jraph.utils.dynamically_batch."""

from typing import Callable, Generator, Iterator

import jraph
import numpy as np

_NUMBER_FIELDS = ("n_node", "n_edge", "n_graph")


def _get_graph_size(graph: jraph.GraphsTuple) -> tuple[int, int, int]:
    n_node = np.sum(graph.n_node)
    n_edge = len(graph.senders)
    n_graph = len(graph.n_node)
    return n_node, n_edge, n_graph


def _is_over_batch_size(
    graph: jraph.GraphsTuple, graph_batch_size: tuple[int, int, int]
) -> bool:
    graph_size = _get_graph_size(graph)
    return any(x > y for x, y in zip(graph_size, graph_batch_size))


def dynamically_batch(
    graphs_tuple_iterator: Iterator[jraph.GraphsTuple],
    n_node: int,
    n_edge: int,
    n_graph: int,
    pad_fn: Callable[[jraph.GraphsTuple], jraph.GraphsTuple] = None,
    skip_last_batch: bool = False,
) -> Generator[jraph.GraphsTuple, None, None]:
    """Dynamically batches trees with ``jraph.GraphsTuples`` up to specified sizes.

    Elements of the ``graphs_tuple_iterator`` will be incrementally added to a batch
    until the limits defined by ``n_node``, ``n_edge`` and ``n_graph`` are reached. This
    means each element yielded by this generator may have a differing number of
    graphs in its batch.

    Args:
      graphs_tuple_iterator: An iterator of ``jraph.GraphsTuples``.
      n_node: The maximum number of nodes in a batch, at least the maximum sized
              graph + 1.
      n_edge: The maximum number of edges in a batch, at least the maximum sized
              graph.
      n_graph: The maximum number of graphs in a batch, at least 2.
      pad_fn: A function for padding. If ``None`` (default),
              then use jraph.pad_with_graphs.
      skip_last_batch: Whether to skip the last batch. The default is false.

    Yields:
      A ``jraph.GraphsTuple`` batch of graphs.

    Raises:
      ValueError: if the number of graphs is < 2.
      RuntimeError: if the ``graphs_tuple_iterator`` contains elements which are not
                    ``jraph.GraphsTuple`` objects.
      RuntimeError: if a graph is found which is larger than the batch size.
    """
    if pad_fn is None:

        def jraph_pad(graphs):
            return jraph.pad_with_graphs(graphs, n_node, n_edge, n_graph)

        pad_fn = jraph_pad
    if n_graph < 2:
        raise ValueError(
            "The number of graphs in a batch size must be greater or "
            f"equal to `2` for padding with graphs, got {n_graph}."
        )
    valid_batch_size = (n_node - 1, n_edge, n_graph - 1)
    accumulated_graphs = []
    num_accumulated_nodes = 0
    num_accumulated_edges = 0
    num_accumulated_graphs = 0
    for element in graphs_tuple_iterator:
        element_nodes, element_edges, element_graphs = _get_graph_size(element)
        if _is_over_batch_size(element, valid_batch_size):
            # First yield the batched graph so far if exists.
            if accumulated_graphs:
                yield pad_fn(jraph.batch_np(accumulated_graphs))

            # Then report the error.
            graph_size = element_nodes, element_edges, element_graphs
            graph_size = dict(zip(_NUMBER_FIELDS, graph_size))
            batch_size = dict(zip(_NUMBER_FIELDS, valid_batch_size))
            raise RuntimeError(
                "Found graph bigger than batch size. Valid Batch "
                f"Size: {batch_size}, Graph Size: {graph_size}"
            )

        # If this is the first element of the batch, set it and continue.
        # Otherwise, check if there is space for the graph in the batch:
        #   if there is, add it to the batch
        #   if there isn't, return the old batch and start a new batch.
        if not accumulated_graphs:
            accumulated_graphs = [element]
            num_accumulated_nodes = element_nodes
            num_accumulated_edges = element_edges
            num_accumulated_graphs = element_graphs
            continue
        else:
            if (
                (num_accumulated_graphs + element_graphs > n_graph - 1)
                or (num_accumulated_nodes + element_nodes > n_node - 1)
                or (num_accumulated_edges + element_edges > n_edge)
            ):
                yield pad_fn(jraph.batch_np(accumulated_graphs))
                accumulated_graphs = [element]
                num_accumulated_nodes = element_nodes
                num_accumulated_edges = element_edges
                num_accumulated_graphs = element_graphs
            else:
                accumulated_graphs.append(element)
                num_accumulated_nodes += element_nodes
                num_accumulated_edges += element_edges
                num_accumulated_graphs += element_graphs

    # We may still have data in batched graph.
    if accumulated_graphs and not skip_last_batch:
        yield pad_fn(jraph.batch_np(accumulated_graphs))
