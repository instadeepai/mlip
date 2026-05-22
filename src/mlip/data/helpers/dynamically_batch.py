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

import functools
from typing import Callable, Generator, Iterable

import numpy as np

from mlip.graph import Graph
from mlip.graph.batching_helpers import batch_graphs, pad_with_graphs

_NUMBER_FIELDS = ("n_node", "n_edge", "n_graph", "n_edge_long_range")


def _get_graph_size(graph: Graph) -> tuple[int, int, int, int]:
    n_node = np.sum(graph.n_node)
    n_edge = len(graph.senders)
    n_graph = graph.num_graphs
    n_edge_long_range = (
        len(graph.senders_long_range) if graph.senders_long_range is not None else 0
    )
    return n_node, n_edge, n_graph, n_edge_long_range


def _is_over_batch_size(
    graph: Graph, graph_batch_size: tuple[int, int, int, int]
) -> bool:
    graph_size = _get_graph_size(graph)
    return any(x > y for x, y in zip(graph_size, graph_batch_size))


def dynamically_batch(
    graphs_iterator: Iterable[Graph],
    n_node: int,
    n_edge: int,
    n_graph: int,
    n_edge_long_range: int | None = None,
    pad_fn: Callable[[Graph], Graph] = None,
    skip_last_batch: bool = False,
) -> Generator[Graph, None, None]:
    """Dynamically batches trees with `Graphs` up to specified sizes.

    Elements of the `graphs_iterator` will be incrementally added to a batch
    until the limits defined by `n_node`, `n_edge` and `n_graph` are reached. This
    means each element yielded by this generator may have a differing number of
    graphs in its batch.

    Args:
      graphs_iterator: An iterator of `Graph`.
      n_node: The maximum number of nodes in a batch, at least the maximum sized
              graph + 1.
      n_edge: The maximum number of edges in a batch, at least the maximum sized
              graph.
      n_graph: The maximum number of graphs in a batch, at least 2.
      n_edge_long_range: The maximum number of long-range edges in a batch.
              Set to `None` (default) when the graphs do not carry a long-range
              neighbor list; otherwise must be at least the maximum sized graph's
              long-range edge count.
      pad_fn: A function for padding. If `None` (default), then use the standard
              `pad_with_graphs`.
      skip_last_batch: Whether to skip the last batch. The default is false.

    Yields:
      A `Graph` batch of graphs.

    Raises:
      ValueError: if the number of graphs is < 2.
      RuntimeError: if the `graphs_iterator` contains elements which are not
                    `Graph` objects.
      RuntimeError: if a graph is found which is larger than the batch size.
    """
    if pad_fn is None:
        pad_fn = functools.partial(
            pad_with_graphs,
            n_node=n_node,
            n_edge=n_edge,
            n_graph=n_graph,
            n_edge_long_range=n_edge_long_range,
        )

    if n_graph < 2:
        raise ValueError(
            "The number of graphs in a batch size must be greater or "
            f"equal to `2` for padding with graphs, got {n_graph}."
        )
    valid_n_edge_long_range = n_edge_long_range if n_edge_long_range is not None else 0
    valid_batch_size = (n_node - 1, n_edge, n_graph - 1, valid_n_edge_long_range)
    accumulated_graphs = []
    num_accumulated_nodes = 0
    num_accumulated_edges = 0
    num_accumulated_graphs = 0
    for element in graphs_iterator:
        element_nodes, element_edges, element_graphs, element_edge_long_range = (
            _get_graph_size(element)
        )
        if _is_over_batch_size(element, valid_batch_size):
            # First yield the batched graph so far if exists.
            if accumulated_graphs:
                yield pad_fn(batch_graphs(accumulated_graphs))

            # Then report the error.
            graph_size = (
                element_nodes,
                element_edges,
                element_graphs,
                element_edge_long_range,
            )
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
            num_accumulated_edge_long_range = element_edge_long_range
            continue
        else:
            if (
                (num_accumulated_graphs + element_graphs > n_graph - 1)
                or (num_accumulated_nodes + element_nodes > n_node - 1)
                or (num_accumulated_edges + element_edges > n_edge)
                or (
                    num_accumulated_edge_long_range + element_edge_long_range
                    > valid_n_edge_long_range
                )
            ):
                yield pad_fn(batch_graphs(accumulated_graphs))
                accumulated_graphs = [element]
                num_accumulated_nodes = element_nodes
                num_accumulated_edges = element_edges
                num_accumulated_graphs = element_graphs
                num_accumulated_edge_long_range = element_edge_long_range
            else:
                accumulated_graphs.append(element)
                num_accumulated_nodes += element_nodes
                num_accumulated_edges += element_edges
                num_accumulated_graphs += element_graphs
                num_accumulated_edge_long_range += element_edge_long_range

    # We may still have data in batched graph.
    if accumulated_graphs and not skip_last_batch:
        yield pad_fn(batch_graphs(accumulated_graphs))
