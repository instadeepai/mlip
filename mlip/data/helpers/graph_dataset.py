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
import random

import jraph

from mlip.data.helpers.dynamically_batch import dynamically_batch

logger = logging.getLogger("mlip")


class GraphsDiscardedError(Exception):
    """Exception to be raised if some graphs are invalid due to the given parameters
    for batch size, max. number of nodes, and max. number of edges."""


class GraphDataset:
    """Class for holding a dataset consisting of graphs, i.e., ``jraph.GraphsTuple``,
    and managing batching.
    """

    def __init__(
        self,
        graphs: list[jraph.GraphsTuple],
        batch_size: int,
        max_n_node: int,
        max_n_edge: int,
        min_n_node: int = 1,
        min_n_edge: int = 1,
        min_n_graph: int = 1,
        should_shuffle: bool = True,
        should_shuffle_between_epochs: bool = True,
        skip_last_batch: bool = False,
        raise_exc_if_graphs_discarded: bool = False,
    ):
        """Constructor.

        Args:
            graphs: The graphs to store and manage in this class.
            batch_size: The batch size.
            max_n_node: The maximum number of nodes contributed by one graph in a batch.
            max_n_edge: The maximum number of edges contributed by one graph in a batch.
            min_n_node: The minimum number of nodes in a batch, defaults to 1.
            min_n_edge: The minimum number of edges in a batch, defaults to 1.
            min_n_graph: The minimum number of graphs in a batch, defaults to 1.
            should_shuffle: Whether to shuffle the graphs before iterating, defaults
                            to True.
            should_shuffle_between_epochs: If true, then reshuffle data between epochs
                                           but only if should_shuffle is also true.
            skip_last_batch: Whether to skip the last batch. The default is false.
            raise_exc_if_graphs_discarded: Whether to raise an exception if there are
                                           graphs that must be discarded due to size
                                           constraints. Default is False, which means
                                           only a warning is logged.
        """
        self.graphs = graphs
        self.total_num_graphs = len(graphs)
        self.batch_size = batch_size
        self.max_n_node = max_n_node
        self.max_n_edge = max_n_edge
        # Plus one for the extra padding node.
        self.n_node = self.batch_size * self.max_n_node + 1
        # Times two because we want backwards edges.
        self.n_edge = self.batch_size * self.max_n_edge * 2
        self.n_graph = batch_size + 1
        self.min_n_node = min_n_node
        self.min_n_edge = min_n_edge
        self.min_n_graph = min_n_graph
        self.should_shuffle = should_shuffle
        self._should_shuffle_between_epochs = (
            should_shuffle_between_epochs and should_shuffle
        )
        self._skip_last_batch = skip_last_batch
        # Length means number of batches here
        self._length = None

        keep_graphs = [
            graph
            for graph in self.graphs
            if graph.n_node.sum() <= self.n_node - 1
            and graph.n_edge.sum() <= self.n_edge
        ]
        if len(keep_graphs) != len(self.graphs):
            if raise_exc_if_graphs_discarded:
                raise GraphsDiscardedError(
                    "With the given values of batch_size, max_n_node, "
                    "and max_n_edge, not all graphs are valid."
                )
            logger.warning(
                "Discarded %s graphs due to size constraints.",
                len(self.graphs) - len(keep_graphs),
            )
        self.graphs = keep_graphs
        self.total_num_graphs = len(self.graphs)

        if self.should_shuffle:
            random.seed(len(self.graphs))
            logger.debug("Shuffling data now...")
            random.shuffle(self.graphs)

    def __iter__(self):
        """Batch over the dataset, according to a batching strategy."""
        graphs = self.graphs.copy()  # this is a shallow copy

        if self.should_shuffle and self._should_shuffle_between_epochs:
            logger.debug("Shuffling data now...")
            random.shuffle(graphs)

        for batched_graph in dynamically_batch(
            graphs,
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
            skip_last_batch=self._skip_last_batch,
        ):
            yield batched_graph

    def __len__(self):
        """Returns the number of batches but does not recompute them each time."""
        if self._length is not None:
            return self._length

        self._length = 0
        copy_should_shuffle = self.should_shuffle
        self.should_shuffle = False
        for _ in self:
            self._length += 1
        self.should_shuffle = copy_should_shuffle
        return self._length

    def subset(self, i: slice | int | list | float):
        """Constructs and returns a new graph dataset containing a subset of
        graphs of the current one with given slicing information ``i``.

        Args:
            i: The slicing information. See source code for options.

        Returns:
            A new graph dataset containing only a subset of the graphs of the
            current one.
        """
        graphs = self.graphs
        if isinstance(i, slice):
            graphs = graphs[i]
        elif isinstance(i, int):
            graphs = graphs[:i]
        elif isinstance(i, list):
            graphs = [graphs[j] for j in i]
        elif isinstance(i, float):
            graphs = graphs[: int(len(graphs) * i)]
        else:
            raise TypeError("Subset slicing information i has incorrect type.")

        logger.debug("Constructing subset with %s graphs...", len(graphs))
        dataset_subset = GraphDataset(
            graphs=graphs,
            max_n_node=self.max_n_node,
            max_n_edge=self.max_n_edge,
            batch_size=self.batch_size,
            min_n_node=self.min_n_node,
            min_n_edge=self.min_n_edge,
            min_n_graph=self.min_n_graph,
            should_shuffle=self.should_shuffle,
            should_shuffle_between_epochs=self._should_shuffle_between_epochs,
            skip_last_batch=self._skip_last_batch,
        )
        logger.debug("...and with %s batches.", len(dataset_subset))
        return dataset_subset
