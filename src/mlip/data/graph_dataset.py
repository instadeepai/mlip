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
import logging
from typing import Callable

import jax
import jax.numpy as jnp
from flax.struct import dataclass as flax_dataclass

from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.data.helpers.exceptions import GraphsDiscardedError
from mlip.graph import Graph
from mlip.graph.batching_helpers import (
    homogenize_graph_fields,
    validate_batch_compatible,
)

logger = logging.getLogger("mlip")


@flax_dataclass
class GraphDatasetState:
    """State of a graph dataset, checkpointable through the wrapper chain.

    Attributes:
        rng: Random key for shuffling.  Stays at the *pre-split* value
            during an epoch so that mid-epoch checkpoints can replay the
            same `jax.random.split` to reproduce the shuffle permutation.
            Advanced to the post-split value at the end of the epoch.
        num_graphs_processed: Number of graphs yielded so far in the current epoch.
            Used to resume iteration from the correct position after a checkpoint.
    """

    rng: jax.random.PRNGKey = dataclasses.field(
        default_factory=lambda: jax.random.PRNGKey(0)
    )
    num_graphs_processed: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.int32(0)
    )


class GraphDataset:
    """Class for holding a dataset consisting of graphs, i.e., `Graph`,
    and managing batching.
    """

    def __init__(
        self,
        graphs: list[Graph],
        batch_size: int,
        max_n_node: int,
        max_n_edge: int,
        min_n_node: int = 1,
        min_n_edge: int = 1,
        min_n_graph: int = 1,
        max_n_edge_long_range: int | None = None,
        shuffle: bool = True,
        shuffle_between_epochs: bool = True,
        skip_last_batch: bool = False,
        raise_exc_if_graphs_discarded: bool = False,
        graph_postprocessing: list[Callable[[Graph], Graph]] | None = None,
        seed: int = 0,
        homogenize: bool = False,
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
            max_n_edge_long_range: The maximum number of long range edges contributed by
                                   one graph in a batch. If None, long range
                                   interactions are not considered.
            shuffle: Whether to shuffle the graphs before iterating, defaults
                            to True.
            shuffle_between_epochs: If true, then reshuffle data between epochs
                                           but only if should_shuffle is also true.
            skip_last_batch: Whether to skip the last batch. The default is false.
            raise_exc_if_graphs_discarded: Whether to raise an exception if there are
                                           graphs that must be discarded due to size
                                           constraints. Default is False, which means
                                           only a warning is logged.
            graph_postprocessing: Optional list of functions applied to each
                batched graph when iterating. Default is None.
            seed: The random seed to use for shuffling. Default is 0.
            homogenize: If True, pad missing
                `Prediction`-targeted optional fields (e.g. `stress`, `forces`)
                with NaN so graphs from heterogeneous datasets share the same
                pytree structure and can be batched. After optional
                homogenization, the dataset validates that the provided graphs
                are batch-compatible and raises a clear error otherwise.
                Defaults to False.
        """
        if homogenize:
            graphs = homogenize_graph_fields(graphs)
        validate_batch_compatible(graphs, homogenize)

        self.graphs = graphs
        self.total_num_graphs = len(graphs)
        self.batch_size = batch_size
        self.max_n_node = max_n_node
        self.max_n_edge = max_n_edge
        if max_n_edge_long_range is not None:
            self.max_n_edge_long_range = max_n_edge_long_range
            self.n_edge_long_range = self.batch_size * self.max_n_edge_long_range
        else:
            self.max_n_edge_long_range = None
            self.n_edge_long_range = None
        # Plus one for the extra padding node.
        self.n_node = self.batch_size * self.max_n_node + 1
        # Times two because we want backwards edges.
        self.n_edge = self.batch_size * self.max_n_edge * 2
        self.n_graph = batch_size + 1
        self.min_n_node = min_n_node
        self.min_n_edge = min_n_edge
        self.min_n_graph = min_n_graph
        self.shuffle = shuffle
        self.shuffle_between_epochs = shuffle_between_epochs and shuffle
        self._skip_last_batch = skip_last_batch
        # Length means number of batches here
        self._length = None
        self._homogenize = homogenize

        # if applicable, the state is updated using the restored state from checkpoint
        self._next_rng = None
        self._state = GraphDatasetState(
            rng=jax.random.PRNGKey(seed),
            num_graphs_processed=jnp.int32(0),
        )

        keep_graphs = [
            graph
            for graph in self.graphs
            if graph.n_node.sum() <= self.n_node - 1
            and graph.n_edge.sum() <= self.n_edge
            and (
                (
                    self.n_edge_long_range is None
                )  # We are not considering long range interactions
                or (
                    graph.n_edge_long_range is not None
                    and graph.n_edge_long_range.sum() <= self.n_edge_long_range
                )
            )
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

        if self.shuffle:
            logger.debug("Shuffling data now...")
            rng, subkey = jax.random.split(self._state.rng, 2)
            self._state = self._state.replace(rng=rng)
            indices = jax.random.permutation(subkey, len(self.graphs))
            self.graphs = [self.graphs[i] for i in indices]

        self._graph_postprocessing = graph_postprocessing or []

    @property
    def state(self) -> GraphDatasetState:
        return self._state

    @state.setter
    def state(self, new_state: GraphDatasetState) -> None:
        self._state = new_state
        self._next_rng = None

    def __iter__(self):
        graphs = self.graphs.copy()

        if self.shuffle and self.shuffle_between_epochs:
            logger.debug("Shuffling data now...")
            rng, subkey = jax.random.split(self._state.rng, 2)
            # We dont update the rng immediately so the state
            # has the info to yield these batches
            self._next_rng = rng
            graphs = [graphs[i] for i in jax.random.permutation(subkey, len(graphs))]

        graphs = graphs[int(self._state.num_graphs_processed) :]

        for batched_graph in dynamically_batch(
            graphs,
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
            n_edge_long_range=self.n_edge_long_range,
            skip_last_batch=self._skip_last_batch,
        ):
            for f in self._graph_postprocessing:
                batched_graph = f(batched_graph)  # noqa
            num_graphs_in_batch = batched_graph.graph_mask().sum()
            next_processed_graphs = (
                self._state.num_graphs_processed + num_graphs_in_batch
            )
            self._state = self._state.replace(
                num_graphs_processed=next_processed_graphs
            )
            yield batched_graph

        self._state = self._state.replace(num_graphs_processed=jnp.int32(0))
        if self._next_rng is not None:
            self._state = self._state.replace(rng=self._next_rng)
            self._next_rng = None

    def __len__(self):
        """Returns the number of batches but does not recompute them each time."""
        if self._length is not None:
            return self._length

        self._length = 0

        # Here we call dynamically batch instead of iterating over self.__iter__()
        # to avoid shuffling and state updates.
        for _ in dynamically_batch(
            self.graphs.copy(),
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
            n_edge_long_range=self.n_edge_long_range,
            skip_last_batch=self._skip_last_batch,
        ):
            self._length += 1
        return self._length

    def subset(self, i: slice | int | list | float) -> "GraphDataset":
        """Constructs and returns a new graph dataset containing a subset of
        graphs of the current one with given slicing information `i`.

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
            max_n_edge_long_range=self.max_n_edge_long_range,
            shuffle=self.shuffle,
            shuffle_between_epochs=self.shuffle_between_epochs,
            skip_last_batch=self._skip_last_batch,
            graph_postprocessing=self._graph_postprocessing,
            homogenize=self._homogenize,
        )
        logger.debug("...and with %s batches.", len(dataset_subset))
        return dataset_subset

    def number_of_graphs(self) -> int:
        """Returns the number of graphs in the dataset.

        Returns:
            The number of graphs in this dataset.
        """
        return len(self.graphs)

    def number_of_nodes(self) -> int:
        """Returns the number of nodes in the dataset.

        Returns:
            The number of nodes in this dataset.
        """
        total = 0
        for graph in self:
            total += graph.node_mask().sum()
        return total
