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
import functools
import logging
import queue
import threading
from typing import Any, Callable, Iterable

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from mlip.data.graph_dataset import GraphDataset, GraphDatasetState
from mlip.graph import Graph

logger = logging.getLogger("mlip")


class PrefetchIterator:
    """A class to prefetch items from an iterable, with an option to preprocess
    each item.

    Attributes:
        iterable: The original iterable.
        queue: A queue to hold the prefetched items.
        preprocess_fn: An optional function to preprocess each item.
        thread: The thread used for prefetching.

    Example:

    .. code-block:: python

        def double(x):
            return x * 2

        it = PrefetchIterator(range(5), prefetch_count=2, preprocess_fn=double)
        for i in it:
            print(i)
        # Outputs: 0, 2, 4, 6, 8

    """

    def __init__(
        self,
        iterable: Iterable,
        prefetch_count: int = 1,
        preprocess_fn: Callable | None = None,
    ):
        """Constructor.

        Args:
            iterable: The iterable to prefetch from.
            prefetch_count: The maximum number of items to prefetch. Defaults to 1.
            preprocess_fn: A function to preprocess each item.
                           Should accept a single argument and return
                           the processed item. Defaults to None.
        """
        self.iterable = iterable
        self.length = len(self.iterable)
        self.prefetch_count = prefetch_count
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.preprocess_fn = preprocess_fn
        self.last_batch_state = None
        self.thread = None

    # These next two methods allows the class to be pickled if the preprocess function
    # compatible with pickling
    def __getstate__(self):
        return {
            "iterable": self.iterable,
            "length": self.length,
            "prefetch_count": self.prefetch_count,
            "preprocess_fn": self.preprocess_fn,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.queue = queue.Queue(maxsize=self.prefetch_count)
        self.thread = None

    @property
    def state(self) -> GraphDatasetState:
        return self.last_batch_state

    @state.setter
    def state(self, state: GraphDatasetState) -> None:
        assert self.queue.empty(), (
            "Cannot set state while prefetch queue is non-empty. "
            "Pre-fetched batches were produced from the old state and would "
            "be yielded before the new state takes effect."
        )
        self.iterable.state = state

    def _prefetch(self):
        """Prefetch items from the original iterable into the queue.

        If a preprocess function is provided, it will be applied to each item before
        placing it into the queue.

        This method also adds a None at the end to indicate the end of the iterator.
        """
        for raw_item in self.iterable:
            item = self.preprocess_fn(raw_item) if self.preprocess_fn else raw_item
            state = self.iterable.state
            self.queue.put((item, state))  # This will block when the queue is full

        # Indicate the end of the iterator, carrying the final (reset) state
        self.queue.put((None, self.iterable.state))

    def _start_prefetch(self):
        """Start a new prefetch thread."""
        self.thread = threading.Thread(target=self._prefetch, daemon=True)
        self.thread.start()

    def __iter__(self):
        """Implementation of the iterator. It starts a new thread once completed."""
        if self.thread is None or not self.thread.is_alive():
            self._start_prefetch()

        item, state = self.queue.get()
        while item is not None:
            self.last_batch_state = state
            yield item
            item, state = self.queue.get()
        self.last_batch_state = state

        # Thread is done; next __iter__ call will start a new prefetch cycle
        self.thread.join(timeout=1)
        assert not self.thread.is_alive()
        self.thread = None

    def __len__(self):
        """Returns the length of the underlying iterable."""
        return self.length

    def subset(self, i: slice | int | list | float) -> "PrefetchIterator":
        """Returns a new PrefetchIterator over a subset of the underlying iterable."""
        return PrefetchIterator(
            self.iterable.subset(i),
            prefetch_count=self.prefetch_count,
            preprocess_fn=self.preprocess_fn,
        )

    def number_of_graphs(self) -> int:
        """Forwards this function call to the underlying iterable if it can, otherwise
        will return zero."""
        if hasattr(self.iterable, "number_of_graphs"):
            return self.iterable.number_of_graphs()
        return 0

    def number_of_nodes(self) -> int:
        """Forwards this function call to the underlying iterable if it can, otherwise
        will return zero."""
        if hasattr(self.iterable, "number_of_nodes"):
            return self.iterable.number_of_nodes()
        return 0


def create_prefetch_iterator(iterable, prefetch_count=1, preprocess_fn=None):
    """If prefetch_count > 0, return a PrefetchIterator, otherwise just a map."""
    if prefetch_count <= 0:
        return map(preprocess_fn, iterable)
    else:
        return PrefetchIterator(
            iterable,
            prefetch_count=prefetch_count,
            preprocess_fn=preprocess_fn,
        )


class ParallelGraphDataset:
    """A graph dataset that loads multiple batches in parallel."""

    def __init__(self, graph_dataset: GraphDataset, num_parallel: int):
        """Constructor.

        Args:
            graph_dataset: The standard graph dataset to parallelize.
            num_parallel: Number of parallel batches to process.
        """
        self.graph_dataset = graph_dataset
        self.n = num_parallel

    @property
    def state(self) -> GraphDatasetState:
        return self.graph_dataset.state

    @state.setter
    def state(self, state: GraphDatasetState) -> None:
        self.graph_dataset.state = state

    def __iter__(self):
        """The iterator for this parallel graph dataset."""
        batch = []
        for idx, graph in enumerate(self.graph_dataset):
            if idx % self.n == self.n - 1:
                batch.append(graph)
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *batch)
                batch = []
            else:
                batch.append(graph)

        if batch:
            logger.warning(
                "Dropping %d remaining batch(es) that don't fill a complete "
                "parallel group of %d.",
                len(batch),
                self.n,
            )

    def __getattr__(self, name: str) -> Any:
        """This makes sure that the same attributes are available as
        in the standard graph dataset.
        """
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.graph_dataset, name)

    def __len__(self):
        """Returns the number of batches in the underlying graph dataset."""
        return len(self.graph_dataset)

    def subset(self, i: slice | int | list | float) -> "ParallelGraphDataset":
        return ParallelGraphDataset(self.graph_dataset.subset(i), self.n)

    def number_of_graphs(self) -> int:
        """Returns the number of graphs in the dataset.

        Returns:
            The number of graphs in this dataset.
        """
        total = 0

        # In multi-host mode, global arrays only expose local shards.
        # Use a JIT-compiled function to count across ALL devices so
        # every host gets identical totals.
        @jax.jit
        def _count_all(stacked_batch):
            def _count_single(batch):
                return batch.graph_mask().sum()

            _n_graphs = jax.vmap(_count_single)(stacked_batch)
            return _n_graphs.sum()

        for stacked_batch in self:
            n_graphs = _count_all(stacked_batch)
            total += int(jax.device_get(n_graphs))

        return total

    def number_of_nodes(self) -> int:
        """Returns the number of nodes in the dataset.

        Returns:
            The number of nodes in this dataset.
        """
        total = 0

        # In multi-host mode, global arrays only expose local shards.
        # Use a JIT-compiled function to count across ALL devices so
        # every host gets identical totals.
        @jax.jit
        def _count_all(stacked_batch):
            def _count_single(batch):
                return batch.node_mask().sum()

            _n_nodes = jax.vmap(_count_single)(stacked_batch)
            return _n_nodes.sum()

        for stacked_batch in self:
            n_nodes = _count_all(stacked_batch)
            total += int(jax.device_get(n_nodes))

        return total


def create_global_arrays(stacked_batch: Graph, mesh: Mesh) -> Graph:
    """Convert a stacked numpy batch to a global jax.Array for multi-host training.

    Each host provides its local sub-batches (one per local device). This function
    places each sub-batch on the corresponding local device and assembles them into
    a global array that is properly sharded across the full mesh.

    Args:
        stacked_batch: A pytree with arrays of shape (n_local_devices, ...).
        mesh: The JAX device mesh (must have a single axis for data parallelism).

    Returns:
        A pytree of global jax.Arrays properly sharded across the mesh.
    """
    n_global_devices = len(mesh.devices)
    sharding = NamedSharding(mesh, PartitionSpec(mesh.axis_names[0]))

    def _make_global(x):
        indices_map = sharding.addressable_devices_indices_map(x.shape)
        per_device = [jax.device_put(x[s], dev) for dev, s in indices_map.items()]
        global_shape = (n_global_devices, *x.shape[1:])
        return jax.make_array_from_single_device_arrays(
            global_shape, sharding, per_device
        )

    return jax.tree.map(_make_global, stacked_batch)


def wrap_dataset_with_prefetch(
    dataset: GraphDataset,
    mesh: Mesh,
    num_batch_prefetch_host: int,
    num_batch_prefetch_device: int,
) -> PrefetchIterator:
    """Wrap a graph dataset with parallel batching, prefetching, and device placement.

    Combines three layers: (1) a `ParallelGraphDataset` that stacks batches across
    devices, (2) a host-side prefetch iterator that prepares upcoming batches in a
    background thread, and (3) a device-side prefetch iterator that converts stacked
    batches into global JAX arrays sharded across the mesh.

    Args:
        dataset: The graph dataset to wrap.
        mesh: The JAX device mesh used for sharding global arrays.
        num_batch_prefetch_host: Number of stacked batches to prefetch on the host.
        num_batch_prefetch_device: Number of global-array batches to prefetch
            on devices.

    Returns:
        A `PrefetchIterator` that yields sharded global JAX arrays ready for
        multi-device training.
    """
    num_devices = len(mesh.devices)

    device_fn = functools.partial(create_global_arrays, mesh=mesh)

    parallel_dataset = ParallelGraphDataset(dataset, num_devices)
    prefetched_iterator = create_prefetch_iterator(
        create_prefetch_iterator(
            parallel_dataset,
            prefetch_count=num_batch_prefetch_host,
        ),
        prefetch_count=num_batch_prefetch_device,
        preprocess_fn=device_fn,
    )
    return prefetched_iterator
