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
import queue
import threading
from typing import Any, Callable, Iterable, Optional

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jraph import GraphsTuple

from mlip.data.helpers.graph_dataset import GraphDataset

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
        preprocess_fn: Optional[Callable] = None,
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

        # Start the prefetch
        self.thread = threading.Thread(target=self._prefetch, daemon=True)
        self.thread.start()

    def _prefetch(self):
        """Prefetch items from the original iterable into the queue.

        If a preprocess function is provided, it will be applied to each item before
        placing it into the queue.

        This method also adds a None at the end to indicate the end of the iterator.
        """
        for raw_item in self.iterable:
            item = self.preprocess_fn(raw_item) if self.preprocess_fn else raw_item
            self.queue.put(item)  # This will block when the queue is full

        # Indicate the end of the iterator
        self.queue.put(None)

    def __iter__(self):
        """Implementation of the iterator. It starts a new thread once completed."""
        item = self.queue.get()
        while item is not None:
            yield item
            item = self.queue.get()

        # Wait for the prefetch thread to finish before restarting.
        # A brief join handles the race between queue.put(None) and thread exit.
        self.thread.join(timeout=1)
        if self.thread.is_alive():
            raise RuntimeError("Prefetch thread did not terminate")
        self.thread = threading.Thread(target=self._prefetch, daemon=True)
        self.thread.start()

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

    def __iter__(self) -> GraphsTuple:
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


class UnsqueezeGraphDatasetWrapper:
    """Wraps a GraphDataset to yield batches with a leading (1, ...) dimension.

    This makes single-device GraphDataset batches compatible with the vmap+jit
    training step which expects stacked batches of shape (n_devices, ...).
    """

    def __init__(self, dataset: GraphDataset):
        self._dataset = dataset

    def __iter__(self):
        for batch in self._dataset:
            yield jax.tree.map(lambda x: x[None, ...], batch)

    def __len__(self):
        return len(self._dataset)

    def subset(self, i: slice | int | list | float) -> "UnsqueezeGraphDatasetWrapper":
        return UnsqueezeGraphDatasetWrapper(self._dataset.subset(i))


def create_global_arrays(stacked_batch: GraphsTuple, mesh: Mesh) -> GraphsTuple:
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
