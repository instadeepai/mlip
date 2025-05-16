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

import queue
import threading
from typing import Any, Callable, Iterable, Optional

import jax
import jax.numpy as jnp

from mlip.data.helpers.graph_dataset import GraphDataset


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
        for item in self.iterable:
            if self.preprocess_fn:
                item = self.preprocess_fn(item)
            self.queue.put(item)  # This will block when the queue is full

        # Indicate the end of the iterator
        self.queue.put(None)

    def __iter__(self):
        """Implementation of the iterator. It starts a new thread once completed."""
        item = self.queue.get()
        while item is not None:
            yield item
            item = self.queue.get()

        # Restart a new prefetch cycle
        assert not self.thread.is_alive()  # it should be dead
        self.thread = threading.Thread(target=self._prefetch, daemon=True)
        self.thread.start()

    def __len__(self):
        """Returns the length of the underlying iterable."""
        return self.length


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

    def __iter__(self):
        """The iterator for this parallel graph dataset."""
        batch = []
        for idx, graph in enumerate(self.graph_dataset):
            if idx % self.n == self.n - 1:
                batch.append(graph)
                yield jax.tree.map(lambda *x: jnp.stack(x, axis=0), *batch)
                batch = []
            else:
                batch.append(graph)

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
