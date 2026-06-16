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
from typing import Iterable, Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from typing_extensions import Self

from mlip.data.graph_dataset import GraphDataset, GraphDatasetState
from mlip.data.helpers.data_prefetching import (
    PrefetchIterator,
    wrap_dataset_with_prefetch,
)
from mlip.graph import Graph
from mlip.utils.multihost import create_device_mesh

logger = logging.getLogger("mlip")

BatchGenerator: TypeAlias = Iterable[Graph]
InterleavingMethod: TypeAlias = Literal["regular", "random"]


class CombinedGraphDataset:
    """A dataset wrapper that combines multiple `GraphDataset` instances into
    a single iterable dataset.

    This class enables iteration over multiple graph datasets either by randomly mixing
    them or by interleaving them in a deterministic way.

    The deterministic interleaving (regular) approach combines two iterables
    (`iterators_long` and `iterators_short`) into a single generator while preserving
    an ordering compatible with a multi-host setup. Each group of `N` consecutive items
    must be homogeneous, i.e all items in the group must come from the same source
    iterable. To enforce this, the generator interleaves items as follows: after every
    `R * N` items drawn from `iterators_long`, `N` items are drawn from
    `iterators_short`. Both sequences are therefore consumed in chunks that are
    divisible by `N`, ensuring that each `N` loaded batches are items from only
    one iterable.

    Where:
    - `R` is the ratio between `iterators_long` and `iterators_short`.
    - `N` is the number of devices.
    """

    def __init__(
        self,
        graph_datasets: list[GraphDataset],
        interleaving_method: InterleavingMethod = "regular",
        mesh: Mesh | None = None,
        seed: int = 0,
    ):
        """Class constructor, does not handle prefetching and parallelism"""
        self.graph_datasets = graph_datasets
        self.interleaving_method = interleaving_method
        self.mesh = mesh if mesh is not None else create_device_mesh()
        # state object of the CombinedGraphDataset
        self.state = GraphDatasetState(
            rng=jax.random.PRNGKey(seed),
            num_graphs_processed=jnp.int32(0),
        )
        self._harmonize_states()

    @classmethod
    def init(
        cls,
        graph_datasets: list[GraphDataset | PrefetchIterator],
        interleaving_method: InterleavingMethod = "regular",
        mesh: Mesh | None = None,
    ) -> PrefetchIterator | Self:
        """Initializes an instance of `CombinedGraphDataset` and automatically
        handles its conversion into a `PrefetchedIterator` object in case all
        items in the `graph_datasets` list are instances of `PrefetchedIterator`.
        """
        # prefetch the combined dataset if all graph datasets
        # are instances of `PrefetchIterator`
        prefetch = all((isinstance(ds, PrefetchIterator) for ds in graph_datasets))

        if prefetch:
            prefetch_count = min(ds.iterable.prefetch_count for ds in graph_datasets)
            prefetch_num_devices = graph_datasets[0].prefetch_count

            # unwrap prefetch iterators into graph datasets
            graph_datasets = [
                ds.iterable.iterable.graph_dataset for ds in graph_datasets
            ]

            combined_dataset = cls(graph_datasets, interleaving_method, mesh)
            mesh = combined_dataset.mesh

            # wrap the combined graph dataset into a prefetch iterator
            return wrap_dataset_with_prefetch(
                combined_dataset, mesh, prefetch_count, prefetch_num_devices
            )
        return cls(graph_datasets, interleaving_method, mesh)

    def __len__(self):
        """Returns the total number of graphs in both graph datasets."""
        return sum(len(i) for i in self.graph_datasets)

    def number_of_graphs(self) -> int:
        """Total number of graphs across all sub-datasets."""
        return sum(ds.number_of_graphs() for ds in self.graph_datasets)

    def number_of_nodes(self) -> int:
        """Total number of nodes across all sub-datasets."""
        return sum(ds.number_of_nodes() for ds in self.graph_datasets)

    def subset(self):
        """Constructs a new `CombinedGraphDataset` object containing a new
        list of `GraphDataset` objects each containing a subset of graphs
        of the current ones.
        """
        raise NotImplementedError(
            "Generating a subset is not implemented for CombinedGraphDataset."
        )

    def __iter__(self):
        """Iterate over the combined dataset according to the interleaving strategy:
        randomized or deterministic (regular) interleaving.
        """
        next_rng, shuffle_key = self._harmonize_states()

        if self.interleaving_method == "random":
            to_sample_from = []
            for idx, ds in enumerate(self.graph_datasets):
                to_sample_from += [idx] * len(ds)

            to_sample_from = [
                to_sample_from[i]
                for i in jax.random.permutation(shuffle_key, len(to_sample_from))
            ]

        elif self.interleaving_method == "regular":
            if len(self.graph_datasets) != 2:
                raise NotImplementedError(
                    "Regular interleaving only supports exactly two datasets."
                )
            indices_a = [0] * len(self.graph_datasets[0])
            indices_b = [1] * len(self.graph_datasets[1])

            a_is_short = len(indices_a) < len(indices_b)
            indices_short = indices_a if a_is_short else indices_b
            indices_long = indices_b if a_is_short else indices_a

            to_sample_from = self._get_regular_interleaving_indices(
                indices_long, indices_short, len(self.mesh.devices)
            )

        else:
            raise NotImplementedError("Unknown interleaving method.")

        # `self.state.num_graphs_processed` counts BATCHES yielded by the
        # combined iterator (the loop below increments by 1 per yield), not
        # raw graphs.  Each yielded batch from a sub-iterator can hold a
        # variable number of graphs, so we cannot derive sub-cursors with
        # pure arithmetic.  Instead, replay the prefix on each sub-iterator
        # — its internal `_state.num_graphs_processed` advances naturally
        # per yield, leaving it positioned to resume the suffix correctly.
        num_processed = int(self.state.num_graphs_processed)
        batches_per_sub = [0] * len(self.graph_datasets)
        for ds_idx in to_sample_from[:num_processed]:
            batches_per_sub[ds_idx] += 1

        iterators = [iter(ds) for ds in self.graph_datasets]
        for sub_iter, n in zip(iterators, batches_per_sub):
            for _ in range(n):
                next(sub_iter)

        to_sample_from = to_sample_from[num_processed:]
        for ds_idx in to_sample_from:
            self.state = self.state.replace(
                num_graphs_processed=self.state.num_graphs_processed + 1
            )
            try:
                item = next(iterators[ds_idx])
            except StopIteration:
                continue
            yield item
        self.state = self.state.replace(rng=next_rng, num_graphs_processed=jnp.int32(0))

    def _get_regular_interleaving_indices(
        self,
        indices_long: list[int],
        indices_short: list[int],
        num_devices: int | None = None,
    ) -> BatchGenerator:
        """Return a generator that yields a merged stream of items from two datasets."""
        ratio = len(indices_long) // max(1, len(indices_short))
        to_sample_from = []
        num_from_long = ratio * num_devices if num_devices is not None else ratio
        num_from_short = num_devices or 1
        for idx, item in enumerate(indices_long):
            to_sample_from.append(item)
            if (idx + 1) % num_from_long == 0:
                # check short iterator still contains enough items
                if len(indices_short) >= num_from_short:
                    for _ in range(num_from_short):
                        to_sample_from.append(indices_short.pop(0))
        if indices_short:
            logger.warning(
                "Dropping %d trailing batch(es) from the shorter dataset to "
                "preserve homogeneous %d-batch chunks for multi-device sharding.",
                len(indices_short),
                num_from_short,
            )
        return to_sample_from

    def _harmonize_states(self) -> tuple[jax.Array, jax.Array]:
        """Reseed sub-dataset states from the combined rng.

        Splits `self.state.rng` into one key per sub-dataset (so each
        sub-dataset shuffles independently rather than from a shared key)
        plus `next_rng` and `shuffle_key` for the combined dataset.
        Each sub-dataset's state is replaced with a fresh
        :class:`GraphDatasetState` carrying its own key and a zeroed
        counter; callers position the counters separately when resuming
        mid-epoch.

        Returns:
            A pair `(next_rng, shuffle_key)`: `next_rng` is the rng to
            advance `self.state.rng` to at end-of-epoch; `shuffle_key`
            permutes `to_sample_from` for the current epoch (random
            mode only).
        """
        keys = jax.random.split(self.state.rng, 2 + len(self.graph_datasets))
        next_rng, shuffle_key = keys[0], keys[1]
        for sub_ds, sub_key in zip(self.graph_datasets, keys[2:]):
            sub_ds.state = GraphDatasetState(
                rng=sub_key,
                num_graphs_processed=jnp.int32(0),
            )
        return next_rng, shuffle_key
