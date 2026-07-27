# Copyright 2025 Zhongguancun Academy
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

import grain
import jax
import jax.numpy as jnp

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.graph_dataset import GraphDatasetState
from mlip.data.helpers.atomic_energies import _convert_energy_to_formation_energy
from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.data.helpers.exceptions import GraphsDiscardedError
from mlip.data.helpers.type_aliases import (
    GraphPostProcessingFunction,
    SystemsPreprocessingFunction,
)
from mlip.graph import Graph

_DISCARD_INCONSISTENCY_WARNING = """StreamingGraphDataset is resuming with \
raise_exc_if_graphs_discarded=False, so discarded graphs may be incorrectly counted \
in skips causing inconsistency. Prefer removing discarded samples up front \
(e.g. BatchingInfo.exclude_ids) and set raise_exc_if_graphs_discarded=True."""

logger = logging.getLogger("mlip")


@dataclasses.dataclass
class _SystemToGraphFn:
    """Picklable per-system transform: preprocess -> Graph (or None if dropped)."""

    graph_cutoff_angstrom: float
    long_range_cutoff_angstrom: float | None
    preprocessing_fns: list[SystemsPreprocessingFunction]
    atomic_energies_map: dict[int, float] | None
    n_node_capacity: int
    n_edge_capacity: int
    n_edge_long_range_capacity: int | None
    raise_exc_if_graphs_discarded: bool = False

    def __call__(self, system: ChemicalSystem) -> Graph | None:
        systems = [system]
        for preprocess in self.preprocessing_fns:
            systems = preprocess(systems)
            if not systems:
                return None
        system = systems[0]

        graph = Graph.from_chemical_system(
            chemical_system=system,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            long_range_cutoff_angstrom=self.long_range_cutoff_angstrom,
        )

        n_node = int(graph.n_node.sum())
        n_edge = int(len(graph.senders))
        n_edge_lr = (
            int(len(graph.senders_long_range))
            if graph.senders_long_range is not None
            else 0
        )
        oversize = n_node > self.n_node_capacity or n_edge > self.n_edge_capacity
        if self.n_edge_long_range_capacity is not None:
            oversize = oversize or n_edge_lr > self.n_edge_long_range_capacity
        if oversize:
            if self.raise_exc_if_graphs_discarded:
                raise GraphsDiscardedError(
                    "With the given values of batch_size, max_n_node, and "
                    "max_n_edge, not all graphs are valid."
                )
            return None

        if self.atomic_energies_map is not None:
            graph = graph.replace_globals(
                energy=_convert_energy_to_formation_energy(
                    graph.globals.energy,
                    graph.nodes.atomic_numbers,
                    self.atomic_energies_map,
                )
            )
        return graph


class StreamingGraphDataset:
    """Graph dataset that creates graphs on demand via Grain and dynamically batches.
    """

    def __init__(
        self,
        chemical_dataset: grain.MapDataset,
        *,
        batch_size: int,
        max_n_node: int,
        max_n_edge: int,
        max_n_edge_long_range: int | None = None,
        graph_cutoff_angstrom: float,
        long_range_cutoff_angstrom: float | None = None,
        preprocessing_fns: list[SystemsPreprocessingFunction] | None = None,
        atomic_energies_map: dict[int, float] | None = None,
        shuffle: bool = True,
        shuffle_between_epochs: bool = True,
        skip_last_batch: bool = False,
        raise_exc_if_graphs_discarded: bool = False,
        graph_postprocessing: list[GraphPostProcessingFunction] | None = None,
        seed: int = 0,
        worker_count: int = 0,
        num_nodes: int | None = None,
        num_batches: int | None = None,
    ):
        self.chemical_dataset = chemical_dataset
        self.batch_size = batch_size
        self.max_n_node = max_n_node
        self.max_n_edge = max_n_edge
        self.max_n_edge_long_range = max_n_edge_long_range
        self.graph_cutoff_angstrom = graph_cutoff_angstrom
        self.long_range_cutoff_angstrom = long_range_cutoff_angstrom
        self._preprocessing_fns = list(preprocessing_fns or [])
        self._atomic_energies_map = atomic_energies_map
        self.shuffle = shuffle
        self.shuffle_between_epochs = shuffle_between_epochs and shuffle
        self._skip_last_batch = skip_last_batch
        self._raise_exc_if_graphs_discarded = raise_exc_if_graphs_discarded
        self._graph_postprocessing = graph_postprocessing or []
        self._worker_count = worker_count
        self._seed = seed

        if max_n_edge_long_range is not None:
            self.n_edge_long_range = self.batch_size * self.max_n_edge_long_range
        else:
            self.n_edge_long_range = None
        self.n_node = self.batch_size * self.max_n_node + 1
        self.n_edge = self.batch_size * self.max_n_edge * 2
        self.n_graph = batch_size + 1

        self._num_graphs = len(chemical_dataset)
        self._num_nodes = num_nodes
        self._length = num_batches
        self._next_rng = None
        self._state = GraphDatasetState(
            rng=jax.random.PRNGKey(seed),
            num_graphs_processed=jnp.int32(0),
        )
        self.graphs = None
        self.total_num_graphs = self._num_graphs

        self._to_graph_fn = _SystemToGraphFn(
            graph_cutoff_angstrom=graph_cutoff_angstrom,
            long_range_cutoff_angstrom=long_range_cutoff_angstrom,
            preprocessing_fns=self._preprocessing_fns,
            atomic_energies_map=atomic_energies_map,
            n_node_capacity=self.n_node - 1,
            n_edge_capacity=self.n_edge,
            n_edge_long_range_capacity=self.n_edge_long_range,
            raise_exc_if_graphs_discarded=raise_exc_if_graphs_discarded,
        )

    @property
    def state(self) -> GraphDatasetState:
        return self._state

    @state.setter
    def state(self, new_state: GraphDatasetState) -> None:
        self._state = new_state
        self._next_rng = None

    def with_formation_energies(
        self, atomic_energies_map: dict[int, float]
    ) -> "StreamingGraphDataset":
        """Return a copy that subtracts atomic energies on the fly."""
        return StreamingGraphDataset(
            self.chemical_dataset,
            batch_size=self.batch_size,
            max_n_node=self.max_n_node,
            max_n_edge=self.max_n_edge,
            max_n_edge_long_range=self.max_n_edge_long_range,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            long_range_cutoff_angstrom=self.long_range_cutoff_angstrom,
            preprocessing_fns=self._preprocessing_fns,
            atomic_energies_map=atomic_energies_map,
            shuffle=self.shuffle,
            shuffle_between_epochs=self.shuffle_between_epochs,
            skip_last_batch=self._skip_last_batch,
            raise_exc_if_graphs_discarded=self._raise_exc_if_graphs_discarded,
            graph_postprocessing=self._graph_postprocessing,
            seed=self._seed,
            worker_count=self._worker_count,
            num_nodes=self._num_nodes,
            num_batches=self._length,
        )

    def _build_graph_iterator(self, seed: int):
        """Build a Grain pipeline that yields individual :class:`Graph` objects."""
        ds = self.chemical_dataset
        if self.shuffle:
            ds = ds.shuffle(seed=seed)

        may_discard = not self._raise_exc_if_graphs_discarded
        skip = self._state.num_graphs_processed
        if skip > 0:
            if may_discard:
                logger.warning(_DISCARD_INCONSISTENCY_WARNING)
            ds = ds.slice(slice(skip, None))

        ds = ds.map(self._to_graph_fn)
        if may_discard:
            ds = ds.filter(lambda graph: graph is not None)
        # h5py is not thread-safe, so use multiprocessing instead of threads.
        iter_ds = ds.to_iter_dataset(
            grain.ReadOptions(num_threads=0, prefetch_buffer_size=1)
        )
        if self._worker_count > 0:
            iter_ds = iter_ds.mp_prefetch(
                grain.MultiprocessingOptions(num_workers=self._worker_count)
            )
        return iter(iter_ds)

    def __iter__(self):
        if self.shuffle and self.shuffle_between_epochs:
            rng, subkey = jax.random.split(self._state.rng, 2)
            self._next_rng = rng
            seed = jax.random.randint(
                subkey, shape=(), minval=0, maxval=2**31-1
            ).item()
        else:
            seed = self._seed

        graphs_iter = self._build_graph_iterator(seed)

        for batched_graph in dynamically_batch(
            graphs_iter,
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
            n_edge_long_range=self.n_edge_long_range,
            skip_last_batch=self._skip_last_batch,
        ):
            for f in self._graph_postprocessing:
                batched_graph = f(batched_graph)
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

    def __len__(self) -> int:
        if self._length is not None:
            return self._length

        logger.warning(
            "StreamingGraphDataset length was not precomputed; counting batches "
            "by iterating once (this creates all graphs)."
        )
        length = sum(
            1
            for _ in dynamically_batch(
                self._build_graph_iterator(seed=self._seed),
                n_node=self.n_node,
                n_edge=self.n_edge,
                n_graph=self.n_graph,
                n_edge_long_range=self.n_edge_long_range,
                skip_last_batch=self._skip_last_batch,
            )
        )
        self._length = length
        return length

    def subset(self, i: slice | int | list | float) -> "StreamingGraphDataset":
        """Constructs and returns a new graph dataset containing a subset of
        graphs of the current one with given slicing information `i`.

        Args:
            i: The slicing information. See source code for options.

        Returns:
            A new graph dataset containing only a subset of the graphs of the
            current one.
        """
        n = len(self.chemical_dataset)
        if isinstance(i, slice):
            indices = list(range(*i.indices(n)))
        elif isinstance(i, int):
            indices = list(range(min(i, n)))
        elif isinstance(i, list):
            indices = i
        elif isinstance(i, float):
            indices = list(range(int(n * i)))
        else:
            raise TypeError("Subset slicing information i has incorrect type.")

        # Index map over the chemical MapDataset (arbitrary index lists).
        subset_ds = grain.MapDataset.source(indices).map(
            lambda idx: self.chemical_dataset[int(idx)]
        )

        return StreamingGraphDataset(
            subset_ds,
            batch_size=self.batch_size,
            max_n_node=self.max_n_node,
            max_n_edge=self.max_n_edge,
            max_n_edge_long_range=self.max_n_edge_long_range,
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            long_range_cutoff_angstrom=self.long_range_cutoff_angstrom,
            preprocessing_fns=self._preprocessing_fns,
            atomic_energies_map=self._atomic_energies_map,
            shuffle=self.shuffle,
            shuffle_between_epochs=self.shuffle_between_epochs,
            skip_last_batch=self._skip_last_batch,
            raise_exc_if_graphs_discarded=self._raise_exc_if_graphs_discarded,
            graph_postprocessing=self._graph_postprocessing,
            seed=self._seed,
            worker_count=self._worker_count,
            num_nodes=None,
            num_batches=None,
        )

    def number_of_graphs(self) -> int:
        """Returns the number of graphs in the dataset.

        Returns:
            The number of graphs in this dataset.
        """
        return self._num_graphs

    def number_of_nodes(self) -> int:
        """Returns the number of nodes in the dataset.

        Returns:
            The number of nodes in this dataset.
        """
        if self._num_nodes is not None:
            return self._num_nodes
        total = 0
        for graph in self:
            total += int(graph.node_mask().sum())
        return total
