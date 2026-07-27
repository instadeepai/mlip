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

from pathlib import Path
from typing import SupportsIndex

import grain
import jax.numpy as jnp
import pytest

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.chemical_systems_dataset import (
    ChemicalSystemsDataset,
)
from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.data.streaming_graph_dataset import StreamingGraphDataset
from mlip.graph import Graph

DATA_DIR = Path(__file__).parent.parent / "sample_data"
SMALL_ASPIRIN_DATASET_PATH = DATA_DIR / "small_aspirin_test.xyz"


class _ListDataset(ChemicalSystemsDataset):
    def __init__(self, systems: list[ChemicalSystem]):
        self._systems = list(systems)

    def __len__(self) -> int:
        return len(self._systems)

    def __getitem__(self, index: SupportsIndex) -> ChemicalSystem:
        return self._systems[index.__index__()]


def _aspirin_systems(n: int = 3) -> list[ChemicalSystem]:
    return ExtxyzReader(
        filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(),
        num_to_load=n,
    ).load()


def _aspirin_map_dataset(n: int = 3) -> grain.MapDataset:
    return grain.MapDataset.source(_ListDataset(_aspirin_systems(n)))


def _make_streaming_dataset(
    chemical_dataset: grain.MapDataset | None = None,
    *,
    n_graphs: int = 3,
    batch_size: int = 5,
    max_n_node: int = 30,
    max_n_edge: int = 90,
    shuffle: bool = False,
    **kwargs,
) -> StreamingGraphDataset:
    if chemical_dataset is None:
        chemical_dataset = _aspirin_map_dataset(n_graphs)
    return StreamingGraphDataset(
        chemical_dataset,
        batch_size=batch_size,
        max_n_node=max_n_node,
        max_n_edge=max_n_edge,
        graph_cutoff_angstrom=2.0,
        shuffle=shuffle,
        **kwargs,
    )


class TestStreamingConstruction:
    @pytest.mark.parametrize("use_formation_energies", [True, False])
    def test_batches_match_in_memory_builder_values(self, use_formation_energies):
        """Spot-check against the known aspirin fixture values from the
        in-memory ``SingleGraphDatasetBuilder`` tests (shuffle=False)."""
        ds = _make_streaming_dataset(
            atomic_energies_map=(
                {1: -875.4269754478838, 6: -984.8553473788693, 8: -437.7134877239419}
                if use_formation_energies
                else None
            ),
            num_batches=1,
        )
        assert ds.graphs is None
        assert ds.number_of_graphs() == 3
        assert len(ds) == 1

        batch = next(iter(ds))
        assert isinstance(batch, Graph)
        assert batch.nodes.positions.shape == (30 * 5 + 1, 3)
        assert list(batch.n_node) == [21, 21, 21, 88, 0, 0]

        expected_e = [-17617.8269, -17618.0474, -17618.0293, 0.0, 0.0, 0.0]
        if use_formation_energies:
            expected_e = [0.140974, -0.0795407, -0.061433, 0.0, 0.0, 0.0]
        assert list(batch.globals.energy) == pytest.approx(expected_e, abs=1e-4)

    def test_with_formation_energies_returns_copy(self):
        ds = _make_streaming_dataset(num_batches=1)
        mapped = ds.with_formation_energies(
            {1: -875.4269754478838, 6: -984.8553473788693, 8: -437.7134877239419}
        )
        assert mapped is not ds
        energies = list(next(iter(mapped)).globals.energy)
        assert energies == pytest.approx(
            [0.140974, -0.0795407, -0.061433, 0.0, 0.0, 0.0], abs=1e-4
        )


class TestStreamingDynamicBatch:
    def test_does_not_force_batch_size_graphs(self):
        ds = _make_streaming_dataset(batch_size=16, num_batches=1)
        batches = list(ds)
        assert len(batches) == 1
        assert int(batches[0].graph_mask().sum()) == 3

    def test_oversized_graphs_filtered(self):
        ds = _make_streaming_dataset(
            batch_size=1,
            max_n_node=10,
            max_n_edge=90,
            raise_exc_if_graphs_discarded=False,
            num_batches=0,
        )
        assert list(ds) == []

    def test_subset_slice(self):
        ds = _make_streaming_dataset(n_graphs=3)
        sub = ds.subset(slice(0, 2))
        assert sub.number_of_graphs() == 2


class TestStreamingResume:
    def test_resume_slices_without_rereading_prefix(self):
        """With raise_exc_if_graphs_discarded=True, skip via MapDataset.slice."""
        systems = _aspirin_systems(3)

        class _CountingDataset(ChemicalSystemsDataset):
            def __init__(self, systems: list[ChemicalSystem]):
                self._systems = list(systems)
                self.accessed: list[int] = []

            def __len__(self) -> int:
                return len(self._systems)

            def __getitem__(self, index: SupportsIndex) -> ChemicalSystem:
                idx = index.__index__()
                self.accessed.append(idx)
                return self._systems[idx]

        counted = _CountingDataset(systems)
        ds = _make_streaming_dataset(
            grain.MapDataset.source(counted),
            batch_size=1,
            raise_exc_if_graphs_discarded=True,
            num_batches=3,
        )

        full_energies = [float(b.globals.energy[0]) for b in ds]
        assert len(full_energies) == 3

        counted.accessed.clear()
        ds.state = ds.state.replace(num_graphs_processed=jnp.int32(1))
        resumed = list(ds)
        assert counted.accessed == [1, 2]
        assert [float(b.globals.energy[0]) for b in resumed] == full_energies[1:]
