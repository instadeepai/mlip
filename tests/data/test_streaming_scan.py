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

from pathlib import Path
from typing import SupportsIndex

import grain
import numpy as np
import pytest

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.chemical_systems_dataset import (
    ChemicalSystemsDataset,
    filter_excluded_indices,
)
from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.data.chemical_systems_readers.hdf5_reader import Hdf5Reader
from mlip.data.configs import GraphDatasetBuilderConfig
from mlip.data.dataset_info import DatasetInfo, compute_dataset_info_from_graphs
from mlip.data.helpers.streaming_scan import scan_chemical_map_dataset
from mlip.data.single_graph_dataset_builder import SingleGraphDatasetBuilder
from mlip.data.streaming_graph_dataset import StreamingGraphDataset
from mlip.graph import Graph

DATA_DIR = Path(__file__).parent.parent / "sample_data"
SMALL_ASPIRIN_DATASET_PATH = DATA_DIR / "small_aspirin_test.xyz"
SMALL_ASPIRIN_WITH_CHARGE_SPIN_PATH = DATA_DIR / "small_aspirin_with_charge_spin.xyz"
SPICE_SMALL_HDF5_PATH = DATA_DIR / "spice2-1000_429_md_0-1.hdf5"


class _ListDataset(ChemicalSystemsDataset):
    def __init__(self, systems: list[ChemicalSystem]):
        self._systems = list(systems)

    def __len__(self) -> int:
        return len(self._systems)

    def __getitem__(self, index: SupportsIndex) -> ChemicalSystem:
        return self._systems[index.__index__()]


def _aspirin_map_dataset(n: int = 3) -> grain.MapDataset:
    systems = ExtxyzReader(
        filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(),
        num_to_load=n,
    ).load()
    return grain.MapDataset.source(_ListDataset(systems))


def _assert_dataset_info_equal(
    streaming: DatasetInfo,
    reference: DatasetInfo,
    *,
    rel: float = 1e-9,
    abs_: float = 1e-9,
) -> None:
    assert streaming.num_graphs == reference.num_graphs
    assert streaming.graph_cutoff_angstrom == reference.graph_cutoff_angstrom
    assert streaming.long_range_cutoff_angstrom == reference.long_range_cutoff_angstrom
    assert streaming.total_charge_set == reference.total_charge_set
    assert streaming.avg_num_neighbors == pytest.approx(
        reference.avg_num_neighbors, rel=rel, abs=abs_
    )
    assert streaming.avg_r_min_angstrom == pytest.approx(
        reference.avg_r_min_angstrom, rel=rel, abs=abs_
    )
    assert set(streaming.atomic_energies_map) == set(reference.atomic_energies_map)
    for z, e0 in reference.atomic_energies_map.items():
        assert streaming.atomic_energies_map[z] == pytest.approx(e0, rel=rel, abs=abs_)


def _assert_batches_equal(stream_batches, mem_batches) -> None:
    assert len(stream_batches) == len(mem_batches)
    for stream_batch, mem_batch in zip(stream_batches, mem_batches, strict=True):
        assert np.asarray(stream_batch.n_node).tolist() == np.asarray(
            mem_batch.n_node
        ).tolist()
        assert np.asarray(stream_batch.globals.energy) == pytest.approx(
            np.asarray(mem_batch.globals.energy), abs=1e-5
        )
        assert np.asarray(stream_batch.nodes.positions) == pytest.approx(
            np.asarray(mem_batch.nodes.positions), abs=1e-5
        )


class TestStreamingScan:
    @pytest.mark.parametrize(
        ("xyz_path", "cutoff", "batch_size", "max_n_node", "max_n_edge"),
        [
            (SMALL_ASPIRIN_DATASET_PATH, 5.0, 6, 30, 50),
            (SMALL_ASPIRIN_DATASET_PATH, 5.0, 2, None, None),
            (SMALL_ASPIRIN_WITH_CHARGE_SPIN_PATH, 2.0, 4, None, None),
        ],
    )
    def test_scan_matches_in_memory_graph_stats(
        self, xyz_path, cutoff, batch_size, max_n_node, max_n_edge
    ):
        reader = ExtxyzReader(filepaths=xyz_path.resolve())
        systems = reader.load()
        chemical_dataset = grain.MapDataset.source(_ListDataset(systems))

        batching = scan_chemical_map_dataset(
            chemical_dataset,
            reader,
            graph_cutoff_angstrom=cutoff,
            long_range_cutoff_angstrom=None,
            preprocessing_fns=None,
            batch_size=batch_size,
            max_n_node=max_n_node,
            max_n_edge=max_n_edge,
            max_n_edge_long_range=None,
        )

        graphs = [
            Graph.from_chemical_system(system, graph_cutoff_angstrom=cutoff)
            for system in systems
        ]
        reference_info = compute_dataset_info_from_graphs(
            graphs,
            graph_cutoff_angstrom=cutoff,
            long_range_cutoff_angstrom=None,
        )
        config = GraphDatasetBuilderConfig(
            graph_cutoff_angstrom=cutoff,
            batch_size=batch_size,
            max_n_node=max_n_node,
            max_n_edge=max_n_edge,
        )
        ref_max_n_node, ref_max_n_edge, ref_max_n_edge_lr = (
            SingleGraphDatasetBuilder._determine_autofill_batch_dimensions_static(
                graphs, config
            )
        )

        _assert_dataset_info_equal(batching.dataset_info, reference_info)
        assert batching.max_n_node == ref_max_n_node
        assert batching.max_n_edge == ref_max_n_edge
        assert batching.max_n_edge_long_range == ref_max_n_edge_lr
        assert batching.exclude_ids == []
        assert batching.num_nodes == sum(int(g.n_node.sum()) for g in graphs)
        assert batching.dataset_info.num_graphs == len(graphs)

    def test_preprocessing_exclude_and_filtered_dataset(self, caplog):
        chemical_dataset = _aspirin_map_dataset(3)
        target_energy = float(chemical_dataset[1].energy)

        def drop_by_energy(systems):
            if systems and abs(float(systems[0].energy) - target_energy) < 1e-6:
                return []
            return systems

        with caplog.at_level("WARNING", logger="mlip"):
            batching = scan_chemical_map_dataset(
                chemical_dataset,
                ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
                graph_cutoff_angstrom=2.0,
                long_range_cutoff_angstrom=None,
                preprocessing_fns=[drop_by_energy],
                batch_size=5,
                max_n_node=30,
                max_n_edge=90,
                max_n_edge_long_range=None,
            )
        assert batching.exclude_ids == [1]
        assert batching.dataset_info.num_graphs == 2
        assert any("preprocessing filters" in r.message for r in caplog.records)

        filtered = filter_excluded_indices(chemical_dataset, batching.exclude_ids)
        assert len(filtered) == 2

        dataset = StreamingGraphDataset(
            filtered,
            batch_size=5,
            max_n_node=batching.max_n_node,
            max_n_edge=batching.max_n_edge,
            graph_cutoff_angstrom=2.0,
            shuffle=False,
            num_nodes=batching.num_nodes,
            num_batches=batching.num_batches,
        )
        batch = next(iter(dataset))
        assert int(batch.graph_mask().sum()) == 2

    def test_batching_info_cache_roundtrip(self, tmp_path):
        chemical_dataset = _aspirin_map_dataset(3)
        reader = ExtxyzReader(
            filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
        )
        kwargs = dict(
            readers=reader,
            graph_cutoff_angstrom=2.0,
            long_range_cutoff_angstrom=None,
            preprocessing_fns=None,
            batch_size=5,
            max_n_node=None,
            max_n_edge=None,
            max_n_edge_long_range=None,
            cache_dir=tmp_path,
        )
        first = scan_chemical_map_dataset(chemical_dataset, **kwargs)
        second = scan_chemical_map_dataset(chemical_dataset, **kwargs)
        assert first.max_n_node == second.max_n_node
        assert first.num_batches == second.num_batches
        assert first.exclude_ids == second.exclude_ids
        assert first.dataset_info.atomic_energies_map == (
            second.dataset_info.atomic_energies_map
        )
        assert first.dataset_info.avg_num_neighbors == pytest.approx(
            second.dataset_info.avg_num_neighbors
        )
        assert list(tmp_path.glob("batching_info_*.json"))


class TestStreamingBuilder:
    @pytest.mark.parametrize(
        "common",
        [
            dict(
                graph_cutoff_angstrom=5.0,
                batch_size=4,
                max_n_node=None,
                max_n_edge=None,
                use_formation_energies=False,
            ),
            dict(
                graph_cutoff_angstrom=5.0,
                batch_size=8,
                max_n_node=20,
                max_n_edge=50,
                use_formation_energies=False,
            ),
            dict(
                graph_cutoff_angstrom=5.0,
                batch_size=4,
                max_n_node=None,
                max_n_edge=None,
                use_formation_energies=True,
            ),
        ],
        ids=["autofill", "preset_caps", "formation_energies"],
    )
    def test_streaming_builder_matches_in_memory(self, common):
        path = SPICE_SMALL_HDF5_PATH.resolve()

        mem_builder = SingleGraphDatasetBuilder(
            Hdf5Reader(filepaths=path),
            GraphDatasetBuilderConfig(keep_in_memory=True, **common),
            dataset_info=True,
            shuffle=False,
        )
        mem_dataset = mem_builder.get_dataset()
        mem_info = mem_builder.dataset_info

        stream_builder = SingleGraphDatasetBuilder(
            Hdf5Reader(filepaths=path),
            GraphDatasetBuilderConfig(keep_in_memory=False, **common),
            dataset_info=True,
            shuffle=False,
        )
        stream_dataset = stream_builder.get_dataset()
        stream_info = stream_builder.dataset_info

        _assert_dataset_info_equal(stream_info, mem_info)
        assert stream_dataset.max_n_node == mem_dataset.max_n_node
        assert stream_dataset.max_n_edge == mem_dataset.max_n_edge
        assert stream_dataset.max_n_edge_long_range == mem_dataset.max_n_edge_long_range
        assert stream_dataset.number_of_graphs() == mem_dataset.number_of_graphs()
        assert len(stream_dataset) == len(mem_dataset)
        if common.get("max_n_node") is not None:
            assert stream_dataset.max_n_node == common["max_n_node"]
            assert stream_dataset.max_n_edge == common["max_n_edge"]

        _assert_batches_equal(list(stream_dataset), list(mem_dataset))
