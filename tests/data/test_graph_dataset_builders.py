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

import random
from pathlib import Path

import numpy as np
import pytest

from mlip.data import DatasetInfo
from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.data.configs import GraphDatasetBuilderConfig
from mlip.data.graph_dataset_builder import (
    BuilderMode,
    GraphDataset,
    GraphDatasetBuilder,
    PrefetchIterator,
    SingleGraphDatasetBuilder,
)
from mlip.data.helpers.exceptions import DatasetsHaveNotBeenProcessedError
from mlip.graph import Graph
from mlip.graph.batching_helpers import batch_graphs, homogenize_graph_fields
from mlip.utils.multihost import create_device_mesh

DATA_DIR = Path(__file__).parent.parent / "sample_data"
SMALL_ASPIRIN_DATASET_PATH = DATA_DIR / "small_aspirin_test.xyz"
SMALL_ASPIRIN_WITH_CHARGE_SPIN_PATH = DATA_DIR / "small_aspirin_with_charge_spin.xyz"
SMALL_ASPIRIN_UNSEEN_ATOMS_DATASET_PATH = (
    DATA_DIR / "small_aspirin_test_unseen_atoms.xyz"
)
SMALL_MP_DATASET_PATH = DATA_DIR / "small_materials_test.extxyz"

CUTOFF_ANGSTROM = 6


@pytest.mark.parametrize("use_formation_energies", [True, False])
def test_single_graph_dataset_builder_works_correctly(use_formation_energies):
    builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=2.0,
        max_n_node=30,
        max_n_edge=90,
        batch_size=5,
        num_batch_prefetch_host=1,
        num_batch_prefetch_device=1,
        use_formation_energies=use_formation_energies,
    )

    reader = ExtxyzReader(
        filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(),
        num_to_load=3,
    )

    graph_dataset_builder = SingleGraphDatasetBuilder(
        reader, builder_config, dataset_info=True, shuffle=True
    )

    with pytest.raises(DatasetsHaveNotBeenProcessedError):
        dataset_info = graph_dataset_builder.dataset_info

    dataset = graph_dataset_builder.get_dataset(prefetch=False)
    assert isinstance(dataset, GraphDataset)
    assert len(dataset.graphs) == 3
    assert len(dataset) == 1

    random.seed(42)
    batch = next(iter(dataset))
    assert isinstance(batch, Graph)

    num_nodes, num_edges = 30 * 5 + 1, 90 * 5 * 2
    assert batch.nodes.positions.shape == (num_nodes, 3)
    assert batch.edges.shifts.shape == (num_edges, 3)
    assert list(batch.globals.weight) == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert len(batch.senders) == num_edges
    assert len(batch.receivers) == num_edges
    assert list(batch.n_node) == [21, 21, 21, 88, 0, 0]
    assert list(batch.n_edge) == [50, 50, 50, 750, 0, 0]

    dataset_info = graph_dataset_builder.dataset_info
    expected_e0s = {1: -875.4269754478838, 6: -984.8553473788693, 8: -437.7134877239419}
    assert dataset_info.atomic_energies_map == pytest.approx(expected_e0s)

    expected_e = [-17618.0474, -17617.8269, -17618.0293, 0.0, 0.0, 0.0]
    if use_formation_energies:  # values are much smaller
        expected_e = [-0.0795407, 0.140974, -0.061433, 0.0, 0.0, 0.0]

    assert list(batch.globals.energy) == pytest.approx(expected_e, abs=1e-4)

    assert dataset_info.avg_num_neighbors == pytest.approx(2.3809523809)
    assert dataset_info.avg_r_min_angstrom == pytest.approx(0.96318441629)
    assert dataset_info.graph_cutoff_angstrom == 2.0
    assert dataset_info.scaling_mean == 0.0
    assert dataset_info.scaling_stdev == 1.0
    assert dataset_info.dataset_name is None

    dataset = graph_dataset_builder.get_dataset(
        prefetch=True, mesh=create_device_mesh()
    )
    isinstance(dataset, PrefetchIterator)

    dataset = graph_dataset_builder.get_dataset(prefetch=True, mesh=None)
    assert isinstance(dataset, PrefetchIterator)


def test_single_graph_dataset_builder_preserves_charge_and_spin_multiplicity():
    builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=2.0,
        max_n_node=30,
        max_n_edge=90,
        batch_size=2,
    )
    reader = ExtxyzReader(
        filepaths=SMALL_ASPIRIN_WITH_CHARGE_SPIN_PATH.resolve(),
        property_name_mapping={"spin_multiplicity": "spin"},
    )
    graph_dataset_builder = SingleGraphDatasetBuilder(
        reader, builder_config, dataset_info=True, shuffle=False
    )

    dataset = graph_dataset_builder.get_dataset(prefetch=False)
    batch = next(iter(dataset))

    charges = np.asarray(batch.globals.charge)
    spin_multiplicities = np.asarray(batch.globals.spin_multiplicity)
    assert np.issubdtype(charges.dtype, np.integer)
    assert np.issubdtype(spin_multiplicities.dtype, np.integer)
    assert charges.tolist() == [0, 1, 0]
    assert spin_multiplicities.tolist() == [1, 2, 0]


@pytest.mark.parametrize("use_formation_energies", [True, False])
def test_graph_dataset_builder_training_mode(use_formation_energies):
    mode = "training"
    builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=2.0,
        max_n_node=30,
        max_n_edge=90,
        batch_size=5,
        num_batch_prefetch_host=1,
        num_batch_prefetch_device=1,
        use_formation_energies=use_formation_energies,
    )

    readers = {
        "train": ExtxyzReader(
            filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
        ),
        "valid": ExtxyzReader(
            filepaths=SMALL_ASPIRIN_UNSEEN_ATOMS_DATASET_PATH.resolve(),
        ),
        "test": ExtxyzReader(
            filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
        ),
    }

    graph_dataset_builder = GraphDatasetBuilder(readers, builder_config, mode=mode)

    datasets = graph_dataset_builder.get_datasets(prefetch=False)

    for i in range(len(datasets)):
        assert isinstance(list(datasets.values())[i], GraphDataset)

    assert len(datasets["train"].graphs) == 3
    assert len(datasets["valid"].graphs) == 1  # Unseen atoms gets filtered out
    assert len(datasets["test"].graphs) == 3
    assert len(datasets["train"]) == 1  # Single batch

    random.seed(42)
    batch = next(iter(datasets["train"]))
    assert isinstance(batch, Graph)

    num_nodes, num_edges = 30 * 5 + 1, 90 * 5 * 2
    assert batch.nodes.positions.shape == (num_nodes, 3)
    assert batch.edges.shifts.shape == (num_edges, 3)
    assert list(batch.globals.weight) == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert len(batch.senders) == num_edges
    assert len(batch.receivers) == num_edges
    assert list(batch.n_node) == [21, 21, 21, 88, 0, 0]
    assert list(batch.n_edge) == [50, 50, 50, 750, 0, 0]

    dataset_info = graph_dataset_builder.dataset_info
    expected_e0s = {1: -875.4269754478838, 6: -984.8553473788693, 8: -437.7134877239419}
    assert dataset_info.atomic_energies_map == pytest.approx(expected_e0s)

    expected_e = [-17618.0474, -17617.8269, -17618.0293, 0.0, 0.0, 0.0]
    if use_formation_energies:  # values are much smaller
        expected_e = [-0.0795407, 0.140974, -0.061433, 0.0, 0.0, 0.0]
    assert list(batch.globals.energy) == pytest.approx(expected_e, abs=1e-4)

    assert dataset_info.avg_num_neighbors == pytest.approx(2.3809523809)
    assert dataset_info.avg_r_min_angstrom == pytest.approx(0.96318441629)
    assert dataset_info.graph_cutoff_angstrom == 2.0
    assert dataset_info.scaling_mean == 0.0
    assert dataset_info.scaling_stdev == 1.0

    assert dataset_info.dataset_name is None

    datasets = graph_dataset_builder.get_datasets(
        prefetch=True, mesh=create_device_mesh()
    )
    splits = [*datasets.values()]
    for i in range(3):
        assert isinstance(splits[i], PrefetchIterator)


@pytest.mark.parametrize("use_dataset_info", [True, False])
def test_graph_dataset_builder_custom_mode(use_dataset_info, dataset_info):
    mode = "custom"
    if use_dataset_info:
        graph_cutoff_angstrom = dataset_info.graph_cutoff_angstrom
    else:
        graph_cutoff_angstrom = 2.0
    builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=graph_cutoff_angstrom,
        max_n_node=30,
        max_n_edge=90,
        batch_size=5,
        num_batch_prefetch_host=1,
        num_batch_prefetch_device=1,
    )

    readers = {
        "a": ExtxyzReader(
            filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
        ),
        "b": ExtxyzReader(
            filepaths=SMALL_ASPIRIN_UNSEEN_ATOMS_DATASET_PATH.resolve(),
        ),
        "c": ExtxyzReader(
            filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=5
        ),
    }
    if use_dataset_info:
        preset_dataset_info = dataset_info
    else:
        preset_dataset_info = None

    graph_dataset_builder = GraphDatasetBuilder(
        readers, builder_config, mode=mode, dataset_info=preset_dataset_info
    )

    # Pre-build: CUSTOM mode returns the preset directly; absent a preset it
    # still raises because the post-build value is computed from the graphs.
    if use_dataset_info:
        assert graph_dataset_builder.dataset_info is dataset_info
    else:
        with pytest.raises(DatasetsHaveNotBeenProcessedError):
            _ = graph_dataset_builder.dataset_info

    datasets = graph_dataset_builder.get_datasets(prefetch=False)
    for i in range(len(datasets)):
        assert isinstance(list(datasets.values())[i], GraphDataset)

    assert len(datasets["a"].graphs) == 3
    assert len(datasets["b"].graphs) == 2
    assert len(datasets["c"].graphs) == 5
    assert len(datasets["a"]) == 1  # Single batch

    random.seed(42)
    batch = next(iter(datasets["a"]))
    assert isinstance(batch, Graph)

    _dataset_info = graph_dataset_builder.dataset_info

    if not use_dataset_info:
        expected_e0s = {
            1: -875.4269754478838,
            6: -984.8553473788693,
            8: -437.7134877239419,
        }
        assert _dataset_info["a"].atomic_energies_map == pytest.approx(expected_e0s)

        assert _dataset_info["a"].avg_num_neighbors == pytest.approx(2.3809523809)

        assert _dataset_info["a"].dataset_name is None

    else:
        assert _dataset_info.atomic_energies_map == dataset_info.atomic_energies_map

    datasets = graph_dataset_builder.get_datasets(
        prefetch=True, mesh=create_device_mesh()
    )
    splits = [*datasets.values()]
    for i in range(3):
        assert isinstance(splits[i], PrefetchIterator)


def test_dataset_info_allowed_atomic_numbers_with_list_fields():
    """allowed_atomic_numbers returns the union of all maps when fields are lists."""
    info = DatasetInfo(
        atomic_energies_map=[{1: -1.0, 6: -6.0}, {1: -1.0, 8: -8.0, 12: -12.0}],
        graph_cutoff_angstrom=3.0,
    )
    assert info.allowed_atomic_numbers == [1, 6, 8, 12]


class TestCombineDatasetInfosNoneSafety:
    """Regression tests for merging `DatasetInfo`s that have `num_graphs=None`
    (common in unit-test fixtures and pre-scan datasets) or missing optional
    statistics."""

    def test_multi_case_handles_num_graphs_none(self):
        """Multi-pretraining path must not crash when any dataset has
        `num_graphs=None`."""
        infos = [
            DatasetInfo(
                dataset_name="a",
                atomic_energies_map={1: -1.0},
                graph_cutoff_angstrom=3.0,
                avg_num_neighbors=2.0,
                avg_r_min_angstrom=0.9,
                num_graphs=None,
            ),
            DatasetInfo(
                dataset_name="b",
                atomic_energies_map={6: -6.0},
                graph_cutoff_angstrom=3.0,
                avg_num_neighbors=4.0,
                avg_r_min_angstrom=1.1,
                num_graphs=10,
            ),
        ]
        merged = GraphDatasetBuilder._combine_dataset_infos_in_multi_case(infos)
        # `num_graphs=None` entries are skipped in the sum; the resulting sum
        # is over the datasets that do have counts.
        assert merged.num_graphs == 10
        # With one `num_graphs=None` we fall back to an equal-weight mean
        # across contributing datasets rather than crashing.
        assert merged.avg_num_neighbors == pytest.approx(3.0)
        assert merged.avg_r_min_angstrom == pytest.approx(1.0)
        assert merged.dataset_name == ["a", "b"]

    def test_multi_case_all_num_graphs_none_returns_none(self):
        """With every `num_graphs` unset, the merged info has
        `num_graphs=None` and statistics fall back to equal-weight means."""
        infos = [
            DatasetInfo(
                dataset_name="a",
                atomic_energies_map={1: -1.0},
                graph_cutoff_angstrom=3.0,
                avg_num_neighbors=2.0,
                avg_r_min_angstrom=0.9,
            ),
            DatasetInfo(
                dataset_name="b",
                atomic_energies_map={6: -6.0},
                graph_cutoff_angstrom=3.0,
                avg_num_neighbors=4.0,
                avg_r_min_angstrom=None,
            ),
        ]
        merged = GraphDatasetBuilder._combine_dataset_infos_in_multi_case(infos)
        assert merged.num_graphs is None
        assert merged.avg_num_neighbors == pytest.approx(3.0)
        # Only one dataset has `avg_r_min_angstrom`, so it passes through.
        assert merged.avg_r_min_angstrom == pytest.approx(0.9)

    def test_finetuning_case_handles_num_graphs_none(self):
        """Fine-tuning path sums defined `num_graphs` and keeps pretrained
        scalar fields."""
        infos = [
            DatasetInfo(
                dataset_name="replay",
                atomic_energies_map={1: -1.0, 6: -6.0},
                graph_cutoff_angstrom=3.0,
                num_graphs=None,
            ),
            DatasetInfo(
                dataset_name="new",
                atomic_energies_map={1: -1.0, 8: -8.0},
                graph_cutoff_angstrom=3.0,
                num_graphs=5,
            ),
        ]
        merged = GraphDatasetBuilder._combine_dataset_infos_if_finetuning(infos)
        assert merged.num_graphs == 5
        assert merged.dataset_name == ["replay", "new"]
        assert merged.atomic_energies_map == [
            {1: -1.0, 6: -6.0},
            {1: -1.0, 8: -8.0},
        ]

    def test_multi_case_long_range_cutoff_propagates_max_or_none(self):
        """If any dataset has no long-range cutoff, the merged info must
        disable long-range too (otherwise we'd claim a cutoff that is not
        honoured by every subset). If all datasets have one, take the max."""
        base = {"atomic_energies_map": {1: -1.0}, "graph_cutoff_angstrom": 3.0}

        # Any None → merged None
        infos_mixed = [
            DatasetInfo(**base, dataset_name="a", long_range_cutoff_angstrom=6.0),
            DatasetInfo(**base, dataset_name="b", long_range_cutoff_angstrom=None),
        ]
        merged_mixed = GraphDatasetBuilder._combine_dataset_infos_in_multi_case(
            infos_mixed
        )
        assert merged_mixed.long_range_cutoff_angstrom is None

        # All set → max
        infos_all = [
            DatasetInfo(**base, dataset_name="a", long_range_cutoff_angstrom=4.0),
            DatasetInfo(**base, dataset_name="b", long_range_cutoff_angstrom=7.5),
        ]
        merged_all = GraphDatasetBuilder._combine_dataset_infos_in_multi_case(infos_all)
        assert merged_all.long_range_cutoff_angstrom == 7.5

    def test_multi_case_total_charge_set_unions_non_none(self):
        """Merged `total_charge_set` is the union across datasets that
        provide one; datasets with `None` are skipped. If every dataset has
        `None`, the merged info keeps `None`."""
        base = {"atomic_energies_map": {1: -1.0}, "graph_cutoff_angstrom": 3.0}

        # Union skipping a None entry
        infos = [
            DatasetInfo(**base, dataset_name="a", total_charge_set={0, 1}),
            DatasetInfo(**base, dataset_name="b", total_charge_set=None),
            DatasetInfo(**base, dataset_name="c", total_charge_set={-1, 0}),
        ]
        merged = GraphDatasetBuilder._combine_dataset_infos_in_multi_case(infos)
        assert merged.total_charge_set == {-1, 0, 1}

        # All None → None
        infos_all_none = [
            DatasetInfo(**base, dataset_name="a"),
            DatasetInfo(**base, dataset_name="b"),
        ]
        merged_all_none = GraphDatasetBuilder._combine_dataset_infos_in_multi_case(
            infos_all_none
        )
        assert merged_all_none.total_charge_set is None


class TestGraphDatasetBuilderMultiMode:
    """Tests for GraphDatasetBuilder in MULTI mode."""

    @pytest.fixture()
    def builder_config(self, dataset_info):
        return GraphDatasetBuilderConfig(
            graph_cutoff_angstrom=dataset_info.graph_cutoff_angstrom,
            max_n_node=30,
            max_n_edge=90,
            batch_size=5,
            num_batch_prefetch_host=1,
            num_batch_prefetch_device=1,
        )

    @pytest.fixture()
    def builder_config_2A(self):
        return GraphDatasetBuilderConfig(
            graph_cutoff_angstrom=2.0,
            max_n_node=30,
            max_n_edge=90,
            batch_size=5,
            num_batch_prefetch_host=1,
            num_batch_prefetch_device=1,
        )

    def test_replay_always_gets_dataset_idx_zero(self, builder_config, dataset_info):
        """Replay gets dataset_idx=0 even when it appears second in the dict."""
        readers = {
            "omat20": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
            },
            "replay": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=2
                ),
            },
        }

        builder = GraphDatasetBuilder(
            readers, builder_config, mode="multi", dataset_info=dataset_info
        )
        datasets = builder.get_datasets(prefetch=False)

        assert builder.dataset_info.dataset_name == ["replay", "omat20"]

        # Replay graphs (2) should have dataset_idx=0,
        # omat20 graphs (3) should have dataset_idx=1.
        for graph in datasets["train"].graphs:
            idx = graph.globals.dataset_idx.item()
            assert idx in (0, 1)

        replay_graphs = [
            g for g in datasets["train"].graphs if g.globals.dataset_idx.item() == 0
        ]
        omat_graphs = [
            g for g in datasets["train"].graphs if g.globals.dataset_idx.item() == 1
        ]
        assert len(replay_graphs) == 2
        assert len(omat_graphs) == 3

    def test_dataset_name_ordering(self, builder_config, dataset_info):
        """dataset_name in DatasetInfo has replay first, others in order."""
        readers = {
            "omat20": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=2
                ),
            },
            "replay": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=2
                ),
            },
        }

        builder = GraphDatasetBuilder(
            readers, builder_config, mode="multi", dataset_info=dataset_info
        )
        builder.get_datasets(prefetch=False)

        assert builder.dataset_info.dataset_name == ["replay", "omat20"]

    def test_dataset_name_ordering_without_replay(self, builder_config_2A):
        """Without replay, datasets are ordered by insertion order."""
        readers = {
            "oc20": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=2
                ),
            },
            "omat20": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=2
                ),
            },
        }

        builder = GraphDatasetBuilder(readers, builder_config_2A, mode="multi")
        builder.get_datasets(prefetch=False)

        assert builder.dataset_info.dataset_name == ["oc20", "omat20"]

    def test_asymmetric_splits(self, builder_config_2A):
        """One dataset has {train, valid, test}, the other only {train}."""
        readers = {
            "full": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
                "valid": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=2
                ),
                "test": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=1
                ),
            },
            "train_only": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=4
                ),
            },
        }

        builder = GraphDatasetBuilder(readers, builder_config_2A, mode="multi")
        datasets = builder.get_datasets(prefetch=False)

        # Train contains graphs from both datasets.
        assert len(datasets["train"].graphs) == 3 + 4
        train_idxs = {g.globals.dataset_idx.item() for g in datasets["train"].graphs}
        assert train_idxs == {0, 1}

        # Valid and test only contain graphs from "full" (dataset_idx=0).
        assert len(datasets["valid"].graphs) == 2
        assert all(g.globals.dataset_idx.item() == 0 for g in datasets["valid"].graphs)

        assert len(datasets["test"].graphs) == 1
        assert all(g.globals.dataset_idx.item() == 0 for g in datasets["test"].graphs)

    def test_dataset_info_list_structure(self, builder_config_2A):
        """dataset_info has list-valued fields for dataset_name and
        atomic_energies_map, scalar fields for the rest."""
        readers = {
            "ds_a": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
            },
            "ds_b": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
            },
        }

        builder = GraphDatasetBuilder(readers, builder_config_2A, mode="multi")
        builder.get_datasets(prefetch=False)

        info = builder.dataset_info
        assert isinstance(info, DatasetInfo)
        assert isinstance(info.dataset_name, list)
        assert len(info.dataset_name) == 2
        assert isinstance(info.atomic_energies_map, list)
        assert len(info.atomic_energies_map) == 2

    def test_replay_head_preserves_preset_dataset_info(
        self, builder_config, dataset_info
    ):
        """In finetuning (with replay), the replay head preserves preset values."""
        readers = {
            "omat20": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
            },
            "replay": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
            },
        }

        builder = GraphDatasetBuilder(
            readers, builder_config, mode="multi", dataset_info=dataset_info
        )
        builder.get_datasets(prefetch=False)

        info = builder.dataset_info
        # Replay is always index 0 in the combined info.
        assert info.atomic_energies_map[0] == dataset_info.atomic_energies_map
        assert info.graph_cutoff_angstrom == dataset_info.graph_cutoff_angstrom
        assert info.avg_num_neighbors == dataset_info.avg_num_neighbors
        assert info.scaling_mean == dataset_info.scaling_mean
        assert info.scaling_stdev == dataset_info.scaling_stdev

    def test_unseen_atoms_in_non_train_split_raises_error(self, builder_config_2A):
        """Non-train splits with unseen atoms raise an error in multi mode."""
        readers = {
            "ds_a": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
                "valid": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_UNSEEN_ATOMS_DATASET_PATH.resolve(),
                ),
            },
            "ds_b": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
            },
        }

        builder = GraphDatasetBuilder(readers, builder_config_2A, mode="multi")
        with pytest.raises(ValueError, match="unseen"):
            builder.get_datasets(prefetch=False)

    def test_merged_graph_counts(self, builder_config_2A):
        """Merged train split has the sum of graphs from all datasets."""
        num_a, num_b = 3, 5

        readers = {
            "ds_a": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=num_a
                ),
            },
            "ds_b": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=num_b
                ),
            },
        }

        builder = GraphDatasetBuilder(readers, builder_config_2A, mode="multi")
        datasets = builder.get_datasets(prefetch=False)

        assert len(datasets["train"].graphs) == num_a + num_b

    def test_replay_with_unseen_atoms_raises_error(self, builder_config, dataset_info):
        """Replay data with atoms not in the preset DatasetInfo should error."""
        readers = {
            "replay": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_UNSEEN_ATOMS_DATASET_PATH.resolve(),
                ),
            },
            "omat20": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
            },
        }

        builder = GraphDatasetBuilder(
            readers, builder_config, mode="multi", dataset_info=dataset_info
        )
        with pytest.raises(ValueError, match="unseen"):
            builder.get_datasets(prefetch=False)


class TestHomogenizeGraphFields:
    """Tests for homogenize_graph_fields."""

    def test_fills_missing_stress(self, make_customizable_graph):
        """Graphs without stress get NaN stress after homogenization.

        NaN (not zero) is used as a sentinel so downstream loss and eval-metric
        code can mask out samples whose dataset did not provide the field.
        """
        g_with_stress = make_customizable_graph(3, 3)
        assert g_with_stress.globals.stress is not None
        g_without_stress = make_customizable_graph(2, 2).replace_globals(stress=None)

        result = homogenize_graph_fields([g_with_stress, g_without_stress])

        filled = result[1]
        assert filled.globals.stress is not None
        assert filled.globals.stress.shape == (1, 3, 3)
        assert np.all(np.isnan(filled.globals.stress))

    def test_homogenized_graphs_can_batch(self, make_customizable_graph):
        """After homogenization, graphs with heterogeneous fields can be batched."""
        g_with_stress = make_customizable_graph(3, 3)
        g_without_stress = make_customizable_graph(2, 2).replace_globals(stress=None)

        result = homogenize_graph_fields([g_with_stress, g_without_stress])
        batched = batch_graphs(result)
        assert batched.globals.stress.shape == (2, 3, 3)

    def test_no_op_when_fields_match(self, make_customizable_graph):
        """Homogenization is a no-op when all fields are uniformly None."""
        g1 = make_customizable_graph(3, 3).replace_globals(
            stress=None, pressure=None, spin_multiplicity=None
        )
        g2 = make_customizable_graph(2, 2).replace_globals(
            stress=None, pressure=None, spin_multiplicity=None
        )

        result = homogenize_graph_fields([g1, g2])

        for key in ["stress", "pressure", "spin_multiplicity"]:
            assert getattr(result[0].globals, key) is None, key
            assert getattr(result[1].globals, key) is None, key

    def test_fills_missing_spin_multiplicity_only_for_mixed_batches(
        self, make_customizable_graph
    ):
        spin_array = np.array([2], dtype=np.int32)
        g_with_spin = make_customizable_graph(3, 3).replace_globals(
            spin_multiplicity=spin_array
        )
        g_without_spin = make_customizable_graph(2, 2)

        result = homogenize_graph_fields([g_with_spin, g_without_spin])

        assert np.all(result[0].globals.spin_multiplicity == spin_array)
        assert np.all(np.isnan(result[1].globals.spin_multiplicity))

        batched = batch_graphs(result)
        assert np.asarray(batched.globals.spin_multiplicity)[0] == spin_array[0]
        assert np.isnan(np.asarray(batched.globals.spin_multiplicity)[1])


class TestGraphDatasetBuilderValidation:
    """Tests for GraphDatasetBuilder.__init__ validation."""

    @pytest.fixture()
    def builder_config(self):
        return GraphDatasetBuilderConfig(
            graph_cutoff_angstrom=2.0,
            max_n_node=30,
            max_n_edge=90,
            batch_size=5,
            num_batch_prefetch_host=1,
            num_batch_prefetch_device=1,
        )

    @pytest.fixture()
    def flat_readers(self):
        return {
            "train": ExtxyzReader(
                filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
            ),
            "valid": ExtxyzReader(
                filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
            ),
        }

    @pytest.fixture()
    def nested_readers(self):
        return {
            "oc20": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
            },
            "omat": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
                "valid": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=2
                ),
            },
        }

    def test_custom_mode_rejects_nested_readers(self, builder_config, nested_readers):
        with pytest.raises(TypeError, match="CUSTOM mode expects a flat"):
            GraphDatasetBuilder(nested_readers, builder_config, mode=BuilderMode.CUSTOM)

    def test_custom_mode_rejects_empty_readers(self, builder_config):
        with pytest.raises(ValueError, match="must not be empty"):
            GraphDatasetBuilder({}, builder_config, mode=BuilderMode.CUSTOM)

    def test_training_mode_rejects_nested_readers(self, builder_config, nested_readers):
        with pytest.raises(TypeError, match="TRAINING mode expects a flat"):
            GraphDatasetBuilder(
                nested_readers, builder_config, mode=BuilderMode.TRAINING
            )

    def test_training_mode_requires_train_key(self, builder_config):
        readers = {
            "valid": ExtxyzReader(
                filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
            ),
        }
        with pytest.raises(ValueError, match="'train' key"):
            GraphDatasetBuilder(readers, builder_config, mode=BuilderMode.TRAINING)

    def test_multi_mode_rejects_flat_readers(self, builder_config, flat_readers):
        with pytest.raises(TypeError, match="MULTI mode expects a nested"):
            GraphDatasetBuilder(flat_readers, builder_config, mode=BuilderMode.MULTI)

    def test_multi_mode_requires_at_least_two_datasets(self, builder_config):
        readers = {
            "oc20": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
            },
        }
        with pytest.raises(ValueError, match="at least 2"):
            GraphDatasetBuilder(readers, builder_config, mode=BuilderMode.MULTI)

    def test_multi_mode_replay_requires_dataset_info(
        self,
        builder_config,
    ):
        readers = {
            "replay": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
            },
            "omat": {
                "train": ExtxyzReader(
                    filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(), num_to_load=3
                ),
            },
        }
        with pytest.raises(ValueError, match="replay.*requires a preset"):
            GraphDatasetBuilder(
                readers, builder_config, mode=BuilderMode.MULTI, dataset_info=None
            )

    def test_preset_dataset_info_checks_cutoff_compatibility(
        self, builder_config, flat_readers
    ):
        mismatched_info = DatasetInfo(
            atomic_energies_map={1: -1.0},
            graph_cutoff_angstrom=99.0,
        )
        with pytest.raises(ValueError, match="inconsistent cutoff"):
            GraphDatasetBuilder(
                flat_readers,
                builder_config,
                mode=BuilderMode.TRAINING,
                dataset_info=mismatched_info,
            )


@pytest.mark.parametrize(
    "batch_size,expected_n_node,expected_n_edge",
    [
        (2, [3, 3, 1], [3, 3, 2]),
        (4, [3, 3, 7, 0, 0], [3, 3, 10, 0, 0]),
    ],
)
def test_autofill_batch_dims_creates_correct_dummy_graphs(
    batch_size, expected_n_node, expected_n_edge, make_customizable_graph
):
    """Test that auto-filled `max_n_X` values result in correct batching."""
    graphs = [make_customizable_graph(n_nodes=3, n_edges=3) for _ in range(2)]
    config = GraphDatasetBuilderConfig(batch_size=batch_size)
    max_n_node, max_n_edge, _ = (
        SingleGraphDatasetBuilder._determine_autofill_batch_dimensions_static(
            graphs, config
        )
    )
    ds = GraphDataset(
        graphs=graphs,
        batch_size=batch_size,
        max_n_node=max_n_node,
        max_n_edge=max_n_edge,
        shuffle=False,
    )
    batch = next(iter(ds))
    assert list(batch.n_node) == expected_n_node
    assert list(batch.n_edge) == expected_n_edge
