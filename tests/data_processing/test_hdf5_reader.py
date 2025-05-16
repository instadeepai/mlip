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

import jax
import jraph
import numpy as np
import pytest

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.hdf5_reader import Hdf5Reader
from mlip.data.configs import ChemicalSystemsReaderConfig, GraphDatasetBuilderConfig
from mlip.data.graph_dataset_builder import (
    DatasetsHaveNotBeenProcessedError,
    GraphDataset,
    GraphDatasetBuilder,
    PrefetchIterator,
)

DATA_DIR = Path(__file__).parent.parent / "data"
SPICE_SMALL_HDF5_PATH = DATA_DIR / "spice2-1000_429_md_0-1.hdf5"


@pytest.mark.parametrize("train_num_to_load", [None, 1])
def test_hdf5_reading_works_correctly_with_two_hdf5s(train_num_to_load):
    reader_config = ChemicalSystemsReaderConfig(
        train_dataset_paths=[
            SPICE_SMALL_HDF5_PATH.resolve(),
            SPICE_SMALL_HDF5_PATH.resolve(),
        ],
        valid_dataset_paths=None,
        test_dataset_paths=None,
        train_num_to_load=train_num_to_load,
        valid_num_to_load=None,
        test_num_to_load=None,
    )
    reader = Hdf5Reader(config=reader_config)
    train_systems, valid_systems, test_systems = reader.load()

    assert len(valid_systems) == 0
    assert len(test_systems) == 0

    if train_num_to_load is None:
        train_num_to_load = 2
    # because from each hdf5 file will load train_num_to_load rows
    assert len(train_systems) == train_num_to_load * 2

    for system in train_systems:
        assert isinstance(system, ChemicalSystem)
        expected_idx = [2, 2, 2, 4, 2, 2, 1, 1, 1, 3, 0, 0, 0]
        assert list(system.atomic_species) == expected_idx
        expected_num = [8, 8, 8, 16, 8, 8, 6, 6, 6, 15, 1, 1, 1]
        assert list(system.atomic_numbers) == expected_num
        assert system.pbc == (False, False, False)
        assert np.all(system.cell == 0.0)

    assert train_systems[0].energy == pytest.approx(-33533.58818835873)
    assert train_systems[0].positions[0][1] == pytest.approx(-0.03931140731653776)
    assert train_systems[0].forces[1][0] == pytest.approx(0.9863849542696035)


@pytest.mark.parametrize("use_formation_energies", [True, False])
def test_builder_works_correctly(use_formation_energies):
    n_examples = 2
    batch_size = 2
    max_n_node = 30
    max_n_edge = 90
    graph_cutoff_angstrom = 5.0
    reader_config = ChemicalSystemsReaderConfig(
        train_dataset_paths=[SPICE_SMALL_HDF5_PATH.resolve()],
        valid_dataset_paths=None,
        test_dataset_paths=None,
        train_num_to_load=n_examples,
        valid_num_to_load=None,
        test_num_to_load=None,
    )
    builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=graph_cutoff_angstrom,
        use_formation_energies=use_formation_energies,
        max_n_node=max_n_node,
        max_n_edge=max_n_edge,
        batch_size=batch_size,
        num_batch_prefetch=1,
        batch_prefetch_num_devices=1,
    )
    reader = Hdf5Reader(config=reader_config)
    builder = GraphDatasetBuilder(reader, builder_config)

    with pytest.raises(DatasetsHaveNotBeenProcessedError):
        dataset_info = builder.dataset_info
    with pytest.raises(DatasetsHaveNotBeenProcessedError):
        splits = builder.get_splits()

    builder.prepare_datasets()
    splits = builder.get_splits()
    for i in range(n_examples):
        assert isinstance(splits[i], GraphDataset)

    assert len(splits[1].graphs) == 0
    assert len(splits[2].graphs) == 0

    assert len(splits[0].graphs) == n_examples
    assert len(splits[0]) == 1

    random.seed(42)
    batch = next(iter(splits[0]))
    assert isinstance(batch, jraph.GraphsTuple)

    num_nodes, num_edges = max_n_node * batch_size + 1, max_n_edge * batch_size * 2
    assert batch.nodes.positions.shape == (num_nodes, 3)
    assert batch.edges.shifts.shape == (num_edges, 3)
    assert list(batch.globals.weight) == pytest.approx([1.0, 1.0, 0.0])
    assert len(batch.senders) == num_edges
    assert len(batch.receivers) == num_edges
    assert list(batch.n_node) == [13, 13, 35]
    assert list(batch.n_edge) == [150, 150, 60]

    dataset_info = builder.dataset_info
    expected_e0s = {
        1: -2235.5844672644207,
        6: -2235.584467264422,
        8: -3725.97411210737,
        15: -745.194822421474,
        16: -745.194822421474,
    }
    assert dataset_info.atomic_energies_map == pytest.approx(expected_e0s)

    expected_e = [-33533.58818835873, -33533.94582957392, 0.0]
    if use_formation_energies:
        to_subtract = (
            3 * expected_e0s[1]
            + 3 * expected_e0s[6]
            + 5 * expected_e0s[8]
            + expected_e0s[15]
            + expected_e0s[16]
        )
        expected_e = [energy - to_subtract for energy in expected_e]
        expected_e[-1] = 0.0
    assert list(batch.globals.energy) == pytest.approx(expected_e, abs=1e-4)

    assert dataset_info.avg_num_neighbors == pytest.approx(11.538461538461538)
    assert dataset_info.avg_r_min_angstrom == pytest.approx(1.0525181293487549)
    assert dataset_info.cutoff_distance_angstrom == graph_cutoff_angstrom
    assert dataset_info.scaling_mean == 0.0
    assert dataset_info.scaling_stdev == 1.0

    splits = builder.get_splits(prefetch=True, devices=jax.devices())
    for i in range(n_examples):
        assert isinstance(splits[i], PrefetchIterator)
