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

from mlip.data import DatasetInfo
from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.data.configs import ChemicalSystemsReaderConfig, GraphDatasetBuilderConfig
from mlip.data.graph_dataset_builder import (
    DatasetsHaveNotBeenProcessedError,
    GraphDataset,
    GraphDatasetBuilder,
    PrefetchIterator,
)
from mlip.data.helpers import get_edge_relative_vectors
from mlip.data.helpers.atomic_energies import compute_average_e0s_from_graphs
from mlip.data.helpers.atomic_number_table import AtomicNumberTable
from mlip.data.helpers.data_split import (
    DataSplitProportions,
    SplitProportionsInvalidError,
    split_data_by_group,
    split_data_randomly,
    split_data_randomly_by_group,
)
from mlip.data.helpers.graph_creation import create_graph_from_chemical_system

DATA_DIR = Path(__file__).parent.parent / "data"
SMALL_ASPIRIN_DATASET_PATH = DATA_DIR / "small_aspirin_test.xyz"
SMALL_ASPIRIN_UNSEEN_ATOMS_DATASET_PATH = (
    DATA_DIR / "small_aspirin_test_unseen_atoms.xyz"
)
SMALL_MP_DATASET_PATH = DATA_DIR / "small_materials_test.extxyz"

CUTOFF_ANGSTROM = 6


@pytest.mark.parametrize("train_num_to_load", [None, 3])
def test_extxyz_reading_works_correctly(train_num_to_load):
    chemical_systems_reader_config = ChemicalSystemsReaderConfig(
        train_dataset_paths=[str(SMALL_ASPIRIN_DATASET_PATH.resolve())],
        valid_dataset_paths=[str(SMALL_ASPIRIN_UNSEEN_ATOMS_DATASET_PATH.resolve())],
        test_dataset_paths=None,
        train_num_to_load=train_num_to_load,
        valid_num_to_load=None,
        test_num_to_load=None,
    )
    reader = ExtxyzReader(config=chemical_systems_reader_config, data_download_fun=None)
    train_systems, valid_systems, test_systems = reader.load()

    assert len(valid_systems) == 1
    assert len(test_systems) == 0

    if train_num_to_load is None:
        train_num_to_load = 7
    assert len(train_systems) == train_num_to_load

    for system in train_systems:
        assert isinstance(system, ChemicalSystem)
        expected_idx = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        assert list(system.atomic_species) == expected_idx
        expected_num = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
        assert list(system.atomic_numbers) == expected_num
        assert system.pbc == (False, False, False)
        assert np.all(system.cell == 0.0)

    assert train_systems[0].energy == pytest.approx(-17617.826906338443)
    assert train_systems[1].positions[0][1] == pytest.approx(-0.97843255)
    assert train_systems[2].forces[1][0] == pytest.approx(-0.05825649)


@pytest.mark.parametrize("use_formation_energies", [True, False])
def test_graph_dataset_builder_works_correctly(use_formation_energies):
    reader_config = ChemicalSystemsReaderConfig(
        train_dataset_paths=[str(SMALL_ASPIRIN_DATASET_PATH.resolve())],
        valid_dataset_paths=None,
        test_dataset_paths=None,
        train_num_to_load=3,
        valid_num_to_load=None,
        test_num_to_load=None,
    )
    graph_dataset_builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=2.0,
        use_formation_energies=use_formation_energies,
        max_n_node=30,
        max_n_edge=90,
        batch_size=5,
        num_batch_prefetch=1,
        batch_prefetch_num_devices=1,
    )
    reader = ExtxyzReader(config=reader_config)
    graph_dataset_builder = GraphDatasetBuilder(reader, graph_dataset_builder_config)

    with pytest.raises(DatasetsHaveNotBeenProcessedError):
        dataset_info = graph_dataset_builder.dataset_info
    with pytest.raises(DatasetsHaveNotBeenProcessedError):
        splits = graph_dataset_builder.get_splits()

    graph_dataset_builder.prepare_datasets()
    datasets = graph_dataset_builder.get_splits()
    for i in range(3):
        assert isinstance(datasets[i], GraphDataset)

    assert len(datasets[1].graphs) == 0
    assert len(datasets[2].graphs) == 0

    assert len(datasets[0].graphs) == 3
    assert len(datasets[0]) == 1

    random.seed(42)
    batch = next(iter(datasets[0]))
    assert isinstance(batch, jraph.GraphsTuple)

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

    expected_e = [-17618.0293, -17618.0474, -17617.8269, 0.0, 0.0, 0.0]
    if use_formation_energies:
        to_subtract = 8 * expected_e0s[1] + 9 * expected_e0s[6] + 4 * expected_e0s[8]
        expected_e = [energy - to_subtract for energy in expected_e]
        for idx in range(3, 6):
            expected_e[idx] = 0.0
    assert list(batch.globals.energy) == pytest.approx(expected_e, abs=1e-4)

    assert dataset_info.avg_num_neighbors == pytest.approx(2.3809523809)
    assert dataset_info.avg_r_min_angstrom == pytest.approx(0.96318441629)
    assert dataset_info.cutoff_distance_angstrom == 2.0
    assert dataset_info.scaling_mean == 0.0
    assert dataset_info.scaling_stdev == 1.0

    splits = graph_dataset_builder.get_splits(prefetch=True, devices=jax.devices())
    for i in range(3):
        assert isinstance(splits[i], PrefetchIterator)


def test_graph_dataset_builder_with_dataset_info():
    # Create a `DatasetInfo` with more elements : {H, He, C, N, O} >= {H, C, O}
    dataset_info = DatasetInfo(
        atomic_energies_map=dict.fromkeys([1, 2, 6, 7, 8], 0.0),
        cutoff_distance_angstrom=2.0,  # must match config else ValueError is raised
    )
    # Initialize reader / builder configs
    reader_config = ChemicalSystemsReaderConfig(
        train_dataset_paths=[str(SMALL_ASPIRIN_DATASET_PATH.resolve())],
        valid_dataset_paths=None,
        test_dataset_paths=None,
        train_num_to_load=3,
        valid_num_to_load=None,
        test_num_to_load=None,
    )
    builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=2.0,
        use_formation_energies=False,
        max_n_node=30,
        max_n_edge=90,
        batch_size=5,
        num_batch_prefetch=1,
        batch_prefetch_num_devices=1,
    )
    reader = ExtxyzReader(config=reader_config)
    # Pass `DatasetInfo` from the outside.
    graph_dataset_builder = GraphDatasetBuilder(reader, builder_config, dataset_info)
    graph_dataset_builder.prepare_datasets()
    dataset, _, _ = graph_dataset_builder.get_splits()
    # Check that species lie in {0, 2, 4} :
    #   - aspirin only contains H, C, O
    #   - He and N not encountered
    for graph in dataset:
        species = graph.nodes.species[: sum(graph.n_node[:-1])]
        assert set(species) == {0, 2, 4}


def test_atomic_energies_calculation_works():
    e0s = {1: -0.123, 6: -4.34, 8: -6.54}
    h2o = ChemicalSystem(
        atomic_numbers=np.array([1, 8, 1]),
        atomic_species=np.array([0, 2, 0]),
        positions=np.random.rand(3, 3),
        energy=2 * e0s[1] + e0s[8],
    )
    co2 = ChemicalSystem(
        atomic_numbers=np.array([8, 6, 8]),
        atomic_species=np.array([2, 1, 2]),
        positions=np.random.rand(3, 3),
        energy=2 * e0s[8] + e0s[6],
    )
    chooh = ChemicalSystem(
        atomic_numbers=np.array([1, 8, 6, 6, 1]),
        atomic_species=np.array([0, 2, 1, 1, 0]),
        positions=np.random.rand(5, 3),
        energy=2 * (e0s[1] + e0s[6]) + e0s[8],
    )
    co = ChemicalSystem(
        atomic_numbers=np.array([6, 8]),
        atomic_species=np.array([1, 2]),
        positions=np.random.rand(2, 3),
        energy=e0s[6] + e0s[8],
    )

    graphs = [
        create_graph_from_chemical_system(system, 1.0)
        for system in [h2o, co2, chooh, co]
    ]
    atomic_energies_computed = compute_average_e0s_from_graphs(graphs)

    idx_to_z = {0: 1, 1: 6, 2: 8}
    atomic_energies_computed = {
        idx_to_z[idx]: energy for idx, energy in atomic_energies_computed.items()
    }
    assert e0s == pytest.approx(atomic_energies_computed)


def test_atomic_number_table_works_correctly():
    with pytest.raises(AssertionError):
        AtomicNumberTable([7, 8, 8, 16])
    with pytest.raises(AssertionError):
        AtomicNumberTable([8, 7, 16])

    z_table = AtomicNumberTable([1, 6, 8, 16])

    assert len(z_table) == 4

    assert z_table.z_to_index(1) == 0
    assert z_table.z_to_index(6) == 1
    assert z_table.z_to_index(8) == 2
    assert z_table.z_to_index(16) == 3

    assert z_table.index_to_z(0) == 1
    assert z_table.index_to_z(1) == 6
    assert z_table.index_to_z(2) == 8
    assert z_table.index_to_z(3) == 16

    index_map_vec = z_table.z_to_index_map(max_atomic_number=21)
    expected_vec = [0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0]
    assert list(index_map_vec) == expected_vec


@pytest.mark.parametrize(
    "train,valid,test",
    [(0.1, 0.3, 0.6), (0.3, 0.3, 0.4), (0.0, 0.8, 0.2), (0.7, 0.1, 0.3)],
)
def test_data_is_correctly_split_randomly(train, valid, test):
    data = [4, 11, 19, 120, 38, 1, 18, 111, 0, -5]
    proportions = DataSplitProportions(train=train, validation=valid, test=test)

    if train == 0.0 or sum([train, valid, test]) > 1.0:
        with pytest.raises(SplitProportionsInvalidError):
            split_data_randomly(data, proportions, seed=42)
    else:
        train_set, valid_set, test_set = split_data_randomly(data, proportions, seed=42)
        assert len(train_set) == int(10 * train)
        assert len(valid_set) == int(10 * valid)
        assert len(test_set) == int(10 * test)


def test_data_is_correctly_split_randomly_by_group():
    data = (
        [i * 5 + 1 for i in range(7)]
        + [i * 5 + 2 for i in range(7)]
        + [i * 5 + 3 for i in range(7)]
        + [i * 5 + 4 for i in range(7)]
        + [5, 10, 15]
    )
    proportions = DataSplitProportions(train=0.5, validation=0.25, test=0.25)

    def _group_id_fun(data_point: int) -> str:
        mod_5 = data_point % 5
        if mod_5 == 0:
            return "_"
        return str(mod_5)

    train_set, valid_set, test_set = split_data_randomly_by_group(
        data,
        proportions,
        seed=42,
        get_group_id_fun=_group_id_fun,
        placeholder_group_id="_",
    )
    expected_test_set_with_given_settings = [1, 6, 11, 16, 21, 26, 31]

    assert len(train_set) == 17  # two groups plus placeholder group
    assert len(valid_set) == 7  # one group
    assert len(test_set) == 7  # one group
    assert 5 in train_set  # placeholder group must be in train_set
    assert 10 in train_set  # placeholder group must be in train_set
    assert 15 in train_set  # placeholder group must be in train_set

    # check that same seed gives the same results
    assert test_set == expected_test_set_with_given_settings


def test_data_is_correctly_split_by_group():
    num_conformers = 7
    data = [
        (f"abc_{frag_idx}_md_{i}", i * 5 + frag_idx)
        for i in range(num_conformers)
        for frag_idx in range(10)
    ]

    group_ids_by_split = (
        {f"abc_{i}" for i in range(6)},  # train
        {f"abc_{i}" for i in range(6, 8)},  # val
        {f"abc_{i}" for i in range(8, 10)},  # test
    )

    def _group_id_fun(data_point: tuple[str, int]) -> str:
        return "_".join(data_point[0].split("_")[:2])

    train_set, valid_set, test_set = split_data_by_group(
        data,
        get_group_id_fun=_group_id_fun,
        group_ids_by_split=group_ids_by_split,
    )

    assert len(train_set) + len(valid_set) + len(test_set) == len(data)
    assert len(train_set) == 6 * num_conformers
    assert len(valid_set) == 2 * num_conformers
    assert len(test_set) == 2 * num_conformers
    assert ("abc_5_md_0", 5) in train_set
    assert ("abc_6_md_0", 6) in valid_set
    assert ("abc_8_md_0", 8) in test_set


def test_correct_loading_of_stress():
    """Test loading of subset of MP dataset"""
    reader_config = ChemicalSystemsReaderConfig(
        train_dataset_paths=[str(SMALL_MP_DATASET_PATH.resolve())],
        valid_dataset_paths=None,
        test_dataset_paths=None,
        train_num_to_load=None,
        valid_num_to_load=None,
        test_num_to_load=None,
    )
    reader = ExtxyzReader(config=reader_config)
    train_systems, valid_systems, test_systems = reader.load()

    assert len(valid_systems) == 0
    assert len(test_systems) == 0

    assert len(train_systems) == 8

    for system in train_systems:
        assert isinstance(system, ChemicalSystem)
        expected_idx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
        assert list(system.atomic_species) == expected_idx
        expected_num = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 51]
        assert list(system.atomic_numbers) == expected_num
        assert system.pbc == (True, True, True)

    assert train_systems[0].energy == pytest.approx(-30.69383528)
    assert train_systems[1].positions[0][1] == pytest.approx(4.59071684)
    assert train_systems[2].forces[1][0] == pytest.approx(-0.01159171)
    assert np.allclose(
        train_systems[3].stress,
        np.array(
            [
                [-0.00221682, -0.0, 0.0],
                [-0.0, -0.00221682, -0.0],
                [0.0, -0.0, 0.00115818],
            ]
        ),
    )


def test_materials_shifts_and_edges():
    """Check that MP_trj structures are loaded with consistent edge vectors.

    If edge lengths overflow cutoff, then there is an inconsistent processing
    of "shifts" in Z^n. For instance, replicas of the sender node may not be
    placed within closest cell of the receiver node (e.g. sign error on shifts).

    See also unit tests on `get_edge_relative_vectors` in tests/utils.
    """
    reader_config = ChemicalSystemsReaderConfig(
        train_dataset_paths=[str(SMALL_MP_DATASET_PATH.resolve())],
        valid_dataset_paths=None,
        test_dataset_paths=None,
        train_num_to_load=None,
        valid_num_to_load=None,
        test_num_to_load=None,
    )
    reader = ExtxyzReader(config=reader_config)
    train_systems, valid_systems, test_systems = reader.load()

    dset_config = GraphDatasetBuilder.Config(
        graph_cutoff_angstrom=CUTOFF_ANGSTROM,
        batch_size=4,
    )

    dset_builder = GraphDatasetBuilder(
        reader=reader,
        dataset_config=dset_config,
    )

    dset_builder.prepare_datasets()
    dset, *_ = dset_builder.get_splits()

    graph = next(iter(dset))

    vectors = get_edge_relative_vectors(
        graph.nodes.positions,
        graph.senders,
        graph.receivers,
        graph.edges.shifts,
        graph.globals.cell,
        graph.n_edge,
    )

    assert vectors.shape[-1] == 3
    distances = np.sqrt(np.sum(vectors * vectors, axis=-1))
    assert max(distances) < CUTOFF_ANGSTROM
