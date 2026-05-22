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

import numpy as np
import pytest

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.data.configs import GraphDatasetBuilderConfig
from mlip.data.graph_dataset_builder import SingleGraphDatasetBuilder
from mlip.data.helpers.atomic_energies import compute_average_e0s_from_graphs
from mlip.data.helpers.data_split import (
    DataSplitProportions,
    split_data_by_group,
    split_data_randomly,
    split_data_randomly_by_group,
)
from mlip.data.helpers.exceptions import SplitProportionsInvalidError
from mlip.graph import Graph

DATA_DIR = Path(__file__).parent.parent / "sample_data"
SMALL_MP_DATASET_PATH = DATA_DIR / "small_materials_test.extxyz"

CUTOFF_ANGSTROM = 6


def test_atomic_energies_calculation_works():
    e0s = {1: -0.123, 6: -4.34, 8: -6.54}
    h2o = ChemicalSystem(
        atomic_numbers=np.array([1, 8, 1]),
        positions=np.random.rand(3, 3),
        energy=2 * e0s[1] + e0s[8],
    )
    co2 = ChemicalSystem(
        atomic_numbers=np.array([8, 6, 8]),
        positions=np.random.rand(3, 3),
        energy=2 * e0s[8] + e0s[6],
    )
    chooh = ChemicalSystem(
        atomic_numbers=np.array([1, 8, 6, 6, 1]),
        positions=np.random.rand(5, 3),
        energy=2 * (e0s[1] + e0s[6]) + e0s[8],
    )
    co = ChemicalSystem(
        atomic_numbers=np.array([6, 8]),
        positions=np.random.rand(2, 3),
        energy=e0s[6] + e0s[8],
    )

    graphs = [
        Graph.from_chemical_system(system, 1.0) for system in [h2o, co2, chooh, co]
    ]
    atomic_energies_computed = compute_average_e0s_from_graphs(graphs)

    atomic_energies_computed = {
        z: energy for z, energy in atomic_energies_computed.items()
    }
    assert e0s == pytest.approx(atomic_energies_computed)


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
    reader = ExtxyzReader(
        filepaths=SMALL_MP_DATASET_PATH.resolve(),
    )
    systems = reader.load()

    assert len(systems) == 8

    for system in systems:
        assert isinstance(system, ChemicalSystem)
        expected_num = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 51]
        assert list(system.atomic_numbers) == expected_num
        assert system.pbc == (True, True, True)

    assert systems[0].energy == pytest.approx(-30.69383528)
    assert systems[1].positions[0][1] == pytest.approx(4.59071684)
    assert systems[2].forces[1][0] == pytest.approx(-0.01159171)
    assert np.allclose(
        systems[3].stress,
        np.array([
            [-0.00221682, -0.0, 0.0],
            [-0.0, -0.00221682, -0.0],
            [0.0, -0.0, 0.00115818],
        ]),
    )


def test_materials_shifts_and_edges():
    """Check that MP_trj structures are loaded with consistent edge vectors.

    If edge lengths overflow cutoff, then there is an inconsistent processing
    of "shifts" in Z^n. For instance, replicas of the sender node may not be
    placed within closest cell of the receiver node (e.g. sign error on shifts).

    See also unit tests on `Graph.edge_vectors()` in tests/utils.
    """
    builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=CUTOFF_ANGSTROM,
        batch_size=4,
    )
    reader = ExtxyzReader(
        filepaths=SMALL_MP_DATASET_PATH.resolve(),
    )
    graph_dataset_builder = SingleGraphDatasetBuilder(
        reader, builder_config, dataset_info=False
    )
    dset = graph_dataset_builder.get_dataset(prefetch=False)

    graph = next(iter(dset))

    vectors = graph.edge_vectors()

    assert vectors.shape[-1] == 3
    distances = np.sqrt(np.sum(vectors * vectors, axis=-1))
    assert max(distances) < CUTOFF_ANGSTROM
