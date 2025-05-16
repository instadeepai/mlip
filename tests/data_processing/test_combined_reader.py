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
from mlip.data.chemical_systems_readers.combined_reader import CombinedReader
from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.data.chemical_systems_readers.hdf5_reader import Hdf5Reader
from mlip.data.configs import ChemicalSystemsReaderConfig

DATA_DIR = Path(__file__).parent.parent / "data"
SPICE_SMALL_HDF5_PATH = DATA_DIR / "spice2-1000_429_md_0-1.hdf5"
SMALL_ASPIRIN_DATASET_PATH = DATA_DIR / "small_aspirin_test.xyz"


@pytest.mark.parametrize("train_num_to_load", [None, 1])
def test_combined_data_reader_supports_hdf5_and_extxyz(train_num_to_load):
    extxyz_reader_config = ChemicalSystemsReaderConfig(
        reader_type="extxyz",
        train_dataset_paths=[SMALL_ASPIRIN_DATASET_PATH.resolve()],
        valid_dataset_paths=None,
        test_dataset_paths=None,
        train_num_to_load=train_num_to_load,
        valid_num_to_load=None,
        test_num_to_load=None,
    )
    hdf5_reader_config = ChemicalSystemsReaderConfig(
        reader_type="hdf5",
        train_dataset_paths=[SPICE_SMALL_HDF5_PATH.resolve()],
        valid_dataset_paths=None,
        test_dataset_paths=None,
        train_num_to_load=train_num_to_load,
        valid_num_to_load=None,
        test_num_to_load=None,
    )

    reader = CombinedReader(
        [Hdf5Reader(hdf5_reader_config), ExtxyzReader(extxyz_reader_config)]
    )
    train_systems, valid_systems, test_systems = reader.load()

    assert len(valid_systems) == 0
    assert len(test_systems) == 0

    if train_num_to_load is None:
        assert len(train_systems) == 9
    else:
        # because from each hdf5 file will load train_num_to_load rows
        assert len(train_systems) == train_num_to_load * 2

    expected_idx_hdf5 = [2, 2, 2, 4, 2, 2, 1, 1, 1, 3, 0, 0, 0]
    expected_idx_xyz = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_num_hdf5 = [8, 8, 8, 16, 8, 8, 6, 6, 6, 15, 1, 1, 1]
    expected_num_xyz = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
    for system in train_systems:
        assert isinstance(system, ChemicalSystem)
        assert (
            list(system.atomic_species) == expected_idx_hdf5
            or list(system.atomic_species) == expected_idx_xyz
        )
        assert (
            list(system.atomic_numbers) == expected_num_hdf5
            or list(system.atomic_numbers) == expected_num_xyz
        )
        assert system.pbc == (False, False, False)
        assert np.all(system.cell == 0.0)

    assert train_systems[0].energy == pytest.approx(-33533.58818835873)
    assert train_systems[0].positions[0][1] == pytest.approx(-0.03931140731653776)
    assert train_systems[0].forces[1][0] == pytest.approx(0.9863849542696035)
