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

import h5py
import numpy as np
import pytest

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.hdf5_reader import Hdf5Reader

DATA_DIR = Path(__file__).parent.parent / "sample_data"
SPICE_SMALL_HDF5_PATH = DATA_DIR / "spice2-1000_429_md_0-1.hdf5"


@pytest.mark.parametrize("num_to_load", [None, 1])
def test_hdf5_reading(num_to_load):
    reader = Hdf5Reader(
        filepaths=SPICE_SMALL_HDF5_PATH.resolve(),
        num_to_load=num_to_load,
    )
    systems = reader.load()

    if num_to_load is None:
        expected_num = 2
    else:
        expected_num = num_to_load

    assert len(systems) == expected_num

    for system in systems:
        assert isinstance(system, ChemicalSystem)
        expected_atomic_numbers = [8, 8, 8, 16, 8, 8, 6, 6, 6, 15, 1, 1, 1]
        assert list(system.atomic_numbers) == expected_atomic_numbers
        assert system.pbc == (False, False, False)
        assert np.all(system.cell == 0.0)

    assert systems[0].energy == pytest.approx(-33533.58818835873)
    assert systems[0].positions[0][1] == pytest.approx(-0.03931140731653776)
    assert systems[0].forces[1][0] == pytest.approx(0.9863849542696035)


@pytest.mark.parametrize("num_to_load", [None, 1])
def test_hdf5_reader_with_multiple_filepaths(num_to_load):
    reader = Hdf5Reader(
        filepaths=[
            SPICE_SMALL_HDF5_PATH.resolve(),
            SPICE_SMALL_HDF5_PATH.resolve(),
        ],
        num_to_load=num_to_load,
    )
    systems = reader.load()

    if num_to_load is None:
        expected_num = 4
    else:
        expected_num = num_to_load * 2

    assert len(systems) == expected_num

    for system in systems:
        assert isinstance(system, ChemicalSystem)


@pytest.mark.parametrize("num_chunks", [1, 2, 5])
def test_hdf5_reader_prepare_parallel_readers_matches_sequential(num_chunks, tmp_path):
    filepath = SPICE_SMALL_HDF5_PATH.resolve()
    reader = Hdf5Reader(filepaths=filepath)
    sequential_systems = reader.load()

    sub_readers = reader.prepare_parallel_readers(tmp_path, num_chunks)

    if num_chunks <= 1:
        assert len(sub_readers) == 1
    else:
        assert len(sub_readers) > 1

    chunked_systems = []
    for sub_reader in sub_readers:
        chunked_systems.extend(sub_reader.load())

    assert sorted(s.energy for s in chunked_systems) == sorted(
        s.energy for s in sequential_systems
    )


def test_hdf5_reader_only_visits_top_level_groups(tmp_path):
    filepath = tmp_path / "nested.hdf5"
    with h5py.File(filepath, "w") as h5file:
        for i in range(2):
            group = h5file.create_group(f"structure_{i}")
            group.create_dataset("elements", data=np.array([1, 8]))
            group.create_dataset(
                "positions", data=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])
            )
            group.create_dataset("forces", data=np.zeros((2, 3)))
            group.attrs["energy"] = float(-i)
        # A nested subgroup: must be skipped.
        h5file["structure_0"].create_group("nested_metadata")
        # A top-level dataset (not a group): must be skipped.
        h5file.create_dataset("file_level_dataset", data=np.arange(5))

    reader = Hdf5Reader(filepaths=filepath)

    systems = reader.load()
    assert len(systems) == 2
    assert sorted(s.energy for s in systems) == [-1.0, 0.0]

    # prepare_parallel_readers relies on the same enumeration.
    assert sorted(reader._list_group_names(filepath)) == [
        "/structure_0",
        "/structure_1",
    ]


def test_hdf5_reader_with_charge_and_spin_multiplicity(tmp_path):
    filepath = tmp_path / "charge_spin.hdf5"
    with h5py.File(filepath, "w") as h5file:
        group = h5file.create_group("structure_0")
        group.create_dataset("elements", data=np.array([1, 8]))
        group.create_dataset(
            "positions",
            data=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]]),
        )
        group.create_dataset("forces", data=np.zeros((2, 3)))
        group.attrs["energy"] = -1.5
        group.attrs["charge"] = -1
        group.attrs["spin_multiplicity"] = 2

    reader = Hdf5Reader(filepaths=filepath)

    systems = reader.load()

    assert len(systems) == 1
    system = systems[0]
    assert isinstance(system, ChemicalSystem)
    assert system.charge == -1
    assert system.spin_multiplicity == 2
