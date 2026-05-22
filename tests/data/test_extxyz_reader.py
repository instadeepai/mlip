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
from mlip.data.chemical_systems_readers.defaults import DEFAULT_PROPERTY_KEY_MAPPING
from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.graph import Graph

DATA_DIR = Path(__file__).parent.parent / "sample_data"
SMALL_ASPIRIN_DATASET_PATH = DATA_DIR / "small_aspirin_test.xyz"
SMALL_ASPIRIN_WITH_CHARGE_SPIN_PATH = DATA_DIR / "small_aspirin_with_charge_spin.xyz"
DIMETHYL_SULFOXIDE_DATASET_PATH = DATA_DIR / "Dimethyl_sulfoxide.xyz"
SMALL_SPICE_WITH_CHARGES_PATH = DATA_DIR / "SPICE_small_with_charges.xyz"


@pytest.mark.parametrize("num_to_load", [None, 3])
def test_extxyz_reading(num_to_load):
    reader = ExtxyzReader(
        filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(),
        num_to_load=num_to_load,
    )
    systems = reader.load()

    if num_to_load is None:
        expected_num = 7
    else:
        expected_num = num_to_load

    assert len(systems) == expected_num

    for system in systems:
        assert isinstance(system, ChemicalSystem)
        # fmt: off
        expected_atomic_numbers = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1] # noqa
        # fmt: on
        assert list(system.atomic_numbers) == expected_atomic_numbers
        assert system.pbc == (False, False, False)
        assert np.all(system.cell == 0.0)
        assert system.partial_charges is None
        assert system.charge is None
        assert system.spin_multiplicity is None
        assert system.dipole_moment is None

    assert systems[0].energy == pytest.approx(-17617.826906338443)
    assert systems[1].positions[0][1] == pytest.approx(-0.97843255)
    assert systems[2].forces[1][0] == pytest.approx(-0.05825649)


def test_extxyz_reader_with_charges():
    property_name_mapping = dict(DEFAULT_PROPERTY_KEY_MAPPING)
    property_name_mapping["partial_charges"] = "lowdin_charges"
    reader = ExtxyzReader(
        filepaths=SMALL_SPICE_WITH_CHARGES_PATH.resolve(),
        property_name_mapping=property_name_mapping,
    )
    systems = reader.load()

    assert len(systems) == 5
    for system in systems:
        assert isinstance(system, ChemicalSystem)
        n_nodes = len(system.atomic_numbers)
        assert system.partial_charges is not None
        assert system.partial_charges.shape == (n_nodes,)
        assert system.charge is not None
        assert isinstance(system.charge, int)
        assert system.charge == 0
        assert system.spin_multiplicity is None
        assert system.dipole_moment is None


def test_extxyz_reader_with_charge_and_spin_multiplicity():
    reader = ExtxyzReader(
        filepaths=SMALL_ASPIRIN_WITH_CHARGE_SPIN_PATH.resolve(),
        property_name_mapping={"spin_multiplicity": "spin"},
    )

    systems = reader.load()

    assert len(systems) == 2
    assert [system.charge for system in systems] == [0, 1]
    assert [system.spin_multiplicity for system in systems] == [1, 2]

    graph = Graph.from_chemical_system(systems[1], graph_cutoff_angstrom=2.0)

    assert graph.globals.charge is not None
    assert graph.globals.spin_multiplicity is not None
    assert np.asarray(graph.globals.charge).tolist() == pytest.approx([1.0])
    assert np.asarray(graph.globals.spin_multiplicity).tolist() == pytest.approx([2.0])


@pytest.mark.parametrize("num_to_load", [None, 1])
def test_extxyz_reader_with_multiple_filepaths(num_to_load):
    reader = ExtxyzReader(
        filepaths=[
            SMALL_ASPIRIN_DATASET_PATH.resolve(),
            DIMETHYL_SULFOXIDE_DATASET_PATH.resolve(),
        ],
        num_to_load=num_to_load,
    )
    systems = reader.load()

    if num_to_load is None:
        expected_num = 8
    else:
        expected_num = num_to_load * 2

    assert len(systems) == expected_num

    for system in systems:
        assert isinstance(system, ChemicalSystem)
