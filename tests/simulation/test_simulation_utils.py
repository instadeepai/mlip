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

from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from mlip.simulation.utils import resolve_atoms_charge_for_model


def _make_atoms(charge=None, partial_charges=None) -> Atoms:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    if charge is not None:
        atoms.info["charge"] = charge
    if partial_charges is not None:
        atoms.info["partial_charges"] = partial_charges
    return atoms


def _make_force_field(use_total_charge_embedding: bool):
    ff = MagicMock()
    ff.config.use_total_charge_embedding = use_total_charge_embedding
    return ff


@pytest.mark.parametrize(
    "use_charge,set_none_to_zero,expected",
    [
        (False, False, None),
        (False, True, None),
        (True, True, 0),
        (True, False, "ERROR"),
    ],
)
def test_resolve_charge_when_input_has_none(use_charge, set_none_to_zero, expected):
    atoms = _make_atoms(charge=None, partial_charges=None)
    ff = _make_force_field(use_total_charge_embedding=use_charge)

    def _resolve_charge():
        return resolve_atoms_charge_for_model(atoms, ff, set_none_to_zero)

    if expected == "ERROR":
        with pytest.raises(ValueError, match="uses total charge embedding"):
            _resolve_charge()
    else:
        result = _resolve_charge()
        assert result.info.get("charge") == expected


def test_resolve_charge_unchanged_when_already_set():
    atoms = _make_atoms(charge=2)
    ff = _make_force_field(use_total_charge_embedding=True)
    resolve_atoms_charge_for_model(atoms, ff, set_none_charge_to_zero=False)
    assert atoms.info["charge"] == 2


def test_resolve_charge_derives_from_partial_charges():
    atoms = _make_atoms(partial_charges=np.array([0.4, 0.7, -0.2]))
    ff = _make_force_field(use_total_charge_embedding=True)
    resolve_atoms_charge_for_model(atoms, ff, set_none_charge_to_zero=False)
    assert atoms.info["charge"] == 1


def test_resolve_charge_handles_list_of_atoms():
    atoms_list = [_make_atoms(), _make_atoms(charge=1)]
    ff = _make_force_field(use_total_charge_embedding=True)
    result = resolve_atoms_charge_for_model(
        atoms_list, ff, set_none_charge_to_zero=True
    )
    assert result[0].info["charge"] == 0
    assert result[1].info["charge"] == 1
