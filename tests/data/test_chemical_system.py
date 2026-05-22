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

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pydantic import ValidationError

from mlip.data.chemical_system import ChemicalSystem


class TestChemicalSystemValidation:
    def test_valid_full_construction(self):
        system = ChemicalSystem(
            atomic_numbers=np.array([1, 8]),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            energy=-1.0,
            forces=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
            stress=np.eye(3) * 0.01,
            cell=np.eye(3) * 10.0,
            pbc=(True, True, True),
        )
        assert len(system.atomic_numbers) == 2
        assert system.energy == -1.0

    def test_valid_minimal_construction(self):
        system = ChemicalSystem(
            atomic_numbers=np.array([1, 6]),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        assert system.energy is None
        assert system.forces is None
        assert system.stress is None
        assert system.cell is None

    def test_mismatched_positions_shape(self):
        with pytest.raises(ValidationError, match="Positions have incompatible shape"):
            ChemicalSystem(
                atomic_numbers=np.array([1, 8]),
                positions=np.array([[0.0, 0.0, 0.0]]),  # 1 row, need 2
            )

    def test_mismatched_forces_shape(self):
        with pytest.raises(ValidationError, match="Forces have incompatible shape"):
            ChemicalSystem(
                atomic_numbers=np.array([1, 8]),
                positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                forces=np.array([[0.1, 0.0, 0.0]]),  # 1 row, need 2
            )

    def test_wrong_cell_shape(self):
        with pytest.raises(ValidationError, match="Cell must be of shape 3x3"):
            ChemicalSystem(
                atomic_numbers=np.array([1]),
                positions=np.array([[0.0, 0.0, 0.0]]),
                cell=np.eye(4),
            )

    def test_wrong_stress_shape(self):
        with pytest.raises(ValidationError, match="Stress must be of shape 3x3"):
            ChemicalSystem(
                atomic_numbers=np.array([1]),
                positions=np.array([[0.0, 0.0, 0.0]]),
                stress=np.zeros((6,)),
            )

    def test_extra_kwargs_silently_ignored(self):
        system = ChemicalSystem(
            atomic_numbers=np.array([1]),
            positions=np.array([[0.0, 0.0, 0.0]]),
            atomic_species=np.array([0]),  # no longer a field, should be ignored
        )
        assert not hasattr(system, "atomic_species")


class TestFromAseAtoms:
    def test_without_calculator(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        system = ChemicalSystem.from_ase_atoms(atoms)
        np.testing.assert_array_equal(system.atomic_numbers, [1, 1])
        assert system.energy == 0.0  # default when no calculator
        assert system.forces is None
        assert system.stress is None

    def test_with_single_point_calculator(self):
        atoms = Atoms("NaCl", positions=[[0, 0, 0], [2.0, 0, 0]])
        energy = -5.0
        forces = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        stress = np.eye(3) * 0.01
        calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
        atoms.calc = calc
        system = ChemicalSystem.from_ase_atoms(atoms)
        assert system.energy == energy
        np.testing.assert_array_almost_equal(system.forces, forces)
        np.testing.assert_array_equal(system.atomic_numbers, [11, 17])

    def test_cell_and_pbc_extracted(self):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
        system = ChemicalSystem.from_ase_atoms(atoms)
        assert system.cell is not None
        assert system.cell.shape == (3, 3)
