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

import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import ase
import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.data import atomic_numbers as ase_atomic_numbers_map
from ase.io import read as ase_read

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)
from mlip.data.chemical_systems_readers.type_aliases import (
    ChemicalSystems,
    ChemicalSystemsBySplit,
)
from mlip.data.chemical_systems_readers.utils import (
    apply_flatten,
    filter_systems_with_unseen_atoms_and_assign_atomic_species,
)

ENERGY_KEY = "energy"
STRESS_KEY = "stress"
FORCES_KEY = "forces"
STRESS_PREFACTOR = 1.0
REMAP_STRESS = None
DEFAULT_WEIGHT = 1.0


class ExtxyzReader(ChemicalSystemsReader):
    """Implementation of a chemical systems reader that loads data from extxyz format
    via the ``ase`` library."""

    def load(
        self,
        postprocess_fun: Optional[
            Callable[
                [ChemicalSystems, ChemicalSystems, ChemicalSystems],
                ChemicalSystemsBySplit,
            ]
        ] = filter_systems_with_unseen_atoms_and_assign_atomic_species,
    ) -> ChemicalSystemsBySplit:

        # TRAIN
        train_systems = self._load_chemical_systems(
            self.config.train_dataset_paths, self.config.train_num_to_load
        )

        # VALIDATION
        valid_systems = self._load_chemical_systems(
            self.config.valid_dataset_paths, self.config.valid_num_to_load
        )

        # TEST
        test_systems = self._load_chemical_systems(
            self.config.test_dataset_paths, self.config.test_num_to_load
        )

        if postprocess_fun is not None:
            train_systems, valid_systems, test_systems = postprocess_fun(
                train_systems, valid_systems, test_systems
            )

        return train_systems, valid_systems, test_systems

    def _load_chemical_systems_from_single_extxyz(
        self, filepath: None | str | os.PathLike, num_to_load: Optional[int] = None
    ) -> ChemicalSystems:
        """Load atoms from an extxyz file and convert to a list of ChemicalSystems."""
        atoms_list = self._load_atoms_list_from_single_extxyz(filepath, num_to_load)
        return [self._convert_atoms_to_chemical_system(atoms) for atoms in atoms_list]

    def _load_chemical_systems(
        self, filepaths: list[str | os.PathLike], num_to_load: Optional[int] = None
    ) -> ChemicalSystems:
        return apply_flatten(
            self._load_chemical_systems_from_single_extxyz, filepaths, num_to_load
        )

    def _load_atoms_list_from_single_extxyz(
        self, filepath: None | str | os.PathLike, num_to_load: Optional[int] = None
    ) -> list[ase.Atoms]:
        if filepath is None:
            return []

        index_to_load = ":" if num_to_load is None else f":{num_to_load}"

        if self.data_download_fun is None:
            atoms_list = ase_read(filepath, format="extxyz", index=index_to_load)
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_filepath = Path(tmpdir) / "dataset.extxyz"
                self.data_download_fun(filepath, tmp_filepath)
                atoms_list = ase_read(
                    tmp_filepath, format="extxyz", index=index_to_load
                )

        if isinstance(atoms_list, list):
            return atoms_list
        return [atoms_list]

    def _convert_atoms_to_chemical_system(self, atoms: ase.Atoms) -> ChemicalSystem:
        energy = self._get_extxyz_property(atoms.get_potential_energy)  # eV
        stress = self._get_extxyz_property(
            atoms.get_stress, voigt=False
        )  # eV / Ang^3, get stress as 3x3 matrix

        if energy is None:
            energy = 0.0

        if stress is not None:
            stress = STRESS_PREFACTOR * stress

            if REMAP_STRESS is not None:
                remap_stress = np.asarray(REMAP_STRESS)
                assert remap_stress.shape == (3, 3)
                assert remap_stress.dtype.kind == "i"
                stress = stress.flatten()[remap_stress]

            assert stress.shape == (3, 3)

        forces = self._get_extxyz_property(atoms.get_forces)  # eV / Ang
        atomic_numbers = np.array(
            [ase_atomic_numbers_map[symbol] for symbol in atoms.symbols]
        )

        pbc = tuple(atoms.get_pbc())
        cell = np.array(atoms.get_cell())
        assert np.linalg.det(cell) >= 0.0

        weight = DEFAULT_WEIGHT

        return ChemicalSystem(
            atomic_numbers=atomic_numbers,
            atomic_species=np.empty(atomic_numbers.shape[0]),  # will be populated later
            positions=atoms.get_positions(),
            energy=energy,
            forces=forces,
            stress=stress,
            cell=cell,
            pbc=pbc,
            weight=weight,
        )

    @staticmethod
    def _get_extxyz_property(
        property_fun: Callable[..., np.ndarray | float], **kwargs
    ) -> Optional[np.ndarray | float]:
        try:
            return property_fun(**kwargs)
        except PropertyNotImplementedError:
            return None
