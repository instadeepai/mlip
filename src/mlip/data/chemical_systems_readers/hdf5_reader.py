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

import h5py
import numpy as np

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

STRESS_KEY = "stress"
DEFAULT_WEIGHT = 1.0
DEFAULT_PBC = np.zeros(3, bool)
DEFAULT_CELL = np.zeros((3, 3))


class Hdf5Reader(ChemicalSystemsReader):
    """Implementation of a chemical systems reader that loads data from hdf5 format."""

    def load(
        self,
        postprocess_fun: Optional[
            Callable[
                [ChemicalSystems, ChemicalSystems, ChemicalSystems],
                ChemicalSystemsBySplit,
            ]
        ] = filter_systems_with_unseen_atoms_and_assign_atomic_species,
    ) -> ChemicalSystemsBySplit:

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

    def _load_chemical_systems_from_single_hdf5(
        self, filepath: None | str | os.PathLike, num_to_load: Optional[int] = None
    ) -> ChemicalSystems:
        """Load atoms from an hdf5 file and convert to a list of ChemicalSystems."""
        if filepath is None:
            return []

        if self.data_download_fun is None:
            return self._load_hdf5(filepath, num_to_load=num_to_load)
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_filepath = Path(tmpdir) / "dataset.hdf5"
                self.data_download_fun(filepath, tmp_filepath)
                return self._load_hdf5(tmp_filepath, num_to_load=num_to_load)

    def _load_chemical_systems(
        self, filepaths: list[str | os.PathLike], num_to_load: Optional[int] = None
    ) -> ChemicalSystems:
        return apply_flatten(
            self._load_chemical_systems_from_single_hdf5, filepaths, num_to_load
        )

    def _load_hdf5(
        self, filepath: str | os.PathLike, num_to_load: Optional[int] = None
    ) -> ChemicalSystems:
        with h5py.File(filepath, "r") as h5file:
            struct_names = list(h5file.keys())
            if num_to_load:
                struct_names = struct_names[:num_to_load]
            return [
                self._hdf5_row_to_chemical_system(h5file[struct_name])
                for struct_name in struct_names
            ]

    def _hdf5_row_to_chemical_system(self, structure: h5py.Group) -> ChemicalSystem:
        positions = structure["positions"][:]
        element_numbers = structure["elements"][:]
        forces = structure["forces"][:]
        energy = structure.attrs["energy"]
        stress = None
        if STRESS_KEY in structure:
            # currently there's no stress in hdf5 from mlip-datagen, but might be in
            # other hdf5s.
            stress = structure[STRESS_KEY][:]

        return ChemicalSystem(
            atomic_numbers=element_numbers,
            atomic_species=np.empty(element_numbers.shape[0]),
            # will be populated later
            positions=positions,
            energy=energy,
            forces=forces,
            stress=stress,
            cell=DEFAULT_CELL,
            pbc=DEFAULT_PBC,
            weight=DEFAULT_WEIGHT,
        )
