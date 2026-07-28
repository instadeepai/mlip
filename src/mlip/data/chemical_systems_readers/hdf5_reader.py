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

import bisect
import os
from typing import SupportsIndex

import h5py
import numpy as np

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)
from mlip.data.chemical_systems_readers.chemical_systems_dataset import (
    ChemicalSystemsDataset,
)
from mlip.data.chemical_systems_readers.defaults import (
    DEFAULT_CELL,
    DEFAULT_PBC,
    DEFAULT_WEIGHT,
)
from mlip.data.chemical_systems_readers.type_aliases import (
    ChemicalSystems,
)


class Hdf5Reader(ChemicalSystemsReader, ChemicalSystemsDataset):
    """Implementation of a chemical systems reader that loads data from hdf5 format.

    On some platforms / h5py versions, keeping HDF5 file handles open for a long
    time while repeatedly reading can leak memory. Pass ``reopen_every_n_reads``
    to periodically close and reopen handles as a workaround (disabled by default).
    """

    def __init__(self, *args, reopen_every_n_reads: int | None = None, **kwargs):
        # Initialize before super()/validation so __del__ is safe on failed construct.
        self._files: list[h5py.File] | None = None
        self._db_ids: list[list[str]] | None = None
        self._idlen_cumulative: list[int] | None = None
        self._length: int | None = None
        self._read_count = 0
        super().__init__(*args, **kwargs)

        if reopen_every_n_reads is not None and reopen_every_n_reads <= 0:
            raise ValueError(
                "reopen_every_n_reads must be a positive integer or None, "
                f"got {reopen_every_n_reads}."
            )
        self.reopen_every_n_reads = reopen_every_n_reads

    def _open_files(self) -> None:
        self._files = [h5py.File(path, "r") for path in self._paths]

    def _close_files(self) -> None:
        if self._files is None:
            return
        for h5file in self._files:
            h5file.close()
        self._files = None

    def _ensure_index(self) -> None:
        if self._files is None:
            self._open_files()

        if self._db_ids is not None:
            return

        db_ids: list[list[str]] = []
        for h5file in self._files:
            ids = self._visit_groups(h5file, return_names=True)
            if self.num_to_load is not None:
                ids = ids[: self.num_to_load]
            db_ids.append(ids)

        id_lens = [len(ids) for ids in db_ids]
        self._db_ids = db_ids
        self._idlen_cumulative = np.cumsum(id_lens).tolist() if id_lens else []
        self._length = int(sum(id_lens))

    def load(self) -> ChemicalSystems:
        """Load chemical systems from all HDF5 filepaths."""
        filepaths = self.filepaths
        if not isinstance(filepaths, list):
            filepaths = [filepaths]

        all_systems: ChemicalSystems = []
        for filepath in filepaths:
            all_systems.extend(self._load_single_file(filepath))
        return all_systems

    def __len__(self) -> int:
        self._ensure_index()
        return self._length

    def __getitem__(self, index: SupportsIndex) -> ChemicalSystem:
        self._ensure_index()

        idx = int(index)
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)

        file_idx = bisect.bisect(self._idlen_cumulative, idx)
        el_idx = idx if file_idx == 0 else idx - self._idlen_cumulative[file_idx - 1]

        group = self._files[file_idx][self._db_ids[file_idx][el_idx]]
        system = self._hdf5_row_to_chemical_system(group)

        if self.reopen_every_n_reads is not None:
            self._read_count += 1
            if self._read_count >= self.reopen_every_n_reads:
                self._close_files()
                self._open_files()
                self._read_count = 0

        return system

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = None
        state["_read_count"] = 0
        return state

    def __del__(self):
        self._close_files()

    @staticmethod
    def _visit_groups(file, return_names=False):
        groups = []

        def append_group(name, item):
            if isinstance(item, h5py.Group) and item.parent.name == "/":
                groups.append(name if return_names else item)

        file.visititems(append_group)
        return groups

    def _read_file(self, filepath: str | os.PathLike) -> ChemicalSystems:
        """Read structures from an HDF5 file and convert each to a
        :class:`~mlip.data.chemical_system.ChemicalSystem`."""
        with h5py.File(filepath, "r") as h5file:
            groups = self._visit_groups(h5file)
            if self.num_to_load:
                groups = groups[: self.num_to_load]
            return [self._hdf5_row_to_chemical_system(group) for group in groups]

    def _hdf5_row_to_chemical_system(
        self,
        structure: h5py.Group,
    ) -> ChemicalSystem:
        """Convert a single HDF5 group to a :class:`ChemicalSystem`.

        Args:
            structure: An HDF5 group representing one structure.
        """
        atomic_numbers = structure["elements"][:]
        positions = structure["positions"][:]
        energy = structure.attrs[self.property_name_mapping["energy"]]
        forces = structure[self.property_name_mapping["forces"]][:]
        stress = None
        hessian = None
        partial_charges = None
        charge = None
        spin_multiplicity = None
        dipole_moment = None
        if self.property_name_mapping["stress"] in structure:
            # currently there's no stress in hdf5 from mlip-datagen, but might be in
            # other hdf5s.
            stress = structure[self.property_name_mapping["stress"]][:]
        if self.property_name_mapping["partial_charges"] in structure:
            partial_charges = structure[self.property_name_mapping["partial_charges"]][
                :
            ]
        if self.property_name_mapping["charge"] in structure.attrs:
            charge = structure.attrs[self.property_name_mapping["charge"]]
        if self.property_name_mapping["spin_multiplicity"] in structure.attrs:
            spin_multiplicity = structure.attrs[
                self.property_name_mapping["spin_multiplicity"]
            ]
        if self.property_name_mapping["dipole_moment"] in structure.attrs:
            dipole_moment = structure.attrs[self.property_name_mapping["dipole_moment"]]

        if self.property_name_mapping["hessian"] in structure:
            hessian = structure[self.property_name_mapping["hessian"]][:]

        return ChemicalSystem(
            atomic_numbers=atomic_numbers,
            positions=positions,
            energy=energy,
            forces=forces,
            hessian=hessian,
            stress=stress,
            cell=DEFAULT_CELL,
            pbc=DEFAULT_PBC,
            weight=DEFAULT_WEIGHT,
            partial_charges=partial_charges,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            dipole_moment=dipole_moment,
        )
