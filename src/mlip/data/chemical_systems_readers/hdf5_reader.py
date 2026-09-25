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

import math
import os

import h5py

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)
from mlip.data.chemical_systems_readers.defaults import (
    DEFAULT_CELL,
    DEFAULT_PBC,
    DEFAULT_WEIGHT,
)
from mlip.data.chemical_systems_readers.type_aliases import (
    ChemicalSystems,
)


class Hdf5Reader(ChemicalSystemsReader):
    """Implementation of a chemical systems reader that loads data from hdf5 format."""

    # If set, `_read_file` only reads these group names to enable parallel loading.
    _group_names: list[str] | None = None

    def load(self) -> ChemicalSystems:
        """Load chemical systems from all HDF5 filepaths."""
        filepaths = self.filepaths
        if not isinstance(filepaths, list):
            filepaths = [filepaths]

        all_systems: ChemicalSystems = []
        for filepath in filepaths:
            all_systems.extend(self._load_single_file(filepath))
        return all_systems

    @staticmethod
    def _visit_groups(file):
        groups = []
        for name in file:
            item = file[name]
            if isinstance(item, h5py.Group):
                groups.append(item)
        return groups

    def _read_file(self, filepath: str | os.PathLike) -> ChemicalSystems:
        """Read structures from an HDF5 file and convert each to a
        :class:`~mlip.data.chemical_system.ChemicalSystem`."""
        with h5py.File(filepath, "r") as h5file:
            if self._group_names is not None:
                groups = [h5file[name] for name in self._group_names]
            else:
                groups = self._visit_groups(h5file)
                if self.num_to_load:
                    groups = groups[: self.num_to_load]
            return [self._hdf5_row_to_chemical_system(group) for group in groups]

    def prepare_parallel_readers(
        self, download_dir: str | os.PathLike, num_chunks: int
    ) -> list["Hdf5Reader"]:
        """Convert this reader to a list of readers suitable for parallel loading.

        Resolves local paths for all files, then splits into per-chunk readers, each
        responsible for a disjoint slice of the HDF5 groups in the local file(s),
        so multiple workers can parse a single large file concurrently.

        Falls back to a single reader pointed at the local files when `num_chunks <= 1`
        or there is at most one group to load.
        """
        if num_chunks <= 1:
            return super().prepare_parallel_readers(download_dir, num_chunks)

        local_filepaths = self._resolve_local_filepaths(download_dir)

        shards: list[tuple[os.PathLike, list[str]]] = []
        for filepath in local_filepaths:
            names = self._list_group_names(filepath)
            if self.num_to_load is not None:
                names = names[: self.num_to_load]
            shards.append((filepath, names))

        total = sum(len(names) for _, names in shards)
        if total <= 1:
            return [self._with_local_filepaths(local_filepaths)]

        chunk_size = max(1, math.ceil(total / num_chunks))
        sub_readers = []
        for filepath, names in shards:
            for i in range(0, len(names), chunk_size):
                sub_reader = type(self)(
                    filepaths=filepath,
                    property_name_mapping=self.property_name_mapping,
                )
                sub_reader._group_names = names[i : i + chunk_size]
                sub_readers.append(sub_reader)
        return sub_readers

    def _list_group_names(self, filepath: str | os.PathLike) -> list[str]:
        """Fast metadata-only pass listing top-level group names."""
        with h5py.File(filepath, "r") as h5file:
            return [group.name for group in self._visit_groups(h5file)]

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
