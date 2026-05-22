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

    def load(self) -> ChemicalSystems:
        """Load chemical systems from all HDF5 filepaths."""
        filepaths = self.filepaths
        if not isinstance(filepaths, list):
            filepaths = [filepaths]

        all_systems: ChemicalSystems = []
        for filepath in filepaths:
            all_systems.extend(self._load_single_file(filepath))
        return all_systems

    def _read_file(self, filepath: str | os.PathLike) -> ChemicalSystems:
        """Read structures from an HDF5 file and convert each to a
        :class:`~mlip.data.chemical_system.ChemicalSystem`."""
        with h5py.File(filepath, "r") as h5file:
            struct_names = list(h5file.keys())
            if self.num_to_load:
                struct_names = struct_names[: self.num_to_load]
            return [
                self._hdf5_row_to_chemical_system(h5file[struct_name])
                for struct_name in struct_names
            ]

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
