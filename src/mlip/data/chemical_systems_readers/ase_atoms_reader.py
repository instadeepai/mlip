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

import ase

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.defaults import DEFAULT_PROPERTY_KEY_MAPPING
from mlip.data.chemical_systems_readers.type_aliases import ChemicalSystems


class ASEAtomsReader:
    """Transforms a list of `ase.Atoms` into a list of `ChemicalSystem` objects.

    This reader class is useful to reuse the `GraphDatasetBuilder` pipeline
    programmatically, e.g. to batch graphs from `ase.Atoms` objects returned
    by a simulation or relaxation stage.
    """

    def __init__(
        self,
        atoms_list: list[ase.Atoms],
        num_to_load: int | None = None,
        property_name_mapping: dict[str, str] | None = None,
    ):
        self._atoms_list = atoms_list
        self._num_to_load = num_to_load
        if property_name_mapping is None:
            self.property_name_mapping = DEFAULT_PROPERTY_KEY_MAPPING
        else:
            self.property_name_mapping = (
                DEFAULT_PROPERTY_KEY_MAPPING | property_name_mapping
            )

    def load(self) -> ChemicalSystems:
        """
        Converts a single list[ase.Atoms] to a list of `ChemicalSystems`.
        """
        chemical_systems = [
            ChemicalSystem.from_ase_atoms(atoms, self.property_name_mapping)
            for atoms in self._atoms_list[: self._num_to_load]
        ]

        return chemical_systems
