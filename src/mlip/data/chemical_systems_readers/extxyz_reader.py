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
from typing import Callable

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read as ase_read

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)
from mlip.data.chemical_systems_readers.type_aliases import (
    ChemicalSystems,
)


class ExtxyzReader(ChemicalSystemsReader):
    """Implementation of a chemical systems reader that loads data from extxyz format
    via the `ase` library.
    """

    def load(self) -> ChemicalSystems:
        """Load chemical systems from all extxyz filepaths."""
        filepaths = self.filepaths
        if not isinstance(filepaths, list):
            filepaths = [filepaths]

        all_systems: ChemicalSystems = []
        for filepath in filepaths:
            all_systems.extend(self._load_single_file(filepath))
        return all_systems

    def _read_file(self, filepath: str | os.PathLike) -> ChemicalSystems:
        """Read a single extxyz file and convert to chemical systems."""
        index_to_load = ":" if self.num_to_load is None else f":{self.num_to_load}"
        result = ase_read(filepath, format="extxyz", index=index_to_load)
        if not isinstance(result, list):
            result = [result]
        return [
            ChemicalSystem.from_ase_atoms(
                atoms, property_name_mapping=self.property_name_mapping
            )
            for atoms in result
        ]

    @staticmethod
    def _get_atoms_property(
        property_fun: Callable[..., np.ndarray | float], **kwargs
    ) -> np.ndarray | float | None:
        """Try reading an `ase.Atoms` property and return None if missing."""
        try:
            return property_fun(**kwargs)
        except PropertyNotImplementedError:
            return None
        except RuntimeError:
            # During inference, no calculator and no energy label
            # assigned raise RuntimeError
            return None
