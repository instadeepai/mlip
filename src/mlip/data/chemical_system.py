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

from typing import Optional, TypeAlias

import numpy as np
import pydantic
from typing_extensions import Self

Positions: TypeAlias = np.ndarray  # [num_nodes, 3]
Forces: TypeAlias = np.ndarray  # [num_nodes, 3]
AtomicNumbers: TypeAlias = np.ndarray  # [num_nodes]
AtomicSpecies: TypeAlias = np.ndarray  # [num_nodes, num_features]
Cell: TypeAlias = np.ndarray  # [3, 3]
Stress: TypeAlias = np.ndarray  # [3, 3]


class ChemicalSystem(pydantic.BaseModel):
    """Pydantic dataclass for a chemical system.

    The chemical system objects are returned by the chemical systems' reader.
    This class also performs the validations listed below.

    Attributes:
        atomic_numbers: The atomic numbers of the system. This should be a 1-dimensional
                        array of length "number of atoms".
        atomic_species: The atomic species of the system, which are the features of
                        each element (this can be a single value or an array itself).
                        This array can be either one or two-dimensional, depending
                        on the number of features per atom.
        positions: The array of positions for the system in Angstrom.
        energy: Optionally, a reference energy in eV.
        forces: Optionally, reference forces in eV/Angstrom.
        stress: Optionally, the stress in eV/Angstrom^3.
        cell: Optionally, a unit cell, which is an array of shape ``(3, 3)``.
        pbc: Optionally, periodic boundary conditions, which is a tuple of three
             booleans, one for each dimension whether the unit cell is periodic in
             that dimension.
        weight: A weighting factor for this configuration in the dataset, by default
                set to 1.0.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    atomic_numbers: AtomicNumbers
    atomic_species: AtomicSpecies
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    stress: Optional[Stress] = None  # eV/Angstrom^3
    cell: Optional[Cell] = None
    pbc: Optional[tuple[bool, bool, bool]] = None
    weight: float = 1.0  # weight of config in loss

    @pydantic.model_validator(mode="after")
    def validate_variable_shapes(self) -> Self:
        """Validates that atomic species, positions, and forces have the correct
        shape.
        """
        num_nodes = self.atomic_numbers.shape[0]

        if self.atomic_species.shape[0] != num_nodes:
            raise ValueError("Atomic species have incompatible shape.")

        if self.positions.shape != (num_nodes, 3):
            raise ValueError("Positions have incompatible shape.")

        if self.forces is not None and self.forces.shape != (num_nodes, 3):
            raise ValueError("Forces have incompatible shape.")

        return self

    @pydantic.field_validator("cell")
    @classmethod
    def validate_cell_shape(cls, value: Optional[Cell]) -> Optional[Cell]:
        """Validates that the cell has the correct shape."""
        if value is not None and value.shape != (3, 3):
            raise ValueError("Cell must be of shape 3x3.")
        return value

    @pydantic.field_validator("stress")
    @classmethod
    def validate_stress_shape(cls, value: Optional[Stress]) -> Optional[Stress]:
        """Validates that the stress has the correct shape."""
        if value is not None and value.shape != (3, 3):
            raise ValueError("Stress must be of shape 3x3.")
        return value
