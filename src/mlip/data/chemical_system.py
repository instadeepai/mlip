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


from typing import Any, TypeAlias

import ase
import numpy as np
import pydantic
from ase.calculators.calculator import PropertyNotImplementedError
from ase.data import atomic_numbers as ase_atomic_numbers_map
from typing_extensions import Self

from mlip.data.chemical_systems_readers.defaults import DEFAULT_PROPERTY_KEY_MAPPING

Positions: TypeAlias = np.ndarray  # [num_nodes, 3]
Forces: TypeAlias = np.ndarray  # [num_nodes, 3]
AtomicNumbers: TypeAlias = np.ndarray  # [num_nodes]
Cell: TypeAlias = np.ndarray  # [3, 3]
Stress: TypeAlias = np.ndarray  # [3, 3]
Hessian: TypeAlias = np.ndarray  # [num_nodes, 3, num_nodes, 3]
PartialCharges: TypeAlias = np.ndarray  # [num_nodes]
Charge: TypeAlias = int
SpinMultiplicity: TypeAlias = int
DipoleMoment: TypeAlias = np.ndarray  # [3]

REMAP_STRESS = None
STRESS_PREFACTOR = 1.0


class ChemicalSystem(pydantic.BaseModel):
    """A single atomic configuration with optional reference properties.

    Represents one snapshot of an atomistic system as produced by dataset readers
    (e.g. ExtXYZ, HDF5). Downstream, each `ChemicalSystem` is converted into a
    graph representation for model training or inference via
    `Graph.from_chemical_system`.

    Validates on construction that positions, forces, cell, and stress arrays
    have mutually consistent shapes.

    Attributes:
        atomic_numbers: Atomic numbers (Z) for every atom, shape `(N,)`.
        positions: Cartesian coordinates in Angstrom, shape `(N, 3)`.
        energy: Reference total energy in eV.
        forces: Reference per-atom forces in eV/Angstrom, shape `(N, 3)`.
        stress: Reference stress tensor in eV/Angstrom^3, shape `(3, 3)`.
        hessian: Reference energy Hessian matrix in eV/Angstrom^2, shape `(N, 3, N, 3)`
        cell: Unit-cell lattice vectors, shape `(3, 3)`.
        pbc: Per-axis periodic boundary conditions.
        weight: Relative weight of this configuration in the training loss
            (default 1.0).
        partial_charges: Atomic partial charges, shape `(N,)`.
        charge: Integer total system charge.
        spin_multiplicity: Integer total system spin multiplicity.
        dipole_moment: Dipole moment, shape `(3,)`.
        extras: Arbitrary metadata that can be consumed by custom preprocessing
            steps.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    atomic_numbers: AtomicNumbers
    positions: Positions  # Angstrom
    energy: float | None = None  # eV
    forces: Forces | None = None  # eV/Angstrom
    stress: Stress | None = None  # eV/Angstrom^3
    hessian: Hessian | None = None  # eV/Angstrom^2
    cell: Cell | None = None
    pbc: tuple[bool, bool, bool] | None = None
    weight: float = 1.0  # weight of config in loss
    partial_charges: PartialCharges | None = None
    charge: Charge | None = None
    spin_multiplicity: SpinMultiplicity | None = None
    dipole_moment: DipoleMoment | None = None
    extras: dict[str, Any] | None = None

    @pydantic.model_validator(mode="after")
    def validate_variable_shapes(self) -> Self:
        """Validates that positions and forces have the correct shape."""
        num_nodes = self.atomic_numbers.shape[0]

        if self.positions.shape != (num_nodes, 3):
            raise ValueError("Positions have incompatible shape.")

        if self.forces is not None and self.forces.shape != (num_nodes, 3):
            raise ValueError("Forces have incompatible shape.")

        return self

    @pydantic.field_validator("cell")
    @classmethod
    def validate_cell_shape(cls, value: Cell | None) -> Cell | None:
        """Validates that the cell has the correct shape."""
        if value is not None and value.shape != (3, 3):
            raise ValueError("Cell must be of shape 3x3.")
        return value

    @pydantic.field_validator("stress")
    @classmethod
    def validate_stress_shape(cls, value: Stress | None) -> Stress | None:
        """Validates that the stress has the correct shape."""
        if value is not None and value.shape != (3, 3):
            raise ValueError("Stress must be of shape 3x3.")
        return value

    @classmethod
    def from_ase_atoms(
        cls,
        atoms: ase.Atoms,
        get_property_fields: bool = True,
        property_name_mapping: dict[str, str] | None = None,
    ) -> Self:
        """Create a :class:`ChemicalSystem` from an :class:`ase.Atoms` object.

        Extracts atomic numbers, positions, cell, and periodic boundary
        conditions directly. Energy, forces, and stress are read from the
        attached calculator if available; otherwise they default to `None`. For the
        energy, forces, and stress, this can be disabled by setting
        `get_property_fields=False` in the keyword arguments.

        Args:
            atoms: An ASE `Atoms` object, optionally with a calculator
                   providing energy, forces, and/or stress.
            get_property_fields: Whether to also try fetching property fields like
                                 energy, forces, and stress from the atoms object.
                                 By default, this is set to `True`.
            property_name_mapping: Dictionary mapping from canonical property names used
                                   to access the targetted properties required by the
                                   chemical system. By default, field names are used as
                                   is. Currently, only "partial_charges", "charge",
                                   "spin_multiplicity", and "dipole_moment" are
                                   extracted through this mapping from ase.Atoms.

        Returns:
            A new `ChemicalSystem` instance.
        """
        if property_name_mapping is None:
            property_name_mapping = DEFAULT_PROPERTY_KEY_MAPPING

        def _safe_get(getter, **kwargs):
            """Try reading an ase.Atoms property and return None otherwise."""
            try:
                return getter(**kwargs)
            except (PropertyNotImplementedError, RuntimeError):
                # During inference, no calculator and no energy label
                # assigned raises RuntimeError
                return None

        atomic_numbers = np.array([
            ase_atomic_numbers_map[symbol] for symbol in atoms.symbols
        ])

        energy = forces = stress = None
        if get_property_fields:
            energy = _safe_get(atoms.get_potential_energy)
            forces = _safe_get(atoms.get_forces)
            stress = _safe_get(atoms.get_stress, voigt=False)

        cell = np.array(atoms.get_cell())
        pbc = atoms.get_pbc()

        partial_charges = atoms.info.get(property_name_mapping["partial_charges"], None)
        charge = atoms.info.get(property_name_mapping["charge"], None)
        spin_multiplicity = atoms.info.get(
            property_name_mapping["spin_multiplicity"], None
        )
        dipole_moment = atoms.info.get(property_name_mapping["dipole_moment"], None)

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

        assert np.linalg.det(cell) >= 0.0

        return cls(
            atomic_numbers=atomic_numbers,
            positions=atoms.get_positions(),
            energy=energy,
            forces=forces,
            stress=stress,
            cell=cell,
            pbc=pbc,
            partial_charges=partial_charges,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            dipole_moment=dipole_moment,
        )
