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

import logging

import ase
import jax
import jax.numpy as jnp
import jax_md
import numpy as np
from jax import Array

from mlip.models import ForceField

EXPLODED_TEMPERATURE_THRESHOLD = 1e6
logger = logging.getLogger("mlip")


def has_simulation_exploded(temperatures: np.ndarray | float) -> bool:
    """Whether a simulation has exploded based on its temperature."""
    if isinstance(temperatures, float):
        temperatures = np.array([temperatures])

    if jnp.isnan(temperatures).any() or jnp.any(
        jnp.abs(temperatures) > EXPLODED_TEMPERATURE_THRESHOLD
    ):
        return True
    return False


def resolve_atoms_cell(atoms: ase.Atoms, box: float | list[float] | None) -> ase.Atoms:
    """Set the cell/PBCs on an `ase.Atoms` object from a config `box` value.

    If `atoms` already has a cell/PBC configured, this is left untouched.

    Args:
        atoms: The atomic structure to update, mutated in place.
        box: Specifies the PBCs to use if not already configured for `atoms`. A float
            (cubic box side-length), list of floats (lattice vector lengths), or `None`.

    Returns:
        The same atoms object, with `atoms.cell`/`atoms.pbc` resolved.
    """
    if np.any(atoms.cell) or np.any(atoms.pbc):
        logger.warning(
            "Ignoring `box` parameter as `atoms` already has PBC configured."
        )
        return atoms

    if isinstance(box, float):
        atoms.cell = np.eye(3) * box
        atoms.pbc = True
    elif isinstance(box, list):
        atoms.cell = np.diag(np.array(box))
        atoms.pbc = True
    else:
        atoms.cell = None
        atoms.pbc = False

    return atoms


def resolve_atoms_charge_for_model(
    atoms: ase.Atoms | list[ase.Atoms],
    force_field: ForceField,
    set_none_charge_to_zero: bool,
) -> ase.Atoms | list[ase.Atoms]:
    """Resolve the total charge on one or more `ase.Atoms` for a charge-embedding model.

    Args:
        atoms: The atomic structure(s) whose charge may be resolved.
        force_field: The force field used in the simulation.
        set_none_charge_to_zero: Whether to treat missing charge as 0.

    Returns:
        The atoms object(s), with `atoms.info['charge']` resolved when needed.

    Raises:
        ValueError: If the force field uses total charge embedding and no charge
            can be resolved for any of the structures.
    """
    if isinstance(atoms, list):
        return [
            _resolve_single_atoms_charge(a, force_field, set_none_charge_to_zero)
            for a in atoms
        ]
    return _resolve_single_atoms_charge(atoms, force_field, set_none_charge_to_zero)


def _resolve_single_atoms_charge(
    atoms: ase.Atoms,
    force_field: ForceField,
    set_none_charge_to_zero: bool,
) -> ase.Atoms:
    if not getattr(force_field.config, "use_total_charge_embedding", False):
        return atoms

    charge = atoms.info.get("charge")
    if charge is None:
        if atoms.info.get("partial_charges") is not None:
            charge = int(np.round(np.sum(atoms.info["partial_charges"])))
        elif set_none_charge_to_zero:
            logger.warning(
                "Input system has no charge assigned, but the model uses total "
                "charge embedding. Setting to 0 as `set_none_charge_to_zero=True`, "
                "however this may affect simulation quality if not correct. "
                "Consider setting the charge explicitly as `atoms.info['charge']`."
            )
            charge = 0
        else:
            raise ValueError(
                "The model uses total charge embedding, but the input system has "
                "no charge assigned. Either assign an explicit charge as "
                "`atoms.info['charge']`, or set `set_none_charge_to_zero=True`."
            )

    atoms.info["charge"] = charge
    return atoms


def _apply_box_transform(box: Array, R: Array) -> Array:
    """High-precision replacement for `jax_md.space.raw_transform`.

    Used to prevent compounding errors when using `raw_transform` with lower precision.
    """
    with jax.default_matmul_precision("highest"):
        return jax_md.space.raw_transform(box, R)


def fractional_to_positions(
    positions: np.ndarray | list[np.ndarray],
    box: np.ndarray | list[np.ndarray],
) -> np.ndarray | list[np.ndarray]:
    """Convert fractional coordinates to real-space positions.

    Args:
        positions: Fractional positions. Either a single array or a list.
        box: Box-representation of the system cell. Either a single array or a list.

    Returns:
        Real-space positions with the same structure as the input.
    """

    def _to_real(pos: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _apply_box_transform(b, pos).astype(pos.dtype)

    return jax.tree.map(_to_real, positions, box)


def positions_to_fractional(
    positions: np.ndarray | list[np.ndarray],
    box: np.ndarray | list[np.ndarray],
) -> np.ndarray | list[np.ndarray]:
    """Convert real-space positions to fractional coordinates.

    Args:
        positions: Real-space positions. Either a single array or a list.
        box: Box-representation of the system cell. Either a single array or a list.

    Returns:
        Fractional positions with the same structure as the input.
    """

    def _to_frac(pos: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _apply_box_transform(jax_md.space.inverse(b), pos).astype(pos.dtype)

    return jax.tree.map(_to_frac, positions, box)
