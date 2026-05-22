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
import jax.numpy as jnp
import numpy as np

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
