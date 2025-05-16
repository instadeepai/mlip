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
from typing import Callable, TypeVar

import numpy as np

from mlip.data.chemical_systems_readers.type_aliases import (
    ChemicalSystems,
    ChemicalSystemsBySplit,
)
from mlip.data.helpers.atomic_number_table import AtomicNumberTable

logger = logging.getLogger("mlip")


def _update_atomic_species(
    chemical_systems: ChemicalSystems, z_table: AtomicNumberTable
) -> None:
    """Update atomic species from the atomic number table."""
    to_index_fun = np.vectorize(z_table.z_to_index)
    for chemical_system in chemical_systems:
        chemical_system.atomic_species = to_index_fun(chemical_system.atomic_numbers)


def _filter_systems_with_unseen_atoms(
    chemical_systems: ChemicalSystems, z_table: AtomicNumberTable
) -> ChemicalSystems:
    """Remove systems with atoms not present in the training set."""
    original_number_systems = len(chemical_systems)
    filtered_systems = []
    for chemical_system in chemical_systems:
        if np.all(np.isin(chemical_system.atomic_numbers, z_table.zs)):
            filtered_systems.append(chemical_system)
    if len(filtered_systems) < original_number_systems:
        logger.warning(
            f"Removed {original_number_systems - len(filtered_systems)} "
            f"systems due to missing atomic species in the training set."
        )
    return filtered_systems


def filter_systems_with_unseen_atoms_and_assign_atomic_species(
    train_systems: ChemicalSystems,
    valid_systems: ChemicalSystems,
    test_systems: ChemicalSystems,
) -> ChemicalSystemsBySplit:
    """Remove systems with atoms not present in the training set
    and assign atomic species based on the train systems.

    Args:
        train_systems: Loaded train dataset in the format
                       of a list of ChemicalSystems
        valid_systems: Loaded validation dataset in the format
                       of a list of ChemicalSystems
        test_systems: Loaded test dataset in the format
                      of a list of ChemicalSystems

    Returns:
        The modified/filtered output as a tuple of train, validation and test
        datasets as a list of ``ChemicalSystem`` objects.
    """
    # Filter systems then assign atomic species
    z_table = AtomicNumberTable(
        sorted(set(np.concatenate([ts.atomic_numbers for ts in train_systems])))
    )
    valid_systems = _filter_systems_with_unseen_atoms(valid_systems, z_table)
    test_systems = _filter_systems_with_unseen_atoms(test_systems, z_table)
    _update_atomic_species(train_systems, z_table)
    _update_atomic_species(valid_systems, z_table)
    _update_atomic_species(test_systems, z_table)

    return train_systems, valid_systems, test_systems


In = TypeVar("In")
Out = TypeVar("Out")


def apply_flatten(
    fun: Callable[[In, ...], list[Out]],
    elements: list[In],
    *args,
    **kwargs,
) -> list[Out]:
    """Applies a function to each element in a list and flattens the results into
    a list."""
    if not elements:
        return []
    return [x for elem in elements for x in fun(elem, *args, **kwargs)]
