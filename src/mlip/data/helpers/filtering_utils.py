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
import functools
import logging
from typing import Callable

import numpy as np

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.type_aliases import (
    ChemicalSystems,
)
from mlip.data.helpers.type_aliases import SystemsPreprocessingFunction

logger = logging.getLogger("mlip")


def filter_systems_with_unseen_atoms(
    chemical_systems: ChemicalSystems, atomic_numbers: list[int] | np.ndarray
) -> ChemicalSystems:
    """Remove systems with atoms not present in the training set."""
    zs = (
        atomic_numbers
        if isinstance(atomic_numbers, np.ndarray)
        else np.asarray(atomic_numbers)
    )
    original_number_systems = len(chemical_systems)
    filtered_systems = []
    for chemical_system in chemical_systems:
        if np.all(np.isin(chemical_system.atomic_numbers, zs)):
            filtered_systems.append(chemical_system)
    if len(filtered_systems) < original_number_systems:
        logger.warning(
            "Removed %s systems due to missing atomic species in the training set.",
            original_number_systems - len(filtered_systems),
        )
    return filtered_systems


def filter_systems_with_unseen_charge(
    chemical_systems: ChemicalSystems, charge_values: list[int] | np.ndarray
) -> ChemicalSystems:
    """Remove systems with charge values not present in the training set."""
    charge_values = (
        charge_values
        if isinstance(charge_values, np.ndarray)
        else np.asarray(charge_values)
    )
    original_number_systems = len(chemical_systems)
    filtered_systems = []
    for chemical_system in chemical_systems:
        if chemical_system.charge is not None:
            if np.all(
                np.isin(np.asarray(chemical_system.charge).astype(int), charge_values)
            ):
                filtered_systems.append(chemical_system)
    if len(filtered_systems) < original_number_systems:
        logger.warning(
            "Removed %s systems due to missing charge values in the training set.",
            original_number_systems - len(filtered_systems),
        )
    return filtered_systems


def filter_systems_by_predicate(
    systems: ChemicalSystems,
    predicate: Callable[[ChemicalSystem], bool],
) -> ChemicalSystems:
    """Keep only those systems for which predicate function evaluates to True."""
    filtered_systems = list(filter(predicate, systems))
    return filtered_systems


def filter_systems_by_elements(
    atomic_numbers: np.ndarray | list[int] | None = None,
) -> SystemsPreprocessingFunction:
    """Return a filter function from list of atomic numbers to include."""
    return functools.partial(
        filter_systems_with_unseen_atoms, atomic_numbers=atomic_numbers
    )


def filter_systems_by_charges(
    charge_values: np.ndarray | list[int] | None = None,
) -> SystemsPreprocessingFunction:
    """Return a filter function from list of charge values to include."""
    return functools.partial(
        filter_systems_with_unseen_charge, charge_values=charge_values
    )


def filter_systems_by(
    predicate: Callable[[ChemicalSystem], bool],
) -> SystemsPreprocessingFunction:
    """Filtering post-processing function.

    Args:
        predicate: a function mapping `ChemicalSystem` to `bool`.

    Returns:
        A `PostProcessFunction` keeping only those systems for which
        the predicate evaluates to `True`.
    """
    return functools.partial(
        filter_systems_by_predicate,
        predicate=predicate,
    )


# Helper higher-order functions, mapping e.g. list[int] or list[str] to predicates.
# These are useful to parse config fields, although they do not need to live here
# (see `get_postprocess_functions()` in experiments/helper_functions/utils.py).
# One could easily add a filter on elements by exclusion, e.g. no Na/K.


def filter_systems_by_allowed_elements(
    atomic_numbers: np.ndarray | list[int] | None = None,
) -> SystemsPreprocessingFunction:
    """Return a filter function that keeps only systems with atomic numbers in the
    list.
    """
    zs_allowed = np.array(atomic_numbers)
    return filter_systems_by(
        lambda system: np.all(np.isin(system.atomic_numbers, zs_allowed)),
    )


def filter_systems_by_allowed_total_charges(
    charges: np.ndarray | list[int] | None = None,
) -> SystemsPreprocessingFunction:
    """Return a filter function that keeps only systems with total charges in the
    list.
    """
    charges_allowed = np.array(charges)
    return filter_systems_by(
        lambda system: np.all(np.isin(system.charge, charges_allowed)),
    )


def filter_systems_by_excluded_elements(
    atomic_numbers: np.ndarray | list[int] | None = None,
) -> SystemsPreprocessingFunction:
    """Return a filter function from list of atomic numbers to exclude."""
    zs_excluded = np.array(atomic_numbers)
    return filter_systems_by(
        lambda system: not np.any(np.isin(system.atomic_numbers, zs_excluded)),
    )


def filter_systems_by_excluded_total_charges(
    charges: np.ndarray | list[int] | None = None,
) -> SystemsPreprocessingFunction:
    """Return a filter function from list of total charges to exclude."""
    charges_excluded = np.array(charges)
    return filter_systems_by(
        lambda system: not np.any(np.isin(system.charge, charges_excluded)),
    )


filter_charged_systems = filter_systems_by(lambda system: system.total_charge == 0)
filter_systems_without_partial_charges = filter_systems_by(
    lambda system: system.partial_charges is not None
)
filter_systems_without_charge = filter_systems_by(
    lambda system: system.charge is not None
)


def set_system_none_charges_to_zero(
    systems: ChemicalSystems,
) -> ChemicalSystems:
    """Set the charge of a system to zero if it is None."""
    for system in systems:
        if system.charge is None:
            system.charge = 0
    return systems


# Note: Renaming this one to e.g. `process_chemical_systems` has been suggested
#       as it only applies the provided filters.
def process_chemical_systems(
    systems: ChemicalSystems,
    preprocessing_functions: list[SystemsPreprocessingFunction] | None = None,
) -> ChemicalSystems:
    """Filters the chemical systems using the provided postprocess functions
    sequentially.

    Args:
        systems: Loaded dataset in the format of a list of ChemicalSystems
        preprocessing_functions: List of functions to call to postprocess the loaded
                        dataset before returning it.
    """
    # Set default value for preprocessing_functions here instead of in the argument
    # signature to avoid silent errors with default mutable args.
    if preprocessing_functions is None:
        preprocessing_functions = []

    for preprocess in preprocessing_functions:
        systems = preprocess(systems)

    return systems
