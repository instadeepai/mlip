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

import dataclasses
import random
from typing import Any, Callable


@dataclasses.dataclass
class DataSplitProportions:
    """Dataclass holding data split proportions."""

    train: float
    validation: float
    test: float


class SplitProportionsInvalidError(Exception):
    """Exception to be raised if data split proportions don't sum up to one."""


class GroupIDNotInSplitError(Exception):
    """Exception to be raised if group ID not found in the given splits."""


def _validate_proportions(proportions: DataSplitProportions) -> None:
    _proportions = dataclasses.asdict(proportions).values()
    if sum(_proportions) != 1.0 or any(prop > 1.0 for prop in _proportions):
        raise SplitProportionsInvalidError("Data split proportions are invalid.")

    if proportions.train == 0.0:
        raise SplitProportionsInvalidError("Training set must be non-empty.")


def split_data_randomly(
    data: list[Any], proportions: DataSplitProportions, seed: int
) -> tuple[list[Any], list[Any], list[Any]]:
    """Splits the data randomly.

    Args:
        data: The data, which must be a list of any object.
        proportions: The dataset proportions. These must sum to one and none of these
                     can be larger than one. The train proportion must also be greater
                     than zero.
        seed: The random seed for the split.

    Returns:
        The split data, which are three lists of the objects, referring to training set,
        validation set, and test set. The latter two can be empty if the given
        proportions were zero.
    """
    _validate_proportions(proportions)
    random.seed(seed)
    random.shuffle(data)

    num_data = len(data)
    num_train_data = int(proportions.train * num_data)
    num_test_data = int(proportions.test * num_data)

    train_set = data[:num_train_data]
    test_set = data[-num_test_data:]

    # make sure validation set is empty if proportion was zero
    # regardless of rounding errors above
    validation_set = (
        [] if proportions.validation == 0.0 else data[num_train_data:-num_test_data]
    )

    return train_set, validation_set, test_set


def split_data_randomly_by_group(
    data: list[Any],
    proportions: DataSplitProportions,
    seed: int,
    get_group_id_fun: Callable[[Any], str],
    placeholder_group_id: str,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Splits the data randomly, but by respecting some groups.

    This means that data points that belong to the same group must end up in the
    same split. The grouping mechanism can be provided via the get_group_id_fun
    parameter.

    Args:
        data: The data, which must be a list of any object.
        proportions: The dataset proportions. These must sum to one and none of these
                     can be larger than one. The train proportion must also be greater
                     than zero.
        seed: The random seed for the split.
        get_group_id_fun: This function takes in one of the objects (data points) and
                          returns a string representation of its group.
        placeholder_group_id: This group is for any data that does not belong to a
                              predefined group. The data belonging to this group
                              will be assigned to the training set.

    Returns:
        The split data, which are three lists of the objects, referring to training set,
        validation set, and test set. The latter two can be empty if the given
        proportions were zero. Note that the proportions may not be exactly as
        requested in the input, but close.
    """
    _validate_proportions(proportions)

    # convert the set to a sorted list for reproducibility
    groups = sorted({get_group_id_fun(structure) for structure in data})
    if placeholder_group_id in groups:
        groups.remove(placeholder_group_id)

    random.seed(seed)
    random.shuffle(groups)

    num_groups = len(groups)
    num_train_groups = int(proportions.train * num_groups)
    num_test_groups = int(proportions.test * num_groups)

    train_groups = set(groups[:num_train_groups] + [placeholder_group_id])
    test_groups = set(groups[-num_test_groups:])

    # make sure validation groups are empty if proportion was zero
    # regardless of rounding errors above
    validation_groups = set(
        []
        if proportions.validation == 0.0
        else groups[num_train_groups:-num_test_groups]
    )

    train_set = []
    validation_set = []
    test_set = []
    for structure in data:
        if get_group_id_fun(structure) in train_groups:
            train_set.append(structure)
        elif get_group_id_fun(structure) in validation_groups:
            validation_set.append(structure)
        elif get_group_id_fun(structure) in test_groups:
            test_set.append(structure)
    return train_set, validation_set, test_set


def split_data_by_group(
    data: list[Any],
    group_ids_by_split: tuple[set[str], set[str], set[str]],
    get_group_id_fun: Callable[[Any], str],
) -> tuple[list[Any], list[Any], list[Any]]:
    """Splits the data into groups (train, val, test) with the ``get_group_id_fun``
    based on the ``group_ids_by_split``.

    If there's a data point with a group_id that doesn't belong to
    the split (train, val, test), an exception will be raised.

    Args:
        data: The data, which must be a list of any object.
        group_ids_by_split: Tuple of sets of group IDs by the split (train, val, test).
        get_group_id_fun: This function takes in one of the objects (data points) and
                          returns a string representation of its group.

    Returns:
        The split data, which are three lists of the objects, referring to training set,
        validation set, and test set. The latter two can be empty if the given
        split was empty.
    """
    train_group_ids, val_group_ids, test_group_ids = group_ids_by_split
    train_set = []
    validation_set = []
    test_set = []
    unknown_group_ids = set()
    for structure in data:
        group_id = get_group_id_fun(structure)
        if group_id in train_group_ids:
            train_set.append(structure)
        elif group_id in val_group_ids:
            validation_set.append(structure)
        elif group_id in test_group_ids:
            test_set.append(structure)
        else:
            unknown_group_ids.add(group_id)

    if unknown_group_ids:
        raise GroupIDNotInSplitError(
            f"Found {len(unknown_group_ids)} unexpected group IDs: "
            f"{sorted(unknown_group_ids)}"
        )

    return train_set, validation_set, test_set
