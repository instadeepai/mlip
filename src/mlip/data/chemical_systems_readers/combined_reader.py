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

from typing import Callable, Optional

from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)
from mlip.data.chemical_systems_readers.type_aliases import (
    ChemicalSystems,
    ChemicalSystemsBySplit,
)
from mlip.data.chemical_systems_readers.utils import (
    filter_systems_with_unseen_atoms_and_assign_atomic_species,
)


class CombinedReader:
    """Wrapper for a list of
    :class:`~mlip.data.chemical_systems_readers.chemical_systems_reader.ChemicalSystemsReader`
    that combines the result of loading data from each.
    """

    def __init__(self, chemical_systems_readers: list[ChemicalSystemsReader]):
        """Constructor.

        Args:
            chemical_systems_readers: The list of readers to use to load data.
        """
        self.chemical_systems_readers = chemical_systems_readers

    def load(
        self,
        postprocess_fun: Optional[
            Callable[
                [ChemicalSystems, ChemicalSystems, ChemicalSystems],
                ChemicalSystemsBySplit,
            ]
        ] = filter_systems_with_unseen_atoms_and_assign_atomic_species,
    ) -> ChemicalSystemsBySplit:
        """Loads the datasets into their internal formats and combines the
        resulting lists of ``ChemicalSystems``.

        Args:
            postprocess_fun: Function to call to postprocess the loaded dataset
                            before returning it. Accepts train, validation and test
                            systems (``list[ChemicalSystems]``), runs some
                            postprocessing (filtering for example) and
                            returns the postprocessed train, validation and test
                            systems.
                            If ``postprocess_fun`` is ``None`` then no postprocessing
                            will be done. By default, it will run
                            :meth:`~mlip.data.chemical_systems_readers.utils.assign_atomic_species_and_filter_systems_with_unseen_atoms`
                            which assigns atomic species on ``ChemicalSystem`` objects
                            and filters out systems from the validation
                            and test sets that contain chemical elements that
                            are not present in the train systems.
        Returns:
            A tuple of loaded training, validation and test datasets (in this order).
            The internal format is a list of ``ChemicalSystem`` objects.
        """
        """"""
        train_systems, valid_systems, test_systems = [], [], []
        for reader in self.chemical_systems_readers:
            train_sys, valid_sys, test_sys = reader.load(
                postprocess_fun=postprocess_fun
            )
            train_systems += train_sys
            valid_systems += valid_sys
            test_systems += test_sys

        return train_systems, valid_systems, test_systems
