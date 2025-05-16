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

import abc
import os
from typing import Callable, Optional, TypeAlias

from mlip.data.chemical_systems_readers.type_aliases import (
    ChemicalSystems,
    ChemicalSystemsBySplit,
)
from mlip.data.chemical_systems_readers.utils import (
    filter_systems_with_unseen_atoms_and_assign_atomic_species,
)
from mlip.data.configs import ChemicalSystemsReaderConfig

Source: TypeAlias = str | os.PathLike
Target: TypeAlias = str | os.PathLike


class ChemicalSystemsReader(abc.ABC):
    """Abstract base class for reading data from disk into the internal format of lists
    of :class:`~mlip.data.chemical_system.ChemicalSystem` objects,
    one list for training data, one for validation, and one for test data.
    """

    Config = ChemicalSystemsReaderConfig

    def __init__(
        self,
        config: ChemicalSystemsReaderConfig,
        data_download_fun: Optional[Callable[[Source, Target], None]] = None,
    ):
        """Constructor.

        Args:
            config: The configuration defining how and where to load the data from.
            data_download_fun: A function to download data from an external remote
                               system. If ``None`` (default), then this class assumes
                               file paths are local. This function must take two paths
                               as input, source and target, and download the data at
                               source into the target location.
        """
        self.config = config
        self.data_download_fun = data_download_fun

    @abc.abstractmethod
    def load(
        self,
        postprocess_fun: Optional[
            Callable[
                [ChemicalSystems, ChemicalSystems, ChemicalSystems],
                ChemicalSystemsBySplit,
            ]
        ] = filter_systems_with_unseen_atoms_and_assign_atomic_species,
    ) -> ChemicalSystemsBySplit:
        """Loads the dataset into its internal format.

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
        pass
