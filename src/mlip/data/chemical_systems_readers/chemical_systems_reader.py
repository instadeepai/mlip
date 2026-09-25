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
import tempfile
from pathlib import Path
from typing import Callable

from mlip.data.chemical_systems_readers.defaults import DEFAULT_PROPERTY_KEY_MAPPING
from mlip.data.chemical_systems_readers.type_aliases import (
    ChemicalSystems,
    Source,
    Target,
)


class ChemicalSystemsReader(abc.ABC):
    """Abstract base class for reading data from disk into the internal format of lists
    of :class:`~mlip.data.chemical_system.ChemicalSystem` objects,
    one list for training data, one for validation, and one for test data.
    """

    def __init__(
        self,
        filepaths: str | os.PathLike | list[str | os.PathLike],
        data_download_fun: Callable[[Source, Target], None] | None = None,
        num_to_load: int | None = None,
        property_name_mapping: dict[str, str] | None = None,
    ):
        """Constructor.

        Args:
            filepaths: Path or paths to file from which ChemicalSystem objects
                will be read.
            data_download_fun: Optional function to download the data
                from `filepath` (source) to a local target path.
            num_to_load: Optional limit on the number of systems to
                load per file. If `None`, all systems are loaded.
            property_name_mapping: Optional mapping from canonical names
                (`"forces"`, `"energy"`, `"stress"`) to the
                keys used in the data files. By default, it will be mapped
                to the same names. Any entries provided
                will override the corresponding defaults.
        """
        self.filepaths = filepaths
        self.data_download_fun = data_download_fun
        self.num_to_load = num_to_load
        if property_name_mapping is None:
            self.property_name_mapping = DEFAULT_PROPERTY_KEY_MAPPING
        else:
            self.property_name_mapping = (
                DEFAULT_PROPERTY_KEY_MAPPING | property_name_mapping
            )

    @abc.abstractmethod
    def load(self) -> ChemicalSystems:
        """Loads chemical systems from all filepaths.

        Returns:
            A list of :class:`~mlip.data.chemical_system.ChemicalSystem`
            objects.
        """
        pass

    def _load_single_file(self, filepath: str | os.PathLike) -> ChemicalSystems:
        """Load a single file, downloading first if needed."""
        if self.data_download_fun is None:
            return self._read_file(filepath)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_filepath = Path(tmpdir) / Path(filepath).name
            self.data_download_fun(filepath, tmp_filepath)
            return self._read_file(tmp_filepath)

    @abc.abstractmethod
    def _read_file(self, filepath: str | os.PathLike) -> ChemicalSystems:
        """Read chemical systems from a single local file.

        Args:
            filepath: Path to a local file.

        Returns:
            A list of :class:`~mlip.data.chemical_system.ChemicalSystem`
            objects.
        """
        pass

    def prepare_parallel_readers(
        self, download_dir: str | os.PathLike, num_chunks: int
    ) -> list["ChemicalSystemsReader"]:
        """Convert this reader to a list of readers suitable for parallel reading.

        Resolves local paths for all files, then prepares a list of local readers.
        The base implementation returns a single reader pointed at the local files.
        Child classes that can split reading into chunks (e.g. HDF5 groups) should
        override this method to return multiple readers.

        Used by
        :func:`~mlip.data.helpers.parallel_reading.read_chemical_systems_in_parallel`
        to read and process files using multiple workers.
        """
        local_filepaths = self._resolve_local_filepaths(download_dir)
        return [self._with_local_filepaths(local_filepaths)]

    def _resolve_local_filepaths(self, download_dir: str | os.PathLike) -> list[Path]:
        """Resolve all filepaths to local paths.

        If `data_download_fun` is set, downloads all files once into `download_dir`
        so that they can be accessed by all parallel workers during processing.
        """
        filepaths = (
            self.filepaths if isinstance(self.filepaths, list) else [self.filepaths]
        )
        if self.data_download_fun is None:
            return [Path(fp) for fp in filepaths]
        download_dir = Path(download_dir)
        local_paths = []
        for i, filepath in enumerate(filepaths):
            target_dir = download_dir / str(i)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / Path(filepath).name
            self.data_download_fun(filepath, target)
            local_paths.append(target)
        return local_paths

    def _with_local_filepaths(
        self, local_filepaths: list[Path]
    ) -> "ChemicalSystemsReader":
        """Return a copy of this reader pointed at already-local files."""
        filepaths = (
            local_filepaths if isinstance(self.filepaths, list) else local_filepaths[0]
        )
        return type(self)(
            filepaths=filepaths,
            data_download_fun=None,
            num_to_load=self.num_to_load,
            property_name_mapping=self.property_name_mapping,
        )
