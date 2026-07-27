# Copyright 2025 Zhongguancun Academy
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
from typing import SupportsIndex, Sequence

import grain

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)


class ChemicalSystemsDataset(abc.ABC):
    """Grain-compatible random-access source of :class:`ChemicalSystem`. Extends
    :class:`~mlip.data.chemical_systems_readers.chemical_systems_reader.ChemicalSystemsReader`
    for streaming dataloaders."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Number of chemical systems."""

    @abc.abstractmethod
    def __getitem__(self, index: SupportsIndex) -> ChemicalSystem:
        """Return the chemical system at ``index``."""


def map_dataset_from_readers(
    readers: ChemicalSystemsDataset | list[ChemicalSystemsDataset]
) -> grain.MapDataset:
    """Build a Grain ``MapDataset`` from streaming-capable readers.

    Raises:
        TypeError: If any reader does not support random access for streaming.
    """
    reader_list = readers if isinstance(readers, list) else [readers]
    if not reader_list:
        raise ValueError("At least one reader is required.")

    datasets: list[ChemicalSystemsDataset] = []
    for reader in reader_list:
        if not isinstance(reader, ChemicalSystemsDataset):
            raise TypeError(
                f"{type(reader).__name__} does not support random access for "
                "streaming. Inherit from "
                "mlip.data.chemical_systems_readers.dataset.Dataset or use "
                "keep_in_memory=True."
            )
        if (
            isinstance(reader, ChemicalSystemsReader)
            and reader.data_download_fun is not None
        ):
            raise TypeError(
                f"{type(reader).__name__} with data_download_fun cannot be used "
                "for streaming; download data locally first."
            )
        datasets.append(grain.MapDataset.source(reader))

    if len(datasets) == 1:
        return datasets[0]
    return grain.MapDataset.concatenate(datasets)


def filter_excluded_indices(
    chemical_dataset: grain.MapDataset, exclude_ids: Sequence[int]
) -> grain.MapDataset:
    """Return a ``MapDataset`` view that skips ``exclude_ids``."""
    if not exclude_ids:
        return chemical_dataset

    exclude = set(int(i) for i in exclude_ids)
    keep = [i for i in range(len(chemical_dataset)) if i not in exclude]
    if not keep:
        raise ValueError(
            "All graphs were discarded due to size/preprocessing filters."
        )

    return grain.MapDataset.source(keep).map(
        lambda idx, ds=chemical_dataset: ds[int(idx)]
    )
