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

from pathlib import Path
from typing import Optional

import pydantic
from omegaconf import ListConfig
from pydantic import field_validator, model_validator
from typing_extensions import Annotated, Self

PositiveInt = Annotated[int, pydantic.Field(gt=0)]
PositiveFloat = Annotated[float, pydantic.Field(gt=0)]


class ChemicalSystemsReaderConfig(pydantic.BaseModel):
    """Pydantic-based config related to data preprocessing and loading into
    `ChemicalSystem`s.

    Attributes:
        train_dataset_paths: Path(s) to where the training set(s) are located.
                            Cannot be empty.
                            Will be converted to a list after validation.
        valid_dataset_paths: Path(s) to where the validation set(s) are located.
                            This can be empty.
                            Will be converted to a list after validation.
        test_dataset_paths: Path(s) to where the test set(s) are located.
                            This can be empty.
                            Will be converted to a list after validation.
        train_num_to_load: Number of training set data points to load from the given
                           dataset. By default, this is ``None`` which means all the
                           data points are loaded.
                           If multiple dataset paths are given, then this limit will
                           apply to each path separately, not in total.
        valid_num_to_load: Number of validation set data points to load from the given
                           dataset. By default, this is ``None`` which means all the
                           data points are loaded.
                           If multiple dataset paths are given, then this limit will
                           apply to each path separately, not in total.
        test_num_to_load: Number of test set data points to load from the given
                           dataset. By default, this is ``None`` which means all the
                           data points are loaded.
                           If multiple dataset paths are given, then this limit will
                           apply to each path separately, not in total.
    """

    train_dataset_paths: str | Path | list[str | Path]
    valid_dataset_paths: str | Path | list[str | Path] | None = None
    test_dataset_paths: str | Path | list[str | Path] | None = None

    train_num_to_load: Optional[PositiveInt] = None
    valid_num_to_load: Optional[PositiveInt] = None
    test_num_to_load: Optional[PositiveInt] = None

    @field_validator(
        "train_dataset_paths",
        "valid_dataset_paths",
        "test_dataset_paths",
        mode="before",
    )
    @classmethod
    def convert_to_list(
        cls, value: str | Path | list[str | Path] | ListConfig | None
    ) -> list[str | Path]:
        """Support single element input for a list field, by converting it to a list
        internally to simplify usage."""
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            return [value]
        if isinstance(value, list):
            return value
        if isinstance(value, ListConfig):
            return list(value)
        raise ValueError(
            f"*_dataset_paths must be a string, Path, or a list of them, "
            f"but was {type(value)} - {value}"
        )

    @model_validator(mode="after")
    def validate_train_dataset_paths(self) -> Self:
        if self.train_dataset_paths == []:
            raise ValueError("Train dataset paths should contain at least one path")
        return self


class GraphDatasetBuilderConfig(pydantic.BaseModel):
    """Pydantic-based config related to graph dataset building and preprocessing.

    Attributes:
        graph_cutoff_angstrom: Graph cutoff distance in Angstrom to apply when
                               creating the graphs. Default is 5.0.
        max_n_node: This value will be multiplied with the batch size to determine the
                    maximum number of nodes we allow in a batch.
                    Note that a batch will always contain max_n_node * batch_size
                    nodes, as the remaining ones are filled up with dummy nodes.
                    If set to `None`, a reasonable value will be automatically
                    computed. Default is `None`.
        max_n_edge: This value will be multiplied with the batch size to determine the
                    maximum number of edges we allow in a batch.
                    Note that a batch will always contain max_n_edge * batch_size
                    edges, as the remaining ones are filled up with dummy edges.
                    If set to `None`, a reasonable value will be automatically
                    computed. Default is `None`.
        batch_size: The number of graphs in a batch. Will be filled up with dummy graphs
                    if either the maximum number of nodes or edges are reached before
                    the number of graphs is reached. Default is 16.
        num_batch_prefetch: Number of batched graphs to prefetch while iterating
                            over batches. Default is 1.
        batch_prefetch_num_devices: Number of threads to use for prefetching.
                                    Default is 1.
        use_formation_energies: Whether the energies in the dataset should already be
                                transformed to subtract the average atomic energies.
                                Default is ``False``. Make sure that if you set this
                                to ``True``, the models assume ``"zero"`` atomic
                                energies as can be set in the model hyperparameters.
        avg_num_neighbors: The pre-computed average number of neighbors.
        avg_r_min_angstrom: The pre-computed average minimum distance between nodes.

    """

    graph_cutoff_angstrom: PositiveFloat = 5.0
    max_n_node: Optional[PositiveInt] = None
    max_n_edge: Optional[PositiveInt] = None
    batch_size: PositiveInt = 16

    num_batch_prefetch: PositiveInt = 1
    batch_prefetch_num_devices: PositiveInt = 1

    use_formation_energies: bool = False
    avg_num_neighbors: Optional[float] = None
    avg_r_min_angstrom: Optional[float] = None
