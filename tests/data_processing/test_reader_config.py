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

import pytest
from pydantic import ValidationError

from mlip.data.configs import ChemicalSystemsReaderConfig

DATA_DIR = Path(__file__).parent.parent / "data"
SPICE_SMALL_HDF5_PATH = DATA_DIR / "spice2-1000_429_md_0-1.hdf5"


@pytest.mark.parametrize(
    "in_paths, expected_type",
    [
        (str(SPICE_SMALL_HDF5_PATH.resolve()), str),
        (SPICE_SMALL_HDF5_PATH.resolve(), Path),
        ([str(SPICE_SMALL_HDF5_PATH.resolve())], str),
        ([SPICE_SMALL_HDF5_PATH.resolve()], Path),
    ],
)
def test_reader_config_paths_are_converted_to_lists(in_paths, expected_type):
    config = ChemicalSystemsReaderConfig(
        reader_type="extxyz",
        train_dataset_paths=in_paths,
        valid_dataset_paths=[],
        test_dataset_paths=None,
    )

    assert isinstance(config.train_dataset_paths, list)
    assert len(config.train_dataset_paths) == 1
    assert isinstance(config.train_dataset_paths[0], expected_type)

    assert isinstance(config.valid_dataset_paths, list)
    assert len(config.valid_dataset_paths) == 0

    assert isinstance(config.test_dataset_paths, list)
    assert len(config.test_dataset_paths) == 0


def test_reader_config_train_path_must_be_defined():
    with pytest.raises(
        ValidationError,
    ):
        ChemicalSystemsReaderConfig(
            train_dataset_paths=[],
            valid_dataset_paths=[],
            test_dataset_paths=[],
        )
