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

import json
from zipfile import ZipFile

import jax
import numpy as np

from mlip import __version__ as mlip_version
from mlip.models import Mace
from mlip.models.inference_context import InferenceContext
from mlip.models.model_io import (
    MODEL_HYPERPARAMS_FILENAME,
    load_model_from_zip,
    save_model_to_zip,
)
from mlip.utils.dict_flatten import flatten_dict


def test_model_can_be_saved_and_loaded_in_zip_format_correctly(
    mace_force_field, tmp_path
):
    model_ff = mace_force_field

    filepath = tmp_path / "model.zip"

    save_model_to_zip(filepath, model_ff)
    loaded_model_ff = load_model_from_zip(Mace, filepath)

    assert loaded_model_ff.config == model_ff.config
    assert (
        loaded_model_ff.predictor.mlip_network.available_properties
        == model_ff.predictor.mlip_network.available_properties
    )

    assert jax.tree.map(np.shape, loaded_model_ff.params) == jax.tree.map(
        np.shape, model_ff.params
    )

    original_params_flattened = flatten_dict(model_ff.params)
    loaded_params_flattened = flatten_dict(loaded_model_ff.params)
    for key in original_params_flattened:
        np.testing.assert_array_equal(
            original_params_flattened[key], loaded_params_flattened[key]
        )

    # Assert that library version is saved alongside the model
    with ZipFile(filepath, "r") as zip_object:
        with zip_object.open(MODEL_HYPERPARAMS_FILENAME, "r") as json_file:
            hyperparams_raw = json.load(json_file)
    assert "version" in hyperparams_raw
    assert hyperparams_raw["version"] == mlip_version


def test_load_model_from_zip_resolves_inference_context(
    multi_head_mace_force_field,
    tmp_path,
):
    filepath = tmp_path / "model.zip"

    save_model_to_zip(filepath, multi_head_mace_force_field)
    loaded_model_ff = load_model_from_zip(
        Mace,
        filepath,
        inference_context=InferenceContext(dataset_name="dataset_1"),
    )

    assert loaded_model_ff.inference_context == InferenceContext(
        dataset_name="dataset_1",
        dataset_idx=1,
    )
