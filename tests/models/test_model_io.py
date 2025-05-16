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

import jax
import numpy as np

from mlip.models import Mace
from mlip.models.model_io import load_model_from_zip, save_model_to_zip
from mlip.utils.dict_flatten import flatten_dict


def test_model_can_be_saved_and_loaded_in_zip_format_correctly(
    setup_system_and_mace_model, tmp_path
):
    _, _, _, model_ff = setup_system_and_mace_model

    filepath = tmp_path / "model.zip"

    save_model_to_zip(filepath, model_ff)
    loaded_model_ff = load_model_from_zip(Mace, filepath)

    assert loaded_model_ff.config == model_ff.config

    assert jax.tree.map(np.shape, loaded_model_ff.params) == jax.tree.map(
        np.shape, model_ff.params
    )

    original_params_flattened = flatten_dict(model_ff.params)
    loaded_params_flattened = flatten_dict(loaded_model_ff.params)
    for key in original_params_flattened:
        np.testing.assert_array_equal(
            original_params_flattened[key], loaded_params_flattened[key]
        )
