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
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from mlip.data import DatasetInfo
from mlip.models import ForceField, ForceFieldPredictor
from mlip.utils.dict_flatten import flatten_dict, unflatten_dict

PARAMETER_MODULE_DELIMITER = "#"
MODEL_HYPERPARAMS_FILENAME = "hyperparams.json"
MODEL_PARAMETERS_FILENAME = "params.npz"


def save_model_to_zip(
    save_path: str | os.PathLike,
    model: ForceField,
) -> None:
    """Saves a force field model to a zip archive in a
    lightweight format to be easily loaded back for inference later.

    Args:
        save_path: The target path to the zip archive. Should have extension ".zip".
        model: The force field model to save.
               Must be passed as type :class:`~mlip.models.force_field.ForceField`.
    """
    hyperparams = {
        "dataset_info": json.loads(model.dataset_info.model_dump_json()),
        "config": json.loads(model.config.model_dump_json()),
        "predict_stress": model.predictor.predict_stress,
    }

    params_flattened = {
        PARAMETER_MODULE_DELIMITER.join(key_as_tuple): array
        for key_as_tuple, array in flatten_dict(model.params).items()
    }

    with TemporaryDirectory() as tmpdir:
        hyperparams_path = Path(tmpdir) / MODEL_HYPERPARAMS_FILENAME
        params_path = Path(tmpdir) / MODEL_PARAMETERS_FILENAME

        with open(hyperparams_path, "w") as json_file:
            json.dump(hyperparams, json_file)

        np.savez(params_path, **params_flattened)

        with ZipFile(save_path, "w") as zip_object:
            zip_object.write(hyperparams_path, os.path.basename(hyperparams_path))
            zip_object.write(params_path, os.path.basename(params_path))


def load_model_from_zip(
    model_type: type(nn.Module),
    load_path: str | os.PathLike,
) -> ForceField:
    """Loads a model from a zip archive and returns it wrapped as a `ForceField`.

    Args:
        model_type: The model class that corresponds to the saved model.
        load_path: The path to the zip archive to load.

    Returns:
        The loaded model wrapped
        as a :class:`~mlip.models.force_field.ForceField` object.
    """
    with ZipFile(load_path, "r") as zip_object:
        with zip_object.open(MODEL_HYPERPARAMS_FILENAME, "r") as json_file:
            hyperparams_raw = json.load(json_file)
        with zip_object.open(MODEL_PARAMETERS_FILENAME, "r") as params_file:
            params_raw = np.load(params_file)
            params = unflatten_dict(
                {
                    tuple(key.split(PARAMETER_MODULE_DELIMITER)): jnp.asarray(
                        params_raw[key]
                    )
                    for key in params_raw.files
                }
            )

    model_config = model_type.Config(**hyperparams_raw["config"])
    model = model_type(
        config=model_config, dataset_info=DatasetInfo(**hyperparams_raw["dataset_info"])
    )
    predictor = ForceFieldPredictor(model, hyperparams_raw["predict_stress"])
    return ForceField(predictor, params)
