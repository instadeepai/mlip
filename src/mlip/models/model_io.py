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
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZipFile

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from mlip import __version__ as mlip_version
from mlip.data import DatasetInfo
from mlip.data.helpers.dummy_init_graph import get_dummy_graph_for_model_init
from mlip.models import ForceField
from mlip.models.inference_context import InferenceContext
from mlip.models_v1.mlip_network_v1 import MLIPNetworkV1
from mlip.models_v1.predictor_v1 import ForceFieldPredictorV1
from mlip.typing.properties import Properties
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

    Additionally, this function saves the current library version into the zip.

    Args:
        save_path: The target path to the zip archive. Should have extension ".zip".
        model: The force field model to save.
               Must be passed as type :class:`~mlip.models.force_field.ForceField`.
    """
    hyperparams = {
        "dataset_info": json.loads(model.dataset_info.model_dump_json()),
        "config": json.loads(model.config.model_dump_json()),
        "available_properties": asdict(
            model.predictor.mlip_network.available_properties
        ),
        "version": mlip_version,
    }

    params_flattened = {
        PARAMETER_MODULE_DELIMITER.join(key_as_tuple): array
        for key_as_tuple, array in flatten_dict(model.params).items()
    }

    with TemporaryDirectory() as tmpdir:
        hyperparams_path = Path(tmpdir) / MODEL_HYPERPARAMS_FILENAME
        params_path = Path(tmpdir) / MODEL_PARAMETERS_FILENAME

        with open(hyperparams_path, "w", encoding="utf-8") as json_file:
            json.dump(hyperparams, json_file)

        np.savez(params_path, **params_flattened)

        with ZipFile(save_path, "w") as zip_object:
            zip_object.write(hyperparams_path, os.path.basename(hyperparams_path))
            zip_object.write(params_path, os.path.basename(params_path))


def _process_loaded_dataset_info_for_v1_compatibility(
    loaded_dataset_info: dict[str, Any],
) -> dict[str, Any]:
    """The loaded dataset info dictionary needs to be transformed from its v1 to v2
    naming conventions if it is still using the old convention.
    """
    old_cutoff_name = "cutoff_distance_angstrom"
    new_cutoff_name = "graph_cutoff_angstrom"
    if old_cutoff_name in loaded_dataset_info:
        loaded_dataset_info[new_cutoff_name] = loaded_dataset_info[old_cutoff_name]
        del loaded_dataset_info[old_cutoff_name]
    return loaded_dataset_info


def load_model_from_zip(
    model_type: type(nn.Module),
    load_path: str | os.PathLike,
    required_properties: Properties | None = None,
    inference_context: InferenceContext | None = None,
) -> ForceField:
    """Loads a model from a zip archive and returns it wrapped as a `ForceField`.

    Args:
        model_type: The model class that corresponds to the saved model.
        load_path: The path to the zip archive to load.
        required_properties: The properties required from the loaded model. Default is
                             `None` which means that a default `Properties` object will
                             be used (i.e. energy and forces). Set explicitly if you
                             require stress, hessians, or other outputs.
        inference_context: Optional context to apply to the loaded force field for
                           inference. For MoE models, experts are contracted for this
                           context before the force field is returned.
    Returns:
        The loaded model wrapped
        as a :class:`~mlip.models.force_field.ForceField` object.
    """
    with ZipFile(load_path, "r") as zip_object:
        with zip_object.open(MODEL_HYPERPARAMS_FILENAME, "r") as json_file:
            hyperparams_raw = json.load(json_file)
        with zip_object.open(MODEL_PARAMETERS_FILENAME, "r") as params_file:
            params_raw = np.load(params_file)
            params = unflatten_dict({
                tuple(key.split(PARAMETER_MODULE_DELIMITER)): jnp.asarray(
                    params_raw[key]
                )
                for key in params_raw.files
            })

    model_config = model_type.Config(**hyperparams_raw["config"])

    if required_properties is None:
        required_properties = Properties()

    loaded_dataset_info = _process_loaded_dataset_info_for_v1_compatibility(
        hyperparams_raw["dataset_info"]
    )
    model = model_type(
        config=model_config,
        dataset_info=DatasetInfo(**loaded_dataset_info),
    )

    # Loading an old v1 model
    if isinstance(model, MLIPNetworkV1):
        if inference_context is not None:
            raise ValueError(
                "Inference context is not supported when loading v1 models."
            )
        jax_init_key = jax.random.key(123)
        v1_predictor = ForceFieldPredictorV1(
            mlip_network=model, required_properties=required_properties
        )
        init_params = v1_predictor.init(jax_init_key, get_dummy_graph_for_model_init())

        # Remove leading device axis if V1 model was trained with pmap.
        def _drop_device_dim(loaded, ref):
            if loaded.ndim == ref.ndim + 1:
                return loaded[0]
            return loaded

        params = jax.tree.map(_drop_device_dim, params, init_params)
        return ForceField(v1_predictor, params)

    # Loading a v2 model
    force_field = ForceField.from_mlip_network(
        model,
        required_properties,
        inference_context=inference_context,
    )

    # Parameter blocks can be zero-size (e.g. unused irreps). Re-initialize zero-size
    # leaves from the freshly-constructed reference so the shape and dtype are correct.
    def _restore_empty(loaded, reference):
        return (
            jnp.empty(reference.shape, dtype=reference.dtype)
            if reference.size == 0
            else loaded
        )

    params = jax.tree.map(_restore_empty, params, force_field.params)

    loaded_force_field = ForceField(
        force_field.predictor,
        params,
        inference_context=force_field.inference_context,
    )
    if (
        loaded_force_field.inference_context is not None
        and loaded_force_field.predictor.mlip_network.is_moe_model
    ):
        return loaded_force_field.prepare_experts_for_inference()
    return loaded_force_field
