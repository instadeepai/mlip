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

import contextlib
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from absl import logging as absl_logging
from orbax.checkpoint import CheckpointManager

from mlip.typing import ModelParameters
from mlip.utils.multihost import single_host_jax_and_orbax

_NETWORK_KEY = "mlip_network"


@contextlib.contextmanager
def _quiet_absl_logging():
    """Silences Orbax's INFO-level absl logs for the duration of the block."""
    previous_verbosity = absl_logging.get_verbosity()
    absl_logging.set_verbosity(absl_logging.WARNING)
    try:
        yield
    finally:
        absl_logging.set_verbosity(previous_verbosity)


def _restore_args(item: Any) -> Any:
    """Restores every leaf as a numpy array, ignoring the saved sharding."""
    return jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), item)


def _params_only_item(params_template: ModelParameters, state_metadata: Any) -> Any:
    """Builds a restore item that reads the params and skips the rest."""
    item = jax.tree.map(lambda _: ocp.PLACEHOLDER, dict(state_metadata))
    item["params"] = params_template
    return item


def _merge_leaf(path: Any, restored: Any, reference: Any) -> jax.Array:
    """Checks one restored parameter against the model's own parameter."""
    if reference.size == 0:
        return jnp.empty(reference.shape, dtype=reference.dtype)
    if restored.shape != reference.shape:
        raise ValueError(
            f"Checkpoint parameter {jax.tree_util.keystr(path)} has shape "
            f"{restored.shape}, but this model expects {reference.shape}."
        )
    return jnp.asarray(restored)


def load_parameters_from_checkpoint(
    checkpoint_dir: str | os.PathLike,
    initial_params: ModelParameters,
    epoch_to_load: int,
    load_ema_params: bool = False,
) -> ModelParameters:
    """Loads model parameters from a checkpoint.

    The parameters are restored as host numpy arrays, so the checkpoint can be
    loaded on a different number of devices than it was trained on.

    Args:
        checkpoint_dir: The directory (Orbax-compatible) where the model
                        checkpoints are stored. This directory should contain the
                        subdirectories named after the epoch numbers of the
                        checkpoints.
        initial_params: The initial parameters of the model as a template for loading.
        epoch_to_load: The epoch number to load.
        load_ema_params: Whether to load the EMA parameters instead of the standard
                         ones. By default, this is set to `False`.

    Returns:
        The loaded model parameters.

    Raises:
        ValueError: If the checkpoint does not hold exactly the parameters this
                    model expects.
    """
    item_name = "params_ema" if load_ema_params else "training_state"

    with single_host_jax_and_orbax(), _quiet_absl_logging():
        handler = ocp.PyTreeCheckpointHandler()
        ckpt_manager = CheckpointManager(
            checkpoint_dir,
            item_handlers={item_name: handler},
        )

        cpu_device = jax.devices("cpu")[0]
        with jax.default_device(cpu_device):
            # Avoid `ckpt_manager.item_metadata()` warnings about all other items.
            item_dir = Path(checkpoint_dir) / str(epoch_to_load) / item_name
            metadata = handler.metadata(item_dir)
            params_metadata = metadata if load_ema_params else metadata["params"]

            is_old_params_version = _NETWORK_KEY not in params_metadata["params"]
            params_template = (
                {"params": initial_params["params"][_NETWORK_KEY]}
                if is_old_params_version
                else initial_params
            )

            item = (
                params_template
                if load_ema_params
                else _params_only_item(params_template, metadata)
            )
            restored = ckpt_manager.restore(
                epoch_to_load,
                args=ocp.args.Composite(**{
                    item_name: ocp.args.PyTreeRestore(
                        item=item, restore_args=_restore_args(item)
                    )
                }),
            )[item_name]

    params = restored if load_ema_params else restored["params"]
    params = jax.tree_util.tree_map_with_path(_merge_leaf, params, params_template)

    if is_old_params_version:
        return {"params": {_NETWORK_KEY: params["params"]}}
    return params
