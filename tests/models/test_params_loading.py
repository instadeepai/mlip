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

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from mlip.models.params_loading import load_parameters_from_checkpoint
from mlip.training.ema import exponentially_moving_average
from mlip.training.training_state import init_training_state

CHECKPOINT_SUBDIR = "model"
TESTS_ROOT = str(Path(__file__).resolve().parents[1])


def _params(value: float, num_features: int = 4) -> dict:
    """A parameter tree shaped like a real one: nested, with a skipped block."""
    return {
        "params": {
            "mlip_network": {
                "ReadoutBlock_0": {
                    "linear": {"kernel": jnp.full((8, num_features), value)},
                    "skipped": jnp.zeros((0, 3)),
                }
            }
        }
    }


def _save_checkpoint(directory, params: dict, save_ema: bool = False) -> None:
    """Writes a checkpoint the same way `training.checkpointer` does."""
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3))
    training_state = init_training_state(
        params, optimizer, exponentially_moving_average(0.99)
    )

    def serialize(tree):
        """Mirrors `checkpointer._serialize_empty`: zero-size leaves are stored
        as a scalar sentinel."""
        return jax.tree.map(
            lambda t: t if not hasattr(t, "size") or t.size else jnp.array(False),
            tree,
        )

    item_names = ["training_state"] + (["params_ema"] if save_ema else [])
    manager = ocp.CheckpointManager(directory, item_names=item_names)
    to_save = {"training_state": ocp.args.PyTreeSave(serialize(training_state))}
    if save_ema:
        to_save["params_ema"] = ocp.args.PyTreeSave(serialize(params))
    manager.save(1, args=ocp.args.Composite(**to_save))
    manager.wait_until_finished()


def test_params_are_restored_with_values_and_shapes(tmp_path):
    _save_checkpoint(tmp_path / CHECKPOINT_SUBDIR, _params(3.0))

    loaded = load_parameters_from_checkpoint(
        tmp_path / CHECKPOINT_SUBDIR, _params(0.0), epoch_to_load=1
    )

    block = loaded["params"]["mlip_network"]["ReadoutBlock_0"]
    np.testing.assert_allclose(block["linear"]["kernel"], 3.0)
    # Zero-size blocks are rebuilt from the model, not read from the checkpoint.
    assert block["skipped"].shape == (0, 3)
    assert block["skipped"].dtype == jnp.float32


def test_ema_params_are_restored(tmp_path):
    _save_checkpoint(tmp_path / CHECKPOINT_SUBDIR, _params(3.0), save_ema=True)

    loaded = load_parameters_from_checkpoint(
        tmp_path / CHECKPOINT_SUBDIR,
        _params(0.0),
        epoch_to_load=1,
        load_ema_params=True,
    )

    kernel = loaded["params"]["mlip_network"]["ReadoutBlock_0"]["linear"]["kernel"]
    np.testing.assert_allclose(kernel, 3.0)


def test_params_missing_from_checkpoint_raise(tmp_path):
    _save_checkpoint(tmp_path / CHECKPOINT_SUBDIR, _params(3.0))

    # A model with an extra readout head, as produced by adding a fine-tuning
    # head, asks for parameters the checkpoint does not have.
    initial_params = _params(0.0)
    network = initial_params["params"]["mlip_network"]
    network["ReadoutBlock_1"] = network["ReadoutBlock_0"]

    with pytest.raises(ValueError, match="ReadoutBlock_1"):
        load_parameters_from_checkpoint(
            tmp_path / CHECKPOINT_SUBDIR, initial_params, epoch_to_load=1
        )


def test_params_not_used_by_model_raise(tmp_path):
    saved_params = _params(3.0)
    network = saved_params["params"]["mlip_network"]
    network["ReadoutBlock_1"] = network["ReadoutBlock_0"]
    _save_checkpoint(tmp_path / CHECKPOINT_SUBDIR, saved_params)

    with pytest.raises(ValueError, match="ReadoutBlock_1"):
        load_parameters_from_checkpoint(
            tmp_path / CHECKPOINT_SUBDIR, _params(0.0), epoch_to_load=1
        )


def test_mismatching_param_shape_raises(tmp_path):
    _save_checkpoint(tmp_path / CHECKPOINT_SUBDIR, _params(3.0, num_features=4))

    with pytest.raises(ValueError, match="but this model expects"):
        load_parameters_from_checkpoint(
            tmp_path / CHECKPOINT_SUBDIR,
            _params(0.0, num_features=8),
            epoch_to_load=1,
        )


def test_params_from_pre_v2_checkpoint_are_renested(tmp_path):
    # Checkpoints predating the v2 model interface stored the network params
    # without the `mlip_network` level of nesting.
    old_style = {"params": _params(3.0)["params"]["mlip_network"]}
    _save_checkpoint(tmp_path / CHECKPOINT_SUBDIR, old_style)

    loaded = load_parameters_from_checkpoint(
        tmp_path / CHECKPOINT_SUBDIR, _params(0.0), epoch_to_load=1
    )

    kernel = loaded["params"]["mlip_network"]["ReadoutBlock_0"]["linear"]["kernel"]
    np.testing.assert_allclose(kernel, 3.0)


def _save_on_multiple_devices(checkpoint_dir: str) -> None:
    assert jax.device_count() == 4, jax.device_count()
    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
    params = jax.tree.map(
        lambda t: jax.device_put(t, sharding) if t.shape[:1] == (8,) else t,
        _params(3.0),
    )
    _save_checkpoint(checkpoint_dir, params)


def _load_on_one_device(checkpoint_dir: str) -> None:
    assert jax.device_count() == 1, jax.device_count()
    loaded = load_parameters_from_checkpoint(
        checkpoint_dir, _params(0.0), epoch_to_load=1
    )
    kernel = loaded["params"]["mlip_network"]["ReadoutBlock_0"]["linear"]["kernel"]
    np.testing.assert_allclose(kernel, 3.0)


def _run_in_subprocess(local_func, num_devices, checkpoint_dir) -> None:
    """Runs a local function in a subprocess.

    Used to run the save/load functions using different JAX device counts.
    """
    # Add tests root to PYTHONPATH to make local function importable.
    env = os.environ | {
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": os.pathsep.join([
            TESTS_ROOT,
            *os.environ.get("PYTHONPATH", "").split(os.pathsep),
        ]).rstrip(os.pathsep),
    }
    if num_devices > 1:
        env["XLA_FLAGS"] = (
            f"{env.get('XLA_FLAGS', '')} "
            f"--xla_force_host_platform_device_count={num_devices}"
        ).strip()
    code = (
        f"from models.test_params_loading import {local_func.__name__}; "
        f"{local_func.__name__}({str(checkpoint_dir)!r})"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_checkpoint_saved_on_more_devices_can_be_loaded(tmp_path):
    """Params trained on several devices must load on a single device."""
    checkpoint_dir = tmp_path / CHECKPOINT_SUBDIR
    _run_in_subprocess(
        _save_on_multiple_devices, num_devices=4, checkpoint_dir=checkpoint_dir
    )
    _run_in_subprocess(
        _load_on_one_device, num_devices=1, checkpoint_dir=checkpoint_dir
    )
