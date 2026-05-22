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

import dataclasses
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import jax
import jraph
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.data.configs import ChemicalSystemsReaderConfig, GraphDatasetBuilderConfig
from mlip.data.graph_dataset_builder import GraphDatasetBuilder
from mlip.data.helpers.data_prefetching import UnsqueezeGraphDatasetWrapper
from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.data.helpers.graph_dataset import GraphDataset
from mlip.models.loss import HuberLoss, MSELoss
from mlip.models.params_loading import load_parameters_from_checkpoint
from mlip.training.ema import exponentially_moving_average
from mlip.training.training_io_handler import LogCategory, TrainingIOHandler
from mlip.training.training_loop import TrainingLoop
from mlip.training.training_state import init_training_state
from mlip.utils.multihost import create_device_mesh

DATA_DIR = Path(__file__).parent.parent / "data"
SMALL_ASPIRIN_DATASET_PATH = DATA_DIR / "small_aspirin_test.xyz"


@pytest.fixture(params=[True, False], ids=["prefetch", "graphdataset"])
def setup_datasets_for_training(request):
    """Build train/valid dataset splits from the small aspirin dataset."""
    reader_config = ChemicalSystemsReaderConfig(
        reader_type="extxyz",
        train_dataset_paths=[str(SMALL_ASPIRIN_DATASET_PATH.resolve())],
        valid_dataset_paths=[str(SMALL_ASPIRIN_DATASET_PATH.resolve())],
        test_dataset_paths=None,
        train_num_to_load=None,
        valid_num_to_load=2,
        test_num_to_load=None,
    )
    builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=2.0,
        use_formation_energies=False,
        max_n_node=None,
        max_n_edge=None,
        batch_size=4,
        num_batch_prefetch=1,
        batch_prefetch_num_devices=1,
        avg_num_neighbors=None,
        avg_r_min_angstrom=None,
    )

    reader = ExtxyzReader(config=reader_config)
    builder = GraphDatasetBuilder(reader, builder_config)
    builder.prepare_datasets()
    mesh = create_device_mesh()
    if request.param:
        train_set, valid_set, _ = builder.get_splits(prefetch=True, mesh=mesh)
    else:
        train_set, valid_set, _ = builder.get_splits(prefetch=False)
    return train_set, valid_set


def test_model_training_works_correctly_for_mace(
    setup_system_and_mace_model,
    setup_datasets_for_training,
    tmp_path,
):
    _, graph, mace_apply_fun, mace_ff = setup_system_and_mace_model
    train_set, valid_set = setup_datasets_for_training

    assert len(train_set) == 2
    assert len(valid_set) == 1

    training_config = TrainingLoop.Config(
        num_epochs=2,
        num_gradient_accumulation_steps=1,
        random_seed=42,
        ema_decay=0.99,
        use_ema_params_for_eval=True,
        eval_num_graphs=None,
        run_eval_at_start=True,
    )

    loss = MSELoss(
        lambda x: 1.0,
        lambda x: 1.0,
        lambda x: 0,
        extended_metrics=True,
    )

    log_container = []
    train_losses = []

    def _mock_logger(log_category, to_log, epoch_num):
        log_container.append((log_category, len(to_log), epoch_num))
        if log_category == LogCategory.TRAIN_METRICS:
            train_losses.append(to_log["loss"])

    io_handler_config = TrainingIOHandler.Config(
        local_model_output_dir=tmp_path,
        max_checkpoints_to_keep=5,
        save_debiased_ema=True,
        ema_decay=0.99,
        restore_checkpoint_if_exists=False,
        epoch_to_restore=None,
        restore_optimizer_state=False,
        clear_previous_checkpoints=False,
    )
    io_handler = TrainingIOHandler(io_handler_config)
    io_handler.attach_logger(_mock_logger)

    training_loop = TrainingLoop(
        train_dataset=train_set,
        validation_dataset=valid_set,
        force_field=mace_ff,
        loss=loss,
        optimizer=optax.sgd(learning_rate=0.001),
        config=training_config,
        io_handler=io_handler,
    )

    assert training_loop.epoch_number == 0

    training_loop.run()

    assert log_container == [
        (LogCategory.EVAL_METRICS, 7, 0),
        (LogCategory.BEST_MODEL, 2, 0),
        (LogCategory.TRAIN_METRICS, 6, 1),
        (LogCategory.SYSTEM_METRICS, 3, 1),
        (LogCategory.EVAL_METRICS, 7, 1),
        (LogCategory.BEST_MODEL, 2, 1),
        (LogCategory.TRAIN_METRICS, 6, 2),
        (LogCategory.SYSTEM_METRICS, 3, 2),
        (LogCategory.EVAL_METRICS, 7, 2),
        (LogCategory.BEST_MODEL, 2, 2),
    ]
    assert train_losses[0] > train_losses[1]
    assert sorted(os.listdir(tmp_path)) == ["dataset_info.json", "model"]
    assert sorted(os.listdir(tmp_path / "model")) == ["1", "2"]

    for epoch_number in [1, 2]:
        checkpoint_contents = sorted(os.listdir(tmp_path / "model" / str(epoch_number)))
        # Orbax 0.11+ adds _CHECKPOINT_METADATA file
        expected_files = ["_CHECKPOINT_METADATA", "params_ema", "training_state"]
        assert checkpoint_contents == expected_files or checkpoint_contents == [
            "params_ema",
            "training_state",
        ]

    # Now test inference with trained model
    num_nodes = graph.nodes.positions.shape[0]
    num_edges = graph.senders.shape[0]
    batched_graph = next(
        dynamically_batch(
            [graph], n_node=num_nodes + 1, n_edge=num_edges + 1, n_graph=2
        )
    )

    loaded_params = load_parameters_from_checkpoint(
        tmp_path / "model", mace_ff.params, epoch_to_load=2
    )
    result = mace_apply_fun(loaded_params, batched_graph)
    assert result.forces.shape == (num_nodes + 1, 3)
    assert result.energy is not None
    assert result.stress is None

    assert training_loop.epoch_number == 2

    # Make sure test set evaluation can be run without any exception raised
    training_loop.test(valid_set)


def test_model_training_works_correctly_for_visnet(
    setup_system_and_visnet_model,
    setup_datasets_for_training,
    tmp_path,
):
    _, graph, visnet_apply_fun, visnet_ff = setup_system_and_visnet_model
    train_set, valid_set = setup_datasets_for_training

    assert len(train_set) == 2
    assert len(valid_set) == 1

    training_config = TrainingLoop.Config(
        num_epochs=2,
        num_gradient_accumulation_steps=1,
        random_seed=42,
        ema_decay=0.99,
        use_ema_params_for_eval=True,
        eval_num_graphs=None,
        run_eval_at_start=True,
    )

    loss = HuberLoss(
        lambda x: 1.0,
        lambda x: 1.0,
        lambda x: 0,
        extended_metrics=True,
    )

    log_container = []
    train_losses = []

    def _mock_logger(log_category, to_log, epoch_num):
        log_container.append((log_category, len(to_log), epoch_num))
        if log_category == LogCategory.TRAIN_METRICS:
            train_losses.append(to_log["loss"])

    io_handler_config = TrainingIOHandler.Config(
        local_model_output_dir=tmp_path,
        max_checkpoints_to_keep=5,
        save_debiased_ema=True,
        ema_decay=0.99,
        restore_checkpoint_if_exists=False,
        epoch_to_restore=None,
        restore_optimizer_state=False,
        clear_previous_checkpoints=False,
    )
    io_handler = TrainingIOHandler(io_handler_config)
    io_handler.attach_logger(_mock_logger)

    training_loop = TrainingLoop(
        train_dataset=train_set,
        validation_dataset=valid_set,
        force_field=visnet_ff,
        loss=loss,
        optimizer=optax.sgd(learning_rate=0.0001),
        config=training_config,
        io_handler=io_handler,
    )
    training_loop.run()

    assert log_container == [
        (LogCategory.EVAL_METRICS, 7, 0),
        (LogCategory.BEST_MODEL, 2, 0),
        (LogCategory.TRAIN_METRICS, 6, 1),
        (LogCategory.SYSTEM_METRICS, 3, 1),
        (LogCategory.EVAL_METRICS, 7, 1),
        (LogCategory.BEST_MODEL, 2, 1),
        (LogCategory.TRAIN_METRICS, 6, 2),
        (LogCategory.SYSTEM_METRICS, 3, 2),
        (LogCategory.EVAL_METRICS, 7, 2),
        (LogCategory.BEST_MODEL, 2, 2),
    ]

    assert train_losses[0] > train_losses[1]
    assert sorted(os.listdir(tmp_path)) == ["dataset_info.json", "model"]
    assert sorted(os.listdir(tmp_path / "model")) == ["1", "2"]

    for epoch_number in [1, 2]:
        checkpoint_contents = sorted(os.listdir(tmp_path / "model" / str(epoch_number)))
        # Orbax 0.11+ adds _CHECKPOINT_METADATA file
        expected_files = ["_CHECKPOINT_METADATA", "params_ema", "training_state"]
        assert checkpoint_contents == expected_files or checkpoint_contents == [
            "params_ema",
            "training_state",
        ]

    # Now test inference with trained model
    num_nodes = graph.nodes.positions.shape[0]
    num_edges = graph.senders.shape[0]
    batched_graph = next(
        dynamically_batch(
            [graph], n_node=num_nodes + 1, n_edge=num_edges + 1, n_graph=2
        )
    )

    loaded_params = load_parameters_from_checkpoint(
        tmp_path / "model", visnet_ff.params, epoch_to_load=2
    )
    result = visnet_apply_fun(loaded_params, batched_graph)

    assert result.forces.shape == (num_nodes + 1, 3)
    assert result.energy is not None
    assert result.stress is None

    # Make sure test set evaluation can be run without any exception raised
    training_loop.test(valid_set)


def test_best_params_saved_correctly(
    setup_system_and_mace_model,
    setup_datasets_for_training,
    tmp_path,
):
    _, graph, mace_apply_fun, mace_ff = setup_system_and_mace_model
    train_set, valid_set = setup_datasets_for_training

    training_config = TrainingLoop.Config(
        num_epochs=2,
        num_gradient_accumulation_steps=1,
        random_seed=42,
        ema_decay=0.99,
        use_ema_params_for_eval=True,
        eval_num_graphs=None,
        run_eval_at_start=True,
    )

    loss = MSELoss(lambda x: 1.0, lambda x: 1.0, lambda x: 0)

    log_container = []
    train_losses = []

    def _mock_logger(log_category, to_log, epoch_num):
        log_container.append((log_category, len(to_log), epoch_num))
        if log_category == LogCategory.TRAIN_METRICS:
            train_losses.append(to_log["loss"])

    io_handler_config = TrainingIOHandler.Config(
        local_model_output_dir=tmp_path,
        max_checkpoints_to_keep=5,
        save_debiased_ema=True,
        ema_decay=0.99,
        restore_checkpoint_if_exists=False,
        epoch_to_restore=None,
        restore_optimizer_state=False,
        clear_previous_checkpoints=False,
    )
    io_handler = TrainingIOHandler(io_handler_config)
    io_handler.attach_logger(_mock_logger)

    training_loop = TrainingLoop(
        train_dataset=train_set,
        validation_dataset=valid_set,
        force_field=mace_ff,
        loss=loss,
        optimizer=optax.sgd(learning_rate=0.01),
        config=training_config,
        io_handler=io_handler,
    )

    training_loop.run()

    # Ensure that the training worsens
    assert train_losses[0] < train_losses[1]

    # Verify the best model params can be materialized without errors
    leaves, _ = jax.tree.flatten(training_loop.best_model.params)
    assert leaves[0] is not None


def test_graphdataset_multi_device_warning(caplog):
    """Verify that _maybe_wrap_dataset logs a warning for multi-device meshes."""
    graph = jraph.GraphsTuple(
        nodes=np.zeros((2, 1), dtype=np.float32),
        edges=np.zeros((2, 1), dtype=np.float32),
        senders=np.array([0, 1], dtype=np.int32),
        receivers=np.array([1, 0], dtype=np.int32),
        n_node=np.array([2], dtype=np.int32),
        n_edge=np.array([2], dtype=np.int32),
        globals=np.array([0.0], dtype=np.float32),
    )
    dataset = GraphDataset(
        graphs=[graph],
        batch_size=1,
        max_n_node=3,
        max_n_edge=3,
    )

    mock_mesh = MagicMock()
    mock_mesh.devices.flat = [MagicMock(), MagicMock()]

    with caplog.at_level(logging.WARNING, logger="mlip"):
        wrapped, new_mesh = TrainingLoop._maybe_wrap_dataset(dataset, mock_mesh)

    assert isinstance(wrapped, UnsqueezeGraphDatasetWrapper)
    assert "GraphDataset only supports single-device training" in caplog.text
    assert len(new_mesh.devices.flat) == 1


def test_restore_legacy_checkpoint_with_key_field(
    setup_system_and_mace_model, tmp_path
):
    """Restoring a checkpoint saved with the legacy ``key`` field should succeed.

    Old versions of ``TrainingState`` contained a ``key: PRNGKey`` field that
    has since been removed.  ``partial_restore=True`` in the IO handler lets
    Orbax silently skip the extra on-disk field.
    """
    _, _, _, mace_ff = setup_system_and_mace_model

    optimizer = optax.sgd(learning_rate=0.001)
    ema_fun = exponentially_moving_average(0.99)
    training_state = init_training_state(mace_ff.params, optimizer, ema_fun)

    # Save a checkpoint as a dict that includes the legacy ``key`` field.
    legacy_state = {
        f.name: getattr(training_state, f.name)
        for f in dataclasses.fields(training_state)
    }
    legacy_state["key"] = jax.random.PRNGKey(0)

    model_dir = tmp_path / "model"
    ckpt_manager = ocp.CheckpointManager(
        model_dir,
        item_names=("training_state",),
    )
    ckpt_manager.save(
        1,
        args=ocp.args.Composite(
            training_state=ocp.args.PyTreeSave(legacy_state),
        ),
    )
    ckpt_manager.wait_until_finished()

    # Now restore via the IO handler (which uses partial_restore=True).
    io_handler_config = TrainingIOHandler.Config(
        local_model_output_dir=tmp_path,
        restore_checkpoint_if_exists=True,
        epoch_to_restore=1,
        restore_optimizer_state=True,
    )
    io_handler = TrainingIOHandler(io_handler_config)

    restored = io_handler.restore_training_state(training_state)

    # The restored state must not contain the legacy ``key`` field …
    assert not hasattr(restored, "key")
    # … and the params must match the saved values.
    orig_leaves = jax.tree_util.tree_leaves(training_state.params)
    rest_leaves = jax.tree_util.tree_leaves(restored.params)
    for orig, rest in zip(orig_leaves, rest_leaves):
        np.testing.assert_array_equal(orig, rest)
