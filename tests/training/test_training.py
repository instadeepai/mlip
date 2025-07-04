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
from pathlib import Path

import jax.tree_util
import optax
import pytest

from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.data.configs import ChemicalSystemsReaderConfig, GraphDatasetBuilderConfig
from mlip.data.graph_dataset_builder import GraphDatasetBuilder
from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.models.loss import HuberLoss, MSELoss
from mlip.models.params_loading import load_parameters_from_checkpoint
from mlip.training.training_io_handler import LogCategory, TrainingIOHandler
from mlip.training.training_loop import TrainingLoop

DATA_DIR = Path(__file__).parent.parent / "data"
SMALL_ASPIRIN_DATASET_PATH = DATA_DIR / "small_aspirin_test.xyz"


@pytest.fixture
def setup_datasets_for_training():
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
    train_set, valid_set, _ = builder.get_splits()
    return train_set, valid_set


def test_model_training_works_correctly_for_mace(
    setup_system_and_mace_model,
    setup_datasets_for_training,
    tmp_path,
):
    _, graph, mace_apply_fun, mace_ff = setup_system_and_mace_model
    train_set, valid_set = setup_datasets_for_training

    assert len(train_set) == 2
    assert len(train_set.graphs) == 7
    assert len(valid_set) == 1
    assert len(valid_set.graphs) == 2

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
        should_parallelize=False,
    )

    assert training_loop.epoch_number == 0

    training_loop.run()

    assert log_container == [
        (LogCategory.EVAL_METRICS, 7, 0),
        (LogCategory.BEST_MODEL, 2, 0),
        (LogCategory.TRAIN_METRICS, 9, 1),
        (LogCategory.EVAL_METRICS, 7, 1),
        (LogCategory.BEST_MODEL, 2, 1),
        (LogCategory.TRAIN_METRICS, 9, 2),
        (LogCategory.EVAL_METRICS, 7, 2),
        (LogCategory.BEST_MODEL, 2, 2),
    ]
    assert train_losses[0] > train_losses[1]
    assert sorted(os.listdir(tmp_path)) == ["dataset_info.json", "model"]
    assert sorted(os.listdir(tmp_path / "model")) == ["1", "2"]

    for epoch_number in [1, 2]:
        assert sorted(os.listdir(tmp_path / "model" / str(epoch_number))) == [
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
    assert len(train_set.graphs) == 7
    assert len(valid_set) == 1
    assert len(valid_set.graphs) == 2

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
        should_parallelize=False,
    )
    training_loop.run()

    assert log_container == [
        (LogCategory.EVAL_METRICS, 7, 0),
        (LogCategory.BEST_MODEL, 2, 0),
        (LogCategory.TRAIN_METRICS, 9, 1),
        (LogCategory.EVAL_METRICS, 7, 1),
        (LogCategory.BEST_MODEL, 2, 1),
        (LogCategory.TRAIN_METRICS, 9, 2),
        (LogCategory.EVAL_METRICS, 7, 2),
        (LogCategory.BEST_MODEL, 2, 2),
    ]

    assert train_losses[0] > train_losses[1]
    assert sorted(os.listdir(tmp_path)) == ["dataset_info.json", "model"]
    assert sorted(os.listdir(tmp_path / "model")) == ["1", "2"]

    for epoch_number in [1, 2]:
        assert sorted(os.listdir(tmp_path / "model" / str(epoch_number))) == [
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
        should_parallelize=False,
    )

    training_loop.run()

    # Ensure that the training worsens
    assert train_losses[0] < train_losses[1]

    # Materialise a leaf of the pytree to test for error issues
    leaves, _ = jax.tree.flatten(training_loop.best_model.params)
    leaves[0].block_until_ready()
