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

import logging
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import jax
import numpy as np
import optax
import pytest

from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.data.configs import GraphDatasetBuilderConfig
from mlip.data.graph_dataset import GraphDataset
from mlip.data.graph_dataset_builder import GraphDatasetBuilder
from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.graph import Graph
from mlip.models import ForceField
from mlip.models.loss import HuberLoss, MSELoss
from mlip.models.params_loading import load_parameters_from_checkpoint
from mlip.training.training_io_handler import LogCategory, TrainingIOHandler
from mlip.training.training_loop import TrainingLoop
from mlip.utils.multihost import create_device_mesh

DATA_DIR = Path(__file__).parent.parent / "sample_data"
SMALL_ASPIRIN_DATASET_PATH = DATA_DIR / "small_aspirin_test.xyz"
LEARNING_RATE = 1e5  # Use large LR to ensure loss decreases for QuadraticMLIP


@pytest.fixture(params=[True, False], ids=["prefetch", "graphdataset"])
def setup_datasets_for_training(request):
    """Build train/valid dataset splits from the small aspirin dataset."""
    readers = {
        "train": ExtxyzReader(
            filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(),
        ),
        "valid": ExtxyzReader(
            filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(),
            num_to_load=2,
        ),
    }

    builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=2.0,
        use_formation_energies=False,
        max_n_node=None,
        max_n_edge=None,
        batch_size=4,
        num_batch_prefetch_host=1,
        num_batch_prefetch_device=1,
        avg_num_neighbors=None,
        avg_r_min_angstrom=None,
    )
    builder = GraphDatasetBuilder(readers, builder_config)
    mesh = create_device_mesh()
    if request.param:
        datasets = builder.get_datasets(prefetch=True, mesh=mesh)
    else:
        datasets = builder.get_datasets(prefetch=False)

    train_set, valid_set = datasets.values()
    return train_set, valid_set


def test_model_training_works_correctly_with_multiple_val_sets(
    quadratic_force_field,
    setup_system,
    setup_datasets_for_training,
    tmp_path,
):
    _, graph = setup_system
    train_set, valid_set = setup_datasets_for_training

    assert len(train_set) == 2
    assert len(valid_set) == 1

    # For this MACE test, let's use two validation sets
    valid_set = {"a": valid_set, "b": valid_set}

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
        if log_category == LogCategory.EVAL_METRICS:
            assert "a_loss" in to_log or "b_loss" in to_log
            assert "loss" not in to_log

    io_handler_config = TrainingIOHandler.Config(
        checkpoint_dir=tmp_path,
        max_to_keep=5,
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
        force_field=quadratic_force_field,
        loss=loss,
        optimizer=optax.sgd(learning_rate=LEARNING_RATE),
        config=training_config,
        io_handler=io_handler,
    )

    assert training_loop.epoch_number == 0

    training_loop.run()

    assert log_container == [
        (LogCategory.EVAL_METRICS, 7, 0),
        (LogCategory.EVAL_METRICS, 7, 0),
        (LogCategory.BEST_MODEL, 2, 0),
        (LogCategory.TRAIN_METRICS, 10, 1),
        (LogCategory.SYSTEM_METRICS, 3, 1),
        (LogCategory.EVAL_METRICS, 7, 1),
        (LogCategory.EVAL_METRICS, 7, 1),
        (LogCategory.BEST_MODEL, 2, 1),
        (LogCategory.TRAIN_METRICS, 10, 2),
        (LogCategory.SYSTEM_METRICS, 3, 2),
        (LogCategory.EVAL_METRICS, 7, 2),
        (LogCategory.EVAL_METRICS, 7, 2),
        (LogCategory.BEST_MODEL, 2, 2),
    ]
    assert train_losses[0] > train_losses[1]
    assert sorted(os.listdir(tmp_path)) == ["dataset_info.json", "model"]
    assert sorted(os.listdir(tmp_path / "model")) == ["1", "2"]

    for epoch_number in [1, 2]:
        checkpoint_contents = sorted(os.listdir(tmp_path / "model" / str(epoch_number)))
        assert "training_state" in checkpoint_contents
        assert "params_ema" in checkpoint_contents

    # Now test inference with trained model
    num_nodes = graph.nodes.positions.shape[0]
    num_edges = graph.senders.shape[0]
    batched_graph = next(
        dynamically_batch(
            [graph], n_node=num_nodes + 1, n_edge=num_edges + 1, n_graph=2
        )
    )

    loaded_params = load_parameters_from_checkpoint(
        tmp_path / "model", quadratic_force_field.params, epoch_to_load=2
    )

    loaded_ff = ForceField(quadratic_force_field.predictor, loaded_params)
    result = jax.jit(loaded_ff)(batched_graph)
    assert result.forces.shape == (num_nodes + 1, 3)
    assert result.energy.shape == (2,)
    assert result.stress.shape == (2, 3, 3)

    assert training_loop.epoch_number == 2

    # Make sure test set evaluation can be run without any exception raised
    training_loop.test(valid_set["a"])


def _switch_on_formation_energies(ff: ForceField) -> ForceField:
    """Small helper to update the DS info of a force field to have atomic
    energies removed.
    """
    _ds_info = deepcopy(ff.dataset_info)
    _ds_info.atomic_energies_removed = True
    new_mlip_network = replace(ff.predictor.mlip_network, dataset_info=_ds_info)
    return ForceField(replace(ff.predictor, mlip_network=new_mlip_network), ff.params)


@pytest.mark.parametrize("use_formation_energies", [True, False])
def test_model_training_works_correctly_with_single_val_set(
    quadratic_force_field,
    setup_system,
    setup_datasets_for_training,
    tmp_path,
    use_formation_energies,
):
    _, graph = setup_system
    train_set, valid_set = setup_datasets_for_training

    _quadratic_ff = deepcopy(quadratic_force_field)
    if use_formation_energies:
        _quadratic_ff = _switch_on_formation_energies(_quadratic_ff)

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
        checkpoint_dir=tmp_path,
        max_to_keep=5,
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
        force_field=_quadratic_ff,
        loss=loss,
        optimizer=optax.sgd(learning_rate=LEARNING_RATE),
        config=training_config,
        io_handler=io_handler,
    )

    # Check that the force field config was correctly updated
    assert training_loop.force_field.config.add_atomic_energies == (
        not use_formation_energies
    )

    training_loop.run()

    # Check that the best force field's config was restored
    assert training_loop.best_model.config.add_atomic_energies
    assert log_container == [
        (LogCategory.EVAL_METRICS, 7, 0),
        (LogCategory.BEST_MODEL, 2, 0),
        (LogCategory.TRAIN_METRICS, 10, 1),
        (LogCategory.SYSTEM_METRICS, 3, 1),
        (LogCategory.EVAL_METRICS, 7, 1),
        (LogCategory.BEST_MODEL, 2, 1),
        (LogCategory.TRAIN_METRICS, 10, 2),
        (LogCategory.SYSTEM_METRICS, 3, 2),
        (LogCategory.EVAL_METRICS, 7, 2),
        (LogCategory.BEST_MODEL, 2, 2),
    ]

    assert train_losses[0] > train_losses[1]
    assert sorted(os.listdir(tmp_path)) == ["dataset_info.json", "model"]
    assert sorted(os.listdir(tmp_path / "model")) == ["1", "2"]

    for epoch_number in [1, 2]:
        checkpoint_contents = sorted(os.listdir(tmp_path / "model" / str(epoch_number)))
        assert "training_state" in checkpoint_contents
        assert "params_ema" in checkpoint_contents

    # Now test inference with trained model
    num_nodes = graph.nodes.positions.shape[0]
    num_edges = graph.senders.shape[0]
    batched_graph = next(
        dynamically_batch(
            [graph], n_node=num_nodes + 1, n_edge=num_edges + 1, n_graph=2
        )
    )

    loaded_params = load_parameters_from_checkpoint(
        tmp_path / "model", _quadratic_ff.params, epoch_to_load=2
    )

    loaded_ff = ForceField(_quadratic_ff.predictor, loaded_params)
    result = jax.jit(loaded_ff)(batched_graph)
    assert result.forces.shape == (num_nodes + 1, 3)
    assert result.energy.shape == (2,)
    assert result.stress.shape == (2, 3, 3)

    # Make sure test set evaluation can be run without any exception raised
    training_loop.test(valid_set)


def test_best_params_saved_correctly(
    quadratic_force_field,
    setup_datasets_for_training,
    tmp_path,
):
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
        checkpoint_dir=tmp_path,
        max_to_keep=5,
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
        force_field=quadratic_force_field,
        loss=loss,
        optimizer=optax.sgd(learning_rate=LEARNING_RATE),
        config=training_config,
        io_handler=io_handler,
    )

    training_loop.run()

    # Ensure that the training works
    assert train_losses[0] > train_losses[1]

    # Verify the best model params can be materialized without errors
    leaves, _ = jax.tree.flatten(training_loop.best_model.params)
    assert leaves[0] is not None


def test_graphdataset_multi_device_warning(caplog):
    """Multi-device mesh is collapsed to 1 device when a GraphDataset is present."""
    graph = Graph(
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
        homogenize=False,
    )

    mock_mesh = MagicMock()
    mock_mesh.devices.flat = [MagicMock(), MagicMock()]

    is_parallel = TrainingLoop._dataset_yields_parallel_batches(dataset)
    assert is_parallel is False

    with caplog.at_level(logging.WARNING, logger="mlip"):
        new_mesh = TrainingLoop._restrict_mesh_for_graph_dataset(
            dataset, is_parallel, mock_mesh
        )

    assert "GraphDataset only supports single-device training" in caplog.text
    assert len(new_mesh.devices.flat) == 1
