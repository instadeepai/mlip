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

"""Tests for mid-epoch (inside-epoch) checkpoint save/restore.

The GraphDataset supports resuming iteration from a mid-epoch position via
its `num_graphs_processed` field in `GraphDatasetState`. These tests
verify that:

1. Skipping graphs via `num_graphs_processed` produces the same batches
   as the tail of a full iteration.
2. Saving and restoring training state (including dataset state) mid-epoch
   yields EXACTLY the same per-batch losses as a continuous run.
"""

import functools
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mlip.data import DatasetInfo
from mlip.data.chemical_systems_readers.extxyz_reader import ExtxyzReader
from mlip.data.configs import GraphDatasetBuilderConfig
from mlip.data.graph_dataset import GraphDataset, GraphDatasetState
from mlip.data.graph_dataset_builder import SingleGraphDatasetBuilder
from mlip.data.helpers.combined_graph_dataset import CombinedGraphDataset
from mlip.data.helpers.data_prefetching import PrefetchIterator
from mlip.graph import Graph
from mlip.models import ForceField
from mlip.models.loss import MSELoss
from mlip.training.ema import exponentially_moving_average
from mlip.training.training_io_handler import TrainingIOHandler
from mlip.training.training_loop import TrainingLoop
from mlip.training.training_state import init_training_state
from mlip.training.training_step import make_train_step
from mlip.typing.properties import Properties

DATA_DIR = Path(__file__).parent.parent / "sample_data"
SMALL_ASPIRIN_DATASET_PATH = DATA_DIR / "small_aspirin_test.xyz"


def _count_real_graphs(batch: Graph) -> int:
    """Count the number of non-padding graphs in a batched graph."""
    if batch.n_node.ndim > 1:
        _batch = flax.jax_utils.unreplicate(batch)
    else:
        _batch = batch
    return int(_batch.graph_mask().sum())


def _trees_all_close(tree_a, tree_b):
    """Assert that two pytrees have identical leaves (exact match)."""
    leaves_a = jax.tree_util.tree_leaves(tree_a)
    leaves_b = jax.tree_util.tree_leaves(tree_b)
    assert len(leaves_a) == len(leaves_b)
    for a, b in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(np.array(a), np.array(b))


def _build_dataset(prefetch: bool = False) -> tuple[GraphDataset, DatasetInfo]:
    """Build a fresh GraphDataset and its DatasetInfo from the aspirin sample."""
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
    reader = ExtxyzReader(
        filepaths=SMALL_ASPIRIN_DATASET_PATH.resolve(),
    )

    builder = SingleGraphDatasetBuilder(reader, builder_config, dataset_info=True)

    train_set = builder.get_dataset(prefetch=prefetch)

    # Disable inter-epoch shuffling so graph order is deterministic across
    # __iter__ calls. The initial shuffle from __init__ is still applied, so
    # graphs are in a fixed-but-shuffled order.
    train_set.shuffle_between_epochs = False
    return train_set, builder.dataset_info


@pytest.fixture(params=[False, True], ids=["no_prefetch", "prefetch"], scope="session")
def train_dataset(request) -> tuple[GraphDataset, DatasetInfo]:
    """Create a GraphDataset from the small aspirin test data."""
    return _build_dataset(prefetch=request.param)


@pytest.fixture(scope="session")
def force_field(quadratic_mlip):
    ff = ForceField.from_mlip_network(
        quadratic_mlip,
        seed=42,
        required_properties=Properties(stress=False),
    )
    return ForceField(ff.predictor, ff.params)


class TestMidEpochCombinedDatasetResume:
    """Test that CombinedGraphDataset correctly resumes from a mid-epoch position
    via num_graphs_processed."""

    @pytest.mark.parametrize("interleaving_method", ["random", "regular"])
    def test_resume_combined_dataset_mid_epoch(self, interleaving_method):
        # CombinedGraphDataset creates one sub-iterator per dataset and they
        # share mutable state if the same GraphDataset object is reused, so
        # we need distinct instances here.
        dataset_a, _ = _build_dataset()
        dataset_b, _ = _build_dataset()

        def _create_combined_dataset():
            return CombinedGraphDataset.init(
                graph_datasets=[dataset_a, dataset_b],
                interleaving_method=interleaving_method,
            )

        def _reset_sub_dataset_cursors():
            dataset_a.state = dataset_a.state.replace(num_graphs_processed=jnp.int32(0))
            dataset_b.state = dataset_b.state.replace(num_graphs_processed=jnp.int32(0))

        full_ds = _create_combined_dataset()
        full_sequence = [jnp.int32(batch.n_node[0]) for batch in full_ds]

        _reset_sub_dataset_cursors()

        interrupted_ds = _create_combined_dataset()
        iterator = iter(interrupted_ds)

        batches_1 = []
        for _ in range(3):
            batch = next(iterator)
            batches_1.append(jnp.int32(batch.n_node[0]))

        # capture state at the point of interruption
        saved_state = interrupted_ds.state
        if not isinstance(interrupted_ds, PrefetchIterator):
            assert interrupted_ds.state.num_graphs_processed == 3
        else:
            saved_state = saved_state.replace(num_graphs_processed=len(batches_1))

        _reset_sub_dataset_cursors()

        # simulate run resumption
        resumed_ds = _create_combined_dataset()
        resumed_ds.state = saved_state

        batches_2 = [jnp.int32(batch.n_node[0]) for batch in resumed_ds]
        assert len(batches_1) + len(batches_2) == len(full_sequence)
        assert resumed_ds.state.num_graphs_processed == 0


class TestMidEpochDatasetResume:
    """Test that GraphDataset correctly resumes from a mid-epoch position
    via num_graphs_processed."""

    def test_skip_graphs_produces_same_remaining_batches(self, train_dataset):
        """Setting num_graphs_processed to the cumulative graph count after
        K batches should produce the same remaining batches as the tail of
        a full iteration."""
        dataset, _ = train_dataset

        # Full iteration: collect all batches and cumulative graph counts
        all_batches = list(dataset)
        assert len(all_batches) >= 2, "Need at least 2 batches for mid-epoch test"

        cumulative_graphs = [0]
        for batch in all_batches:
            cumulative_graphs.append(cumulative_graphs[-1] + _count_real_graphs(batch))

        # Save state after full iteration (rng unchanged since no inter-epoch
        # shuffle; num_graphs_processed was reset to 0 by __iter__)
        state_after_full = dataset.state

        # For each possible mid-epoch point, verify skip produces correct tail
        for k in range(1, len(all_batches)):
            dataset.state = state_after_full.replace(
                num_graphs_processed=jnp.int32(cumulative_graphs[k])
            )

            resumed_batches = list(dataset)
            expected_tail = all_batches[k:]

            assert len(resumed_batches) == len(expected_tail), (
                f"After skipping {cumulative_graphs[k]} graphs (k={k}), "
                f"expected {len(expected_tail)} batches, got {len(resumed_batches)}"
            )

            for i, (resumed, expected) in enumerate(
                zip(resumed_batches, expected_tail)
            ):
                _trees_all_close(resumed, expected)

    def test_skipping_zero_graphs_reproduces_full_iteration(self, train_dataset):
        """With num_graphs_processed=0, iteration should produce the same
        batches as a fresh iteration (no-op skip)."""
        dataset, _ = train_dataset

        first_batches = list(dataset)
        # num_graphs_processed is already 0 after iteration
        second_batches = list(dataset)

        assert len(first_batches) == len(second_batches)
        for a, b in zip(first_batches, second_batches):
            _trees_all_close(a, b)

    def test_skipping_all_graphs_produces_no_batches(self, train_dataset):
        """Setting num_graphs_processed to total graphs should yield nothing."""
        dataset, _ = train_dataset

        total_graphs = sum(_count_real_graphs(b) for b in dataset)
        dataset.state = dataset.state.replace(
            num_graphs_processed=jnp.int32(total_graphs)
        )

        remaining = list(dataset)
        assert len(remaining) == 0

    def test_state_fields_after_mid_epoch_resume(self, train_dataset):
        """After a mid-epoch resume iteration, num_graphs_processed should be
        reset to 0 (ready for next epoch) and rng should be unchanged."""
        dataset, _ = train_dataset

        all_batches = list(dataset)
        state_before = dataset.state
        cumulative = _count_real_graphs(all_batches[0])

        # Set mid-epoch state and iterate
        dataset.state = state_before.replace(num_graphs_processed=jnp.int32(cumulative))
        list(dataset)  # consume the iterator

        # After iteration, counter should be reset
        assert int(dataset.state.num_graphs_processed) == 0
        # rng should be unchanged (no inter-epoch shuffle)
        np.testing.assert_array_equal(
            np.array(dataset.state.rng), np.array(state_before.rng)
        )


class TestMidEpochCheckpointResume:
    """Test that saving training state mid-epoch and restoring it produces
    EXACTLY the same per-batch losses as a continuous run."""

    def test_exact_loss_match_after_mid_epoch_restore(
        self, force_field, train_dataset, tmp_path
    ):
        """Run a full epoch collecting per-batch losses, then simulate a
        mid-epoch checkpoint+restore and verify the remaining losses match
        exactly."""
        dataset, _ = train_dataset
        raw_batches = list(dataset)
        assert len(raw_batches) >= 2

        # Compute cumulative graph counts from raw (unwrapped) batches
        cumulative_graphs = [0]
        for batch in raw_batches:
            cumulative_graphs.append(cumulative_graphs[-1] + _count_real_graphs(batch))

        all_batches = list(dataset)

        # Setup training components
        loss = MSELoss(
            lambda x: 1.0, lambda x: 1.0, lambda x: 0, extended_metrics=False
        )
        optimizer = optax.sgd(learning_rate=0.0001)
        ema_fun = exponentially_moving_average(decay=0.99)
        training_state = init_training_state(force_field.params, optimizer, ema_fun)

        avg_n_graphs = float(np.mean([_count_real_graphs(b) for b in raw_batches]))
        training_step = make_train_step(
            force_field.predictor,
            functools.partial(loss, eval_metrics=False),
            optimizer,
            ema_fun,
            avg_n_graphs_per_batch=avg_n_graphs,
            num_gradient_accumulation_steps=1,
            should_parallelize=TrainingLoop._dataset_yields_parallel_batches(dataset),
        )

        # --- Full epoch: collect per-batch losses ---
        full_losses = []
        state = training_state
        for batch in all_batches:
            state, metrics = training_step(state, batch, 1)
            full_losses.append(float(jax.device_get(metrics["loss"])))

        # --- Checkpoint after first K batches ---
        k = 1
        state_at_k = training_state
        for i in range(k):
            state_at_k, _ = training_step(state_at_k, all_batches[i], 1)

        # Build dataset state to checkpoint alongside the training state
        ckpt_dataset_state = GraphDatasetState(
            rng=dataset.state.rng,
            num_graphs_processed=jnp.int32(cumulative_graphs[k]),
        )

        # Save intra-epoch checkpoint
        step_number = int(state_at_k.num_steps)
        save_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path,
            save_debiased_ema=False,
            use_intra_epoch_checkpointing=True,
            intra_epoch_save_every_n_steps=1,
        )
        save_handler = TrainingIOHandler(save_config)
        save_handler.save_intra_epoch_checkpoint(
            state_at_k,
            step_number=step_number,
            epoch_number=1,
            dataset_state=ckpt_dataset_state,
            accumulated_metrics=[],
        )
        save_handler.wait_until_finished()

        # --- Restore checkpoint ---
        fresh_state = init_training_state(force_field.params, optimizer, ema_fun)
        restore_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path,
            save_debiased_ema=False,
            use_intra_epoch_checkpointing=True,
            intra_epoch_save_every_n_steps=1,
            restore_checkpoint_if_exists=True,
            restore_optimizer_state=True,
        )
        restore_handler = TrainingIOHandler(restore_config)
        restored_state, restored_dataset_state, _, _ = (
            restore_handler.restore_checkpoint(fresh_state, dataset_state=dataset.state)
        )

        # Restore dataset state so iteration skips already-processed graphs
        dataset.state = restored_dataset_state

        # --- Run remaining batches from restored state ---
        resumed_batches = list(dataset)
        resumed_losses = []
        for batch in resumed_batches:
            restored_state, metrics = training_step(restored_state, batch, 1)
            resumed_losses.append(float(jax.device_get(metrics["loss"])))

        # --- Verify EXACT match ---
        expected_losses = full_losses[k:]
        assert len(resumed_losses) == len(expected_losses), (
            f"Expected {len(expected_losses)} batches after resume, "
            f"got {len(resumed_losses)}"
        )
        for i, (resumed, expected) in enumerate(zip(resumed_losses, expected_losses)):
            assert resumed == expected, (
                f"Loss mismatch at resumed batch {i}: "
                f"got {resumed}, expected {expected}"
            )

    def test_exact_loss_match_restore_without_optimizer(
        self, force_field, train_dataset, tmp_path
    ):
        """Same as above but restoring WITHOUT optimizer state. Losses from
        the remaining batches should still match exactly because the model
        params are identical (optimizer state only affects future updates,
        but params fed to the forward pass are the same)."""
        dataset, _ = train_dataset
        raw_batches = list(dataset)

        cumulative_graphs = [0]
        for batch in raw_batches:
            cumulative_graphs.append(cumulative_graphs[-1] + _count_real_graphs(batch))

        all_batches = list(dataset)

        loss = MSELoss(
            lambda x: 1.0, lambda x: 1.0, lambda x: 0, extended_metrics=False
        )
        optimizer = optax.sgd(learning_rate=0.0001)
        ema_fun = exponentially_moving_average(decay=0.99)
        training_state = init_training_state(force_field.params, optimizer, ema_fun)

        avg_n_graphs = float(np.mean([_count_real_graphs(b) for b in raw_batches]))
        training_step = make_train_step(
            force_field.predictor,
            functools.partial(loss, eval_metrics=False),
            optimizer,
            ema_fun,
            avg_n_graphs_per_batch=avg_n_graphs,
            num_gradient_accumulation_steps=1,
            should_parallelize=TrainingLoop._dataset_yields_parallel_batches(dataset),
        )

        # Full epoch
        full_losses = []
        state = training_state
        for batch in all_batches:
            state, metrics = training_step(state, batch, 1)
            full_losses.append(float(jax.device_get(metrics["loss"])))

        # Checkpoint after batch 1
        k = 1
        state_at_k = training_state
        for i in range(k):
            state_at_k, _ = training_step(state_at_k, all_batches[i], 1)

        ckpt_dataset_state = GraphDatasetState(
            rng=dataset.state.rng,
            num_graphs_processed=jnp.int32(cumulative_graphs[k]),
        )

        step_number = int(state_at_k.num_steps)
        save_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path,
            save_debiased_ema=False,
            use_intra_epoch_checkpointing=True,
            intra_epoch_save_every_n_steps=1,
        )
        save_handler = TrainingIOHandler(save_config)
        save_handler.save_intra_epoch_checkpoint(
            state_at_k,
            step_number=step_number,
            epoch_number=1,
            dataset_state=ckpt_dataset_state,
            accumulated_metrics=[],
        )
        save_handler.wait_until_finished()

        # Restore WITHOUT optimizer state
        fresh_state = init_training_state(force_field.params, optimizer, ema_fun)
        restore_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path,
            save_debiased_ema=False,
            use_intra_epoch_checkpointing=True,
            intra_epoch_save_every_n_steps=1,
            restore_checkpoint_if_exists=True,
            restore_optimizer_state=False,
        )
        restore_handler = TrainingIOHandler(restore_config)
        restored_state, restored_dataset_state, _, _ = (
            restore_handler.restore_checkpoint(fresh_state, dataset_state=dataset.state)
        )

        dataset.state = restored_dataset_state

        # The first forward pass after restore should produce the same loss
        # because model params are identical (optimizer state differs but
        # doesn't affect the forward pass).
        resumed_batches = list(dataset)
        restored_state_step, metrics = training_step(
            restored_state, resumed_batches[0], 1
        )
        first_resumed_loss = float(jax.device_get(metrics["loss"]))

        assert first_resumed_loss == full_losses[k], (
            f"First loss after restore mismatch: "
            f"got {first_resumed_loss}, expected {full_losses[k]}"
        )


class TestPostEpochCheckpointResume:
    """Save at end of epoch 1, restore, run epoch 2 — losses must match an
    uninterrupted 2-epoch baseline."""

    def test_exact_loss_match_across_epoch_boundary(self, force_field, tmp_path):
        # The shared fixture sets shuffle=False (no inter-epoch rng dance).
        # This test specifically exercises the deferred post-shuffle rng,
        # so both shuffle and shuffle_between_epochs must be ON.
        def _new_dataset():
            ds, _ = _build_dataset()
            ds.shuffle = True
            ds.shuffle_between_epochs = True
            return ds

        optimizer = optax.sgd(learning_rate=0.0001)
        ema_fun = exponentially_moving_average(decay=0.99)

        probe = _new_dataset()
        avg_n_graphs = float(np.mean([_count_real_graphs(b) for b in probe]))
        training_step = make_train_step(
            force_field.predictor,
            functools.partial(
                MSELoss(
                    lambda x: 1.0,
                    lambda x: 1.0,
                    lambda x: 0,
                    extended_metrics=False,
                ),
                eval_metrics=False,
            ),
            optimizer,
            ema_fun,
            avg_n_graphs_per_batch=avg_n_graphs,
            num_gradient_accumulation_steps=1,
            should_parallelize=TrainingLoop._dataset_yields_parallel_batches(probe),
        )

        def _run_epoch(state, ds):
            losses = []
            for batch in ds:
                state, m = training_step(state, batch, 1)
                losses.append(float(jax.device_get(m["loss"])))
            return state, losses

        # Baseline: 2 uninterrupted epochs; keep the epoch-2 losses.
        baseline_ds = _new_dataset()
        s, _ = _run_epoch(
            init_training_state(force_field.params, optimizer, ema_fun), baseline_ds
        )
        _, baseline_e2 = _run_epoch(s, baseline_ds)

        # Resumed: train epoch 1, save the post-epoch checkpoint.
        ckpt_ds = _new_dataset()
        s, _ = _run_epoch(
            init_training_state(force_field.params, optimizer, ema_fun), ckpt_ds
        )
        TrainingIOHandler(
            TrainingIOHandler.Config(checkpoint_dir=tmp_path, save_debiased_ema=False)
        ).save_checkpoint(s, epoch_number=1, dataset_state=ckpt_ds.state)

        # Simulate a fresh process: rebuild dataset and state, then restore.
        resumed_ds = _new_dataset()
        restored_state, restored_dataset_state, last_epoch, _ = TrainingIOHandler(
            TrainingIOHandler.Config(
                checkpoint_dir=tmp_path,
                save_debiased_ema=False,
                restore_checkpoint_if_exists=True,
                restore_optimizer_state=True,
            )
        ).restore_checkpoint(
            init_training_state(force_field.params, optimizer, ema_fun),
            dataset_state=resumed_ds.state,
        )
        assert last_epoch == 1
        resumed_ds.state = restored_dataset_state

        _, resumed_e2 = _run_epoch(restored_state, resumed_ds)
        assert resumed_e2 == baseline_e2, (resumed_e2, baseline_e2)
