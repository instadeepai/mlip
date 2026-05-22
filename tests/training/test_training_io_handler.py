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
from concurrent.futures import Future
from unittest.mock import MagicMock

import jax.numpy as jnp
import optax
import pytest

from mlip.data.dataset_info import DatasetInfo
from mlip.training.checkpointer import CheckpointMetadata, RestoreResult
from mlip.training.ema import exponentially_moving_average
from mlip.training.training_io_handler import (
    CheckpointRestorationError,
    TrainingIOHandler,
)
from mlip.training.training_state import TrainingState, init_training_state


@pytest.fixture()
def make_training_state():
    """Factory fixture to create a minimal training state."""

    def _factory(num_steps: int = 0) -> TrainingState:
        params = {"w": jnp.ones((2, 2))}
        optimizer = optax.sgd(learning_rate=0.01)
        ema_fun = exponentially_moving_average(decay=0.99)
        state = init_training_state(params, optimizer, ema_fun)
        if num_steps > 0:
            state = dataclasses.replace(state, num_steps=jnp.array(num_steps))
        return state

    return _factory


class TestRestoreNumSteps:
    """Test that num_steps is correctly restored when restore_optimizer_state=False."""

    def test_restore_without_optimizer_preserves_num_steps(
        self, tmp_path, make_training_state
    ):
        """When restoring without optimizer state, num_steps should be restored
        from the checkpoint, not left at 0."""

        # Create and save a training state with num_steps=42
        original_state = make_training_state(num_steps=42)
        io_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path, save_debiased_ema=False
        )
        io_handler = TrainingIOHandler(io_config)
        io_handler.save_checkpoint(original_state, epoch_number=1)
        io_handler.wait_until_finished()

        # Now restore with restore_optimizer_state=False
        fresh_state = make_training_state(num_steps=0)
        assert int(fresh_state.num_steps) == 0

        restore_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path,
            save_debiased_ema=False,
            restore_checkpoint_if_exists=True,
            restore_optimizer_state=False,
        )
        restore_handler = TrainingIOHandler(restore_config)
        restored_state, _, _, _ = restore_handler.restore_checkpoint(fresh_state)

        assert int(restored_state.num_steps) == 42

    def test_restore_with_optimizer_preserves_num_steps(
        self, tmp_path, make_training_state
    ):
        """When restoring with optimizer state, num_steps should also be correct."""
        original_state = make_training_state(num_steps=100)
        io_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path, save_debiased_ema=False
        )
        io_handler = TrainingIOHandler(io_config)
        io_handler.save_checkpoint(original_state, epoch_number=1)
        io_handler.wait_until_finished()

        fresh_state = make_training_state(num_steps=0)
        restore_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path,
            save_debiased_ema=False,
            restore_checkpoint_if_exists=True,
            restore_optimizer_state=True,
        )
        restore_handler = TrainingIOHandler(restore_config)
        restored_state, _, _, _ = restore_handler.restore_checkpoint(fresh_state)

        assert int(restored_state.num_steps) == 100


class TestUploadFn:
    """Tests for upload_fn integration in TrainingIOHandler."""

    def test_no_upload_when_data_upload_fun_is_none(
        self, tmp_path, make_training_state
    ):
        """Default behavior: no upload_fn means no upload calls."""
        state = make_training_state()
        io_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path, save_debiased_ema=False
        )
        io_handler = TrainingIOHandler(io_config)
        io_handler.save_checkpoint(state, epoch_number=1)
        io_handler.wait_until_finished()
        # No exception means success — there's nothing to assert on
        # because _data_upload_fun is None.

    def test_data_upload_fun_called_after_save_checkpoint(
        self, tmp_path, make_training_state
    ):
        """upload_fn is called with the checkpoint_dir after save_checkpoint."""
        upload_fn = MagicMock(return_value=None)
        state = make_training_state()
        io_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path, save_debiased_ema=False
        )
        io_handler = TrainingIOHandler(io_config, data_upload_fun=upload_fn)
        io_handler.save_checkpoint(state, epoch_number=1)
        io_handler.wait_until_finished()

        upload_fn.assert_called_once_with(tmp_path)

    def test_data_upload_fun_called_after_save_intra_epoch(
        self, tmp_path, make_training_state
    ):
        """upload_fn is called after save_intra_epoch_checkpoint."""
        upload_fn = MagicMock(return_value=None)
        state = make_training_state()
        io_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path,
            save_debiased_ema=False,
            use_intra_epoch_checkpointing=True,
            intra_epoch_save_every_n_steps=1,
        )
        io_handler = TrainingIOHandler(io_config, data_upload_fun=upload_fn)
        io_handler.save_intra_epoch_checkpoint(state, step_number=1, epoch_number=1)
        io_handler.wait_until_finished()

        upload_fn.assert_called_once_with(tmp_path)

    def test_data_upload_fun_called_after_save_dataset_info(self, tmp_path):
        """upload_fn is called after save_dataset_info."""
        upload_fn = MagicMock(return_value=None)
        io_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path, save_debiased_ema=False
        )
        io_handler = TrainingIOHandler(io_config, data_upload_fun=upload_fn)

        dataset_info = DatasetInfo(
            atomic_energies_map={1: 0.0}, graph_cutoff_angstrom=5.0
        )
        io_handler.save_dataset_info(dataset_info)

        upload_fn.assert_called_once_with(tmp_path / "dataset_info.json")

    def test_future_awaited_before_next_upload(self, tmp_path, make_training_state):
        """Previous Future.result() is called before the next upload_fn call."""
        call_order = []

        future = MagicMock(spec=Future)
        future.result.side_effect = lambda: call_order.append("future.result")

        def fake_upload(path):
            call_order.append("upload")
            return future

        state = make_training_state()
        io_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path, save_debiased_ema=False
        )
        io_handler = TrainingIOHandler(io_config, data_upload_fun=fake_upload)

        # First save — no previous future to wait on
        io_handler.save_checkpoint(state, epoch_number=1)
        assert call_order == ["upload"]

        # Second save — should wait for the first future before uploading
        io_handler.save_checkpoint(state, epoch_number=2)
        assert call_order == ["upload", "future.result", "upload"]

    def test_wait_until_finished_blocks_on_upload_future(
        self, tmp_path, make_training_state
    ):
        """wait_until_finished() awaits the pending upload Future."""
        future = MagicMock(spec=Future)
        upload_fn = MagicMock(return_value=future)

        state = make_training_state()
        io_config = TrainingIOHandler.Config(
            checkpoint_dir=tmp_path, save_debiased_ema=False
        )
        io_handler = TrainingIOHandler(io_config, data_upload_fun=upload_fn)
        io_handler.save_checkpoint(state, epoch_number=1)

        future.result.assert_not_called()
        io_handler.wait_until_finished()
        future.result.assert_called_once()


def _make_mock_checkpointer(latest_metadata=None, latest_key=None, restore_result=None):
    """Create a mock OrbaxCheckpointer with configurable metadata/restore."""
    ckpt = MagicMock()
    ckpt.latest_metadata.return_value = latest_metadata
    ckpt.latest_key.return_value = latest_key
    ckpt.restore.return_value = restore_result
    return ckpt


class TestPickMostRecentCheckpoint:
    """Tests for _pick_most_recent_checkpoint logic."""

    def _make_handler(self, post_ckpt=None, intra_ckpt=None):
        """Build an IOHandler with injected mock checkpointers."""
        handler = TrainingIOHandler()
        handler._restore_post_epoch = post_ckpt
        handler._restore_intra_epoch = intra_ckpt
        return handler

    def test_no_intra_checkpointer(self):
        handler = self._make_handler(
            post_ckpt=_make_mock_checkpointer(), intra_ckpt=None
        )
        assert handler._pick_most_recent_checkpoint() is False

    def test_intra_has_no_metadata(self):
        handler = self._make_handler(
            post_ckpt=_make_mock_checkpointer(),
            intra_ckpt=_make_mock_checkpointer(latest_metadata=None),
        )
        assert handler._pick_most_recent_checkpoint() is False

    def test_intra_exists_no_post_checkpointer(self):
        meta = CheckpointMetadata(num_steps=10, epoch_number=1)
        handler = self._make_handler(
            post_ckpt=None,
            intra_ckpt=_make_mock_checkpointer(latest_metadata=meta),
        )
        assert handler._pick_most_recent_checkpoint() is True

    def test_intra_exists_post_has_no_metadata(self):
        intra_meta = CheckpointMetadata(num_steps=10, epoch_number=1)
        handler = self._make_handler(
            post_ckpt=_make_mock_checkpointer(latest_metadata=None),
            intra_ckpt=_make_mock_checkpointer(latest_metadata=intra_meta),
        )
        assert handler._pick_most_recent_checkpoint() is True

    def test_intra_more_recent_than_post(self):
        post_meta = CheckpointMetadata(num_steps=50, epoch_number=1)
        intra_meta = CheckpointMetadata(num_steps=75, epoch_number=2)
        handler = self._make_handler(
            post_ckpt=_make_mock_checkpointer(latest_metadata=post_meta),
            intra_ckpt=_make_mock_checkpointer(latest_metadata=intra_meta),
        )
        assert handler._pick_most_recent_checkpoint() is True

    def test_post_more_recent_than_intra(self):
        post_meta = CheckpointMetadata(num_steps=100, epoch_number=2)
        intra_meta = CheckpointMetadata(num_steps=75, epoch_number=2)
        handler = self._make_handler(
            post_ckpt=_make_mock_checkpointer(latest_metadata=post_meta),
            intra_ckpt=_make_mock_checkpointer(latest_metadata=intra_meta),
        )
        assert handler._pick_most_recent_checkpoint() is False

    def test_equal_num_steps_picks_post(self):
        meta = CheckpointMetadata(num_steps=100, epoch_number=2)
        handler = self._make_handler(
            post_ckpt=_make_mock_checkpointer(latest_metadata=meta),
            intra_ckpt=_make_mock_checkpointer(latest_metadata=meta),
        )
        assert handler._pick_most_recent_checkpoint() is False


class TestRestoreCheckpoint:
    """Tests for restore_checkpoint logic."""

    def _make_handler_with_config(self, config, post_ckpt=None, intra_ckpt=None):
        """Build an IOHandler with a config and injected mock checkpointers."""
        handler = TrainingIOHandler(config)
        handler._restore_post_epoch = post_ckpt
        handler._restore_intra_epoch = intra_ckpt
        return handler

    def _make_restore_result(self, state, num_steps=0, epoch_number=1, metrics=None):
        return RestoreResult(
            training_state=state,
            metadata=CheckpointMetadata(num_steps=num_steps, epoch_number=epoch_number),
            accumulated_metrics=metrics or [],
        )

    def test_restore_disabled_returns_original(self, make_training_state):
        config = TrainingIOHandler.Config(restore_checkpoint_if_exists=False)
        handler = self._make_handler_with_config(config)
        state = make_training_state()
        result_state, ds_state, epoch, metrics = handler.restore_checkpoint(state)
        assert result_state is state
        assert ds_state is None
        assert epoch == 0
        assert metrics == []

    def test_both_checkpointers_none_raises(self, make_training_state):
        config = TrainingIOHandler.Config(restore_checkpoint_if_exists=True)
        handler = self._make_handler_with_config(config)
        with pytest.raises(CheckpointRestorationError):
            handler.restore_checkpoint(make_training_state())

    def test_intra_wins_uses_epoch_minus_one(self, make_training_state):
        saved = make_training_state(num_steps=50)
        result = self._make_restore_result(
            saved, num_steps=50, epoch_number=3, metrics=[{"loss": 0.1}]
        )
        intra_meta = CheckpointMetadata(num_steps=50, epoch_number=3)
        config = TrainingIOHandler.Config(
            restore_checkpoint_if_exists=True,
            restore_optimizer_state=True,
        )
        handler = self._make_handler_with_config(
            config,
            post_ckpt=_make_mock_checkpointer(
                latest_metadata=CheckpointMetadata(num_steps=30, epoch_number=2)
            ),
            intra_ckpt=_make_mock_checkpointer(
                latest_metadata=intra_meta, restore_result=result
            ),
        )
        restored, _, epoch, metrics = handler.restore_checkpoint(make_training_state())
        assert epoch == 2  # epoch_number(3) - 1
        assert metrics == [{"loss": 0.1}]
        assert restored is saved

    def test_post_wins_uses_epoch_directly(self, make_training_state):
        saved = make_training_state(num_steps=100)
        result = self._make_restore_result(saved, num_steps=100, epoch_number=5)
        post_meta = CheckpointMetadata(num_steps=100, epoch_number=5)
        config = TrainingIOHandler.Config(
            restore_checkpoint_if_exists=True,
            restore_optimizer_state=True,
        )
        handler = self._make_handler_with_config(
            config,
            post_ckpt=_make_mock_checkpointer(
                latest_metadata=post_meta, latest_key=5, restore_result=result
            ),
            intra_ckpt=_make_mock_checkpointer(
                latest_metadata=CheckpointMetadata(num_steps=80, epoch_number=5)
            ),
        )
        restored, _, epoch, metrics = handler.restore_checkpoint(make_training_state())
        assert epoch == 5
        assert metrics == []

    def test_post_no_checkpoint_found_returns_scratch(self, make_training_state):
        config = TrainingIOHandler.Config(restore_checkpoint_if_exists=True)
        handler = self._make_handler_with_config(
            config,
            post_ckpt=_make_mock_checkpointer(latest_metadata=None, latest_key=None),
        )
        state = make_training_state()
        restored, _, epoch, metrics = handler.restore_checkpoint(state)
        assert restored is state
        assert epoch == 0

    def test_post_with_epoch_to_restore(self, make_training_state):
        saved = make_training_state(num_steps=60)
        result = self._make_restore_result(saved, num_steps=60, epoch_number=3)
        config = TrainingIOHandler.Config(
            restore_checkpoint_if_exists=True,
            restore_optimizer_state=True,
            epoch_to_restore=3,
        )
        post_ckpt = _make_mock_checkpointer(
            latest_metadata=CheckpointMetadata(num_steps=100, epoch_number=5),
            latest_key=5,
            restore_result=result,
        )
        handler = self._make_handler_with_config(config, post_ckpt=post_ckpt)
        restored, _, epoch, _ = handler.restore_checkpoint(make_training_state())
        post_ckpt.restore.assert_called_once()
        call_kwargs = post_ckpt.restore.call_args
        assert call_kwargs[1]["key"] == 3
        assert epoch == 3

    def test_restore_without_optimizer_replaces_params_only(self, make_training_state):
        saved = make_training_state(num_steps=42)
        result = self._make_restore_result(saved, num_steps=42, epoch_number=2)
        config = TrainingIOHandler.Config(
            restore_checkpoint_if_exists=True,
            restore_optimizer_state=False,
        )
        handler = self._make_handler_with_config(
            config,
            post_ckpt=_make_mock_checkpointer(
                latest_metadata=CheckpointMetadata(num_steps=42, epoch_number=2),
                latest_key=2,
                restore_result=result,
            ),
        )
        fresh = make_training_state(num_steps=0)
        restored, _, _, _ = handler.restore_checkpoint(fresh)
        # Params come from saved checkpoint
        assert (restored.params["w"] == saved.params["w"]).all()
        # num_steps restored from checkpoint
        assert int(restored.num_steps) == 42
        # But it's not the exact same object (dataclasses.replace was used)
        assert restored is not saved

    def test_intra_epoch_zero_metadata_returns_epoch_zero(self, make_training_state):
        """When intra-epoch checkpoint has epoch_number=0 (or None),
        last_completed_epoch should be 0 (not negative)."""
        saved = make_training_state(num_steps=10)
        result = self._make_restore_result(
            saved, num_steps=10, epoch_number=0, metrics=[]
        )
        config = TrainingIOHandler.Config(
            restore_checkpoint_if_exists=True,
            restore_optimizer_state=True,
        )
        handler = self._make_handler_with_config(
            config,
            intra_ckpt=_make_mock_checkpointer(
                latest_metadata=CheckpointMetadata(num_steps=10, epoch_number=0),
                restore_result=result,
            ),
        )
        _, _, epoch, _ = handler.restore_checkpoint(make_training_state())
        # (0 or 1) - 1 = 0
        assert epoch == 0
