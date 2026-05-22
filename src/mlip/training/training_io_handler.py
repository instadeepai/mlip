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
import json
import logging
import os
import shutil
import time
from concurrent.futures import Future
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TypeAlias

import jax
import pydantic
from jax.experimental import multihost_utils
from jax.sharding import Mesh
from typing_extensions import Annotated

from mlip.data.dataset_info import DatasetInfo
from mlip.data.graph_dataset import GraphDatasetState
from mlip.training.checkpointer import (
    INTRA_EPOCH_SUBDIR_NAME,
    MODEL_SUBDIR_NAME,
    CheckpointerConfig,
    OrbaxCheckpointer,
)
from mlip.training.training_state import TrainingState

PathLike: TypeAlias = str | os.PathLike
Source: TypeAlias = PathLike

DATASET_INFO_FILENAME = "dataset_info.json"
PositiveInt = Annotated[int, pydantic.Field(gt=0)]
EMADecay = Annotated[float, pydantic.Field(gt=0.0, le=1.0)]

logger = logging.getLogger("mlip")


class CheckpointRestorationError(Exception):
    """Exception to be raised if issues occur during checkpoint restoration."""


class TrainingIOHandlerConfig(pydantic.BaseModel):
    """Pydantic config holding all settings relevant for the training IO handler.

    Attributes:
        checkpoint_dir: Root checkpoint directory. Supports both local paths
                        and can also support Paths supported by Orbax. When `None`,
                        dataset-info saving and the `clear_previous_checkpoints`
                        guard are skipped.  Defaults to `None`.
        restore_dir: Directory to restore checkpoints from. If `None`, will
                     default to `checkpoint_dir`.
        max_to_keep: Maximum number of post-epoch checkpoints to retain.
                     The default is 5.
        save_debiased_ema: Whether to also save the EMA parameters.
                           The default is `True`.
        ema_decay: The EMA decay rate. The default is 0.99.
        use_intra_epoch_checkpointing: Whether to also use intra-epoch checkpointing.
                                       The default is `False`.
        intra_epoch_save_every_n_steps: Save an intra-epoch checkpoint every
                                        N training steps.  `None` disables
                                        intra-epoch checkpointing.
        intra_epoch_max_to_keep: Maximum intra-epoch checkpoints to retain.
        restore_checkpoint_if_exists: Whether to restore a previous checkpoint if it
                                      exists. By default, this is `False`.
        epoch_to_restore: The epoch number to restore. The default is `None`, which
                          means the latest epoch will be restored.
        restore_optimizer_state: Whether to also restore the optimizer state.
                                 Default is `False`.
        clear_previous_checkpoints: Whether to clear the previous checkpoints if
                                    any exist. Note that this setting can not be set to
                                    `True` if one selects to restore a checkpoint.
                                    The default is `False`.
        use_single_host_patch: Whether to use single-host (process-0-only)
                               checkpointing.  When `True`, Orbax writes
                               without multi-host coordination and async
                               checkpointing is disabled. Defaults to `False`.
        enable_async_checkpointing: Whether Orbax should write checkpoints
                                    asynchronously.  Defaults to `False`.
                                    Incompatible with `use_single_host_patch`;
                                    see :class:`CheckpointerConfig`.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    checkpoint_dir: PathLike | None = None
    restore_dir: PathLike | None = None

    # Checkpointer settings
    max_to_keep: PositiveInt = 5
    save_debiased_ema: bool = True
    ema_decay: EMADecay = 0.99
    use_single_host_patch: bool = False
    enable_async_checkpointing: bool = False

    # Intra-epoch checkpointing
    use_intra_epoch_checkpointing: bool = False
    intra_epoch_save_every_n_steps: PositiveInt | None = 100
    intra_epoch_max_to_keep: PositiveInt | None = 5

    # Restoration settings
    restore_checkpoint_if_exists: bool = False
    epoch_to_restore: PositiveInt | None = None
    restore_optimizer_state: bool = False
    clear_previous_checkpoints: bool = False

    @pydantic.model_validator(mode="after")
    def _disallow_clear_with_restore(self) -> "TrainingIOHandlerConfig":
        if self.clear_previous_checkpoints and self.restore_checkpoint_if_exists:
            raise ValueError(
                "clear_previous_checkpoints=True is incompatible with "
                "restore_checkpoint_if_exists=True: clearing first would "
                "delete the checkpoints intended for restoration."
            )
        return self


class LogCategory(Enum):
    """Enum class for logging categories.

    These values provide a signal to a logging function what type of data is
    being logged.

    Attributes:
        BEST_MODEL: Information about the current best model is logged.
        TRAIN_METRICS: Metrics for the training set are logged.
        EVAL_METRICS: Metrics for the validation set are logged.
        TEST_METRICS: Metrics for the test set are logged.
        SYSTEM_METRICS: Per-process system metrics (runtime, throughput) are logged.
        CLEANUP_AFTER_CKPT_RESTORATION: Allows the logger to clean itself up after a
                                        checkpoint has been restored.
    """

    BEST_MODEL = 0
    TRAIN_METRICS = 1
    EVAL_METRICS = 2
    TEST_METRICS = 3
    SYSTEM_METRICS = 4
    CLEANUP_AFTER_CKPT_RESTORATION = 5


class TrainingIOHandler:
    """An IO handler class for the training loop.

    This handles checkpointing as well as specialized logging, e.g., to some external
    logger that a user can provide. Checkpointing is delegated to
    :class:`OrbaxCheckpointer` instances for post-epoch and intra-epoch saves.
    """

    Config = TrainingIOHandlerConfig
    Checkpointer = OrbaxCheckpointer

    def __init__(
        self,
        config: TrainingIOHandlerConfig | None = None,
        data_upload_fun: Callable[[Source], Future | None] | None = None,
    ) -> None:
        """Constructor.

        Args:
            config: The training IO handler pydantic config. Can be `None` in which
                    case the default config will be used. Default is `None`.
            data_upload_fun: A data upload function to a remote storage.
                             This is optional, and set to None as default.
                             This function should just take in a source path, and then
                             the upload location can be user-defined within that
                             function. The function can be asynchronous in which case it
                             should return a Future.
        """
        self.config = config
        if self.config is None:
            self.config = TrainingIOHandlerConfig()

        self._checkpoint_dir: os.PathLike | None = None
        if self.config.checkpoint_dir is not None:
            cd = self.config.checkpoint_dir
            self._checkpoint_dir = (
                cd if isinstance(cd, os.PathLike) else Path(cd).resolve()
            )

        self._data_upload_fun = data_upload_fun
        self._future: Future | None = None

        self._post_epoch_checkpointer = self._create_post_epoch_checkpointer()
        self._intra_epoch_checkpointer = self._create_intra_epoch_checkpointer()

        rd = self.config.restore_dir
        if rd is not None:
            self.restore_dir = rd if isinstance(rd, os.PathLike) else Path(rd).resolve()
        else:
            self.restore_dir = self._checkpoint_dir
        self._restore_post_epoch = None
        self._restore_intra_epoch = None
        if self.restore_dir is not None:
            if self.restore_dir == self._checkpoint_dir:
                self._restore_post_epoch = self._post_epoch_checkpointer
                if self.config.use_intra_epoch_checkpointing:
                    self._restore_intra_epoch = self._intra_epoch_checkpointer
            else:
                self._restore_post_epoch = self._create_restore_checkpointer(
                    MODEL_SUBDIR_NAME
                )
                if self.config.use_intra_epoch_checkpointing:
                    self._restore_intra_epoch = self._create_restore_checkpointer(
                        INTRA_EPOCH_SUBDIR_NAME,
                        save_metrics=True,
                    )

        self.loggers = []

        if self.config.clear_previous_checkpoints:
            self._clear_checkpoints()

    def _create_post_epoch_checkpointer(self) -> OrbaxCheckpointer | None:
        """Create the post-epoch checkpointer from the IO handler config."""
        if self._checkpoint_dir is None:
            return None
        ckpt_dir = self._checkpoint_dir / MODEL_SUBDIR_NAME
        config = CheckpointerConfig(
            checkpoint_dir=ckpt_dir,
            max_to_keep=self.config.max_to_keep,
            save_interval_steps=1,
            save_debiased_ema=self.config.save_debiased_ema,
            ema_decay=self.config.ema_decay,
            save_metrics=False,
            use_single_host_patch=self.config.use_single_host_patch,
            enable_async_checkpointing=self.config.enable_async_checkpointing,
        )
        return self.Checkpointer(config)

    def _create_intra_epoch_checkpointer(self) -> OrbaxCheckpointer | None:
        """Create the intra-epoch checkpointer from the IO handler config."""
        if (
            not self.config.use_intra_epoch_checkpointing
            or self._checkpoint_dir is None
            or self.config.intra_epoch_save_every_n_steps is None
        ):
            return None
        ckpt_dir = self._checkpoint_dir / INTRA_EPOCH_SUBDIR_NAME
        config = CheckpointerConfig(
            checkpoint_dir=ckpt_dir,
            max_to_keep=self.config.intra_epoch_max_to_keep,
            save_interval_steps=self.config.intra_epoch_save_every_n_steps,
            save_debiased_ema=self.config.save_debiased_ema,
            ema_decay=self.config.ema_decay,
            save_metrics=True,
            use_single_host_patch=self.config.use_single_host_patch,
            enable_async_checkpointing=self.config.enable_async_checkpointing,
        )
        return self.Checkpointer(config)

    def _create_restore_checkpointer(
        self, subdir: str, *, save_metrics: bool = False
    ) -> OrbaxCheckpointer:
        """Create a read-only checkpointer pointed at the restore directory."""
        ckpt_dir = self.restore_dir / subdir
        config = CheckpointerConfig(
            checkpoint_dir=ckpt_dir,
            max_to_keep=None,
            save_debiased_ema=self.config.save_debiased_ema,
            ema_decay=self.config.ema_decay,
            save_metrics=save_metrics,
            use_single_host_patch=self.config.use_single_host_patch,
            enable_async_checkpointing=self.config.enable_async_checkpointing,
        )
        return self.Checkpointer(config)

    def should_save_intra_epoch(self, step_number: int) -> bool:
        """Return `True` if an intra-epoch checkpoint should be saved now."""
        if self._intra_epoch_checkpointer is None:
            return False
        return self._intra_epoch_checkpointer.should_save(step_number)

    def attach_logger(
        self, logger: Callable[[LogCategory, dict[str, Any], int], None]
    ) -> None:
        """Attaches one training loop logging function to the IO handler.

        The logging function must take in three parameter and should not return
        anything. The three parameters are a logging category which describes what
        type of data is logged (it is an enum), the data dictionary to log, and
        the current epoch number.

        Args:
            logger: The logging function to add.
        """
        self.loggers.append(logger)

    def log(
        self, category: LogCategory, to_log: dict[str, Any], epoch_number: int
    ) -> None:
        """Logs data via the logging functions stored in this class.

        Args:
            category: A logging category which describes what type of data is
                      logged (it is an enum)
            to_log: A data dictionary to log (typically, metrics).
            epoch_number: The current epoch number.
        """
        for logger in self.loggers:
            logger(category, to_log, epoch_number)

    def _run_upload(self) -> None:
        """Upload the checkpoint directory using the configured upload function.

        Waits for any previously returned :class:`Future` before starting the
        next upload, ensuring sequential upload ordering.
        """
        if self._data_upload_fun is None or self._checkpoint_dir is None:
            return

        self.wait_until_finished()
        self._future = self._data_upload_fun(self._checkpoint_dir)

    def save_dataset_info(self, dataset_info: DatasetInfo) -> None:
        """Save the dataset information class to disk in JSON format.

        Will also upload with data upload function if it exists.

        Args:
            dataset_info: The dataset information class to save.
        """
        if self._checkpoint_dir is None:
            return

        logger.debug("Saving/uploading dataset info...")

        start_time = time.perf_counter()
        dataset_info_json = self._checkpoint_dir / DATASET_INFO_FILENAME
        with dataset_info_json.open("w") as json_file:
            json.dump(json.loads(dataset_info.model_dump_json()), json_file, indent=4)
        if self._data_upload_fun is not None:
            self._data_upload_fun(dataset_info_json)

        logger.debug(
            "Dataset info was saved and possibly uploaded in %.2f sec.",
            time.perf_counter() - start_time,
        )

    def save_intra_epoch_checkpoint(
        self,
        training_state: TrainingState,
        step_number: int,
        epoch_number: int,
        dataset_state: GraphDatasetState | None = None,
        accumulated_metrics: list | None = None,
    ) -> None:
        """Saves an intra-epoch checkpoint keyed by global step number.

        Delegates to the injected intra-epoch :class:`OrbaxCheckpointer`.

        Args:
            training_state: The training state to save (on-device arrays).
            step_number: The global training step (used as checkpoint key).
            epoch_number: The current epoch number (stored in metadata).
            dataset_state: Iterator state of the training dataset, persisted
                           as a sibling item so it can be restored separately.
                           In multi-host mode the caller must pass a
                           globally-addressable pytree here.
            accumulated_metrics: Per-step training metrics collected so far
                                 in the current epoch.
        """
        if self._intra_epoch_checkpointer is None:
            return

        self._intra_epoch_checkpointer.save(
            training_state,
            key=step_number,
            epoch_number=epoch_number,
            dataset_state=dataset_state,
            accumulated_metrics=accumulated_metrics,
        )

        self._run_upload()

    def save_checkpoint(
        self,
        training_state: TrainingState,
        epoch_number: int,
        dataset_state: GraphDatasetState | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Saves a model checkpoint using the post-epoch checkpointer.

        Args:
            training_state: The training state to save.
            epoch_number: The current epoch number.
            dataset_state: Iterator state of the training dataset, persisted
                           as a sibling item so it can be restored separately.
                           In multi-host mode the caller must pass a
                           globally-addressable pytree here.
            metrics: Scalar metrics used for best-checkpoint ranking.
        """
        if self._post_epoch_checkpointer is None:
            return

        logger.info("Saving checkpoint at epoch %s...", epoch_number)
        self._post_epoch_checkpointer.save(
            training_state,
            key=epoch_number,
            epoch_number=epoch_number,
            dataset_state=dataset_state,
            metrics=metrics,
        )

        self._run_upload()

    def restore_checkpoint(
        self,
        training_state: TrainingState,
        dataset_state: GraphDatasetState | None = None,
        mesh: Mesh | None = None,
    ) -> tuple[TrainingState, GraphDatasetState | None, int, list]:
        """Restores a training state from disk locally.

        When intra-epoch checkpointing is enabled, both the post-epoch and
        intra-epoch checkpoint managers are inspected. The checkpoint with the
        higher `num_steps` (read from Orbax custom metadata) wins.  After
        restoring, any intra-epoch checkpoints are cleared so the next run
        starts fresh.

        Args:
            training_state: An instance of training state, which will serve as a
                            template for the restoration.
            dataset_state: Optional dataset state template. When provided, the
                           dataset state is restored from the checkpoint and
                           returned alongside the training state.
            mesh: Optional JAX :class:`Mesh` for hardware-agnostic restoration.

        Returns:
            A tuple of
            `(restored_training_state, restored_dataset_state,
            last_completed_epoch, metrics)`.
            `restored_dataset_state` is `None` when no template is
            provided or no checkpoint exists.
            The epoch number is always read from the checkpoint metadata.
            For post-epoch checkpoints it is the epoch that was completed.
            For intra-epoch checkpoints it is `epoch_number - 1` (the
            last *fully* completed epoch).  `metrics` is the list of
            per-step training metrics accumulated before the intra-epoch
            checkpoint was taken (empty for post-epoch or fresh starts).
        """
        if not self.config.restore_checkpoint_if_exists:
            return training_state, None, 0, []

        post_ckpt = self._restore_post_epoch
        intra_ckpt = self._restore_intra_epoch

        if post_ckpt is None and intra_ckpt is None:
            raise CheckpointRestorationError(
                "Cannot restore training state as checkpointing is disabled."
            )

        start_time = time.perf_counter()

        use_intra = self._pick_most_recent_checkpoint()

        accumulated_metrics: list = []

        if use_intra:
            result = intra_ckpt.restore(
                training_state, mesh=mesh, dataset_state=dataset_state
            )
            accumulated_metrics = result.accumulated_metrics
            # The checkpoint was taken mid-epoch, so the last *completed*
            # epoch is one before the stored epoch_number.
            last_completed_epoch = (result.metadata.epoch_number or 1) - 1
        elif post_ckpt is not None:
            epoch_to_restore = self.config.epoch_to_restore or post_ckpt.latest_key()
            result = None
            if epoch_to_restore is not None:
                logger.info("Restoring checkpoint from epoch %s.", epoch_to_restore)
                result = post_ckpt.restore(
                    training_state,
                    key=epoch_to_restore,
                    mesh=mesh,
                    dataset_state=dataset_state,
                )

            if epoch_to_restore is None or result is None:
                logger.info("No checkpoint found to restore — starting from scratch.")
                return training_state, None, 0, []

            last_completed_epoch = result.metadata.epoch_number
        else:
            logger.info("No checkpoint found to restore — starting from scratch.")
            return training_state, None, 0, []

        # Apply restored state — either full state or params/EMA only.
        if self.config.restore_optimizer_state:
            logger.debug("Restoring params and optimizer state.")
            training_state = result.training_state
        else:
            logger.debug("Restoring params and EMA, resetting optimizer state.")
            training_state = dataclasses.replace(
                training_state,
                params=result.training_state.params,
                ema_state=result.training_state.ema_state,
                num_steps=result.training_state.num_steps,
            )

        logger.debug(
            "Checkpoint was restored in %.2f sec.",
            time.perf_counter() - start_time,
        )

        return (
            training_state,
            result.dataset_state,
            last_completed_epoch,
            (accumulated_metrics),
        )

    def _pick_most_recent_checkpoint(
        self,
    ) -> bool:
        """Compare post-epoch and intra-epoch checkpoints to find the most recent.

        Returns:
            `True` when the intra-epoch checkpoint is more recent.
        """
        post_ckpt = self._restore_post_epoch
        intra_ckpt = self._restore_intra_epoch

        if intra_ckpt is None:
            return False

        intra_meta = intra_ckpt.latest_metadata()
        if intra_meta is None:
            return False

        # No post-epoch checkpointer — intra-epoch wins by default.
        if post_ckpt is None:
            return True

        post_meta = post_ckpt.latest_metadata()
        if post_meta is None:
            # No post-epoch checkpoint exists; use intra-epoch.
            return True

        if intra_meta.num_steps > post_meta.num_steps:
            logger.info(
                "Intra-epoch checkpoint (num_steps=%s) is more "
                "recent than post-epoch (num_steps=%s).",
                intra_meta.num_steps,
                post_meta.num_steps,
            )
            return True

        return False

    def _clear_checkpoints(self) -> None:
        """Remove all existing checkpoints from the checkpoint directory.

        Called during `__init__` when `clear_previous_checkpoints` is
        `True`.  Clears both post-epoch and intra-epoch checkpoint managers.
        """
        if self._checkpoint_dir is None:
            return
        if jax.process_index() == 0:
            logger.info("Clearing previous checkpoints in %s.", self._checkpoint_dir)
            for subdir in (MODEL_SUBDIR_NAME, INTRA_EPOCH_SUBDIR_NAME):
                sub = self._checkpoint_dir / subdir
                if sub.exists():
                    shutil.rmtree(sub)
                    logger.debug("Removed %s.", sub)

        if jax.process_count() > 1:
            multihost_utils.sync_global_devices("clear_checkpoints")

    def wait_until_finished(self) -> None:
        """Waits until local checkpoints and uploads are finished due to their
        asynchronous nature. To be called at the end of a training run.
        """
        if self._post_epoch_checkpointer is not None:
            self._post_epoch_checkpointer.wait_until_finished()
        if self._intra_epoch_checkpointer is not None:
            self._intra_epoch_checkpointer.wait_until_finished()
        if self._future is not None:
            self._future.result()
