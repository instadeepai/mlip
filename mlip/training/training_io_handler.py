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
from typing import Any, Callable, Optional, TypeAlias

import orbax.checkpoint as ocp
import pydantic
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from typing_extensions import Annotated

from mlip.data.dataset_info import DatasetInfo
from mlip.training.ema import get_debiased_params
from mlip.training.training_state import TrainingState
from mlip.utils.multihost import single_host_jax_and_orbax

PathLike: TypeAlias = str | os.PathLike
Source: TypeAlias = PathLike
Target: TypeAlias = PathLike
MODEL_SUBDIR_NAME = "model"
DATASET_INFO_FILENAME = "dataset_info.json"
PositiveInt = Annotated[int, pydantic.Field(gt=0)]
EMADecay = Annotated[float, pydantic.Field(gt=0.0, le=1.0)]

logger = logging.getLogger("mlip")


class CheckpointRestorationError(Exception):
    """Exception to be raised if issues occur during checkpoint restoration."""


class TrainingIOHandlerConfig(pydantic.BaseModel):
    """Pydantic config holding all settings relevant for the training IO handler.

    Attributes:
        local_model_output_dir: Path to the output directory (local filesystem) where
                                the model/dataset information and checkpoints are
                                stored. If `None`, then local checkpointing will be
                                disabled. Defaults to `None`.
        max_checkpoints_to_keep: Maximum number of old checkpoints to keep.
                                 The default is 5.
        save_debiased_ema: Whether to also save the EMA parameters.
                           The default is ``True``.
        ema_decay: The EMA decay rate. The default is 0.99.
        restore_checkpoint_if_exists: Whether to restore a previous checkpoint if it
                                      exists. By default, this is ``False``.
        epoch_to_restore: The epoch number to restore. The default is ``None``, which
                          means the latest epoch will be restored.
        restore_optimizer_state: Whether to also restore the optimizer state.
                                 Default is ``False``.
        clear_previous_checkpoints: Whether to clear the previous checkpoints if
                                    any exist. Note that this setting can not be set to
                                    ``True`` if one selects to restore a checkpoint.
                                    The default is ``False``.
    """

    local_model_output_dir: Optional[PathLike] = None
    max_checkpoints_to_keep: PositiveInt = 5
    save_debiased_ema: bool = True
    ema_decay: EMADecay = 0.99

    restore_checkpoint_if_exists: bool = False
    epoch_to_restore: Optional[PositiveInt] = None
    restore_optimizer_state: bool = False
    clear_previous_checkpoints: bool = False


class LogCategory(Enum):
    """Enum class for logging categories.

    These values provide a signal to a logging function what type of data is
    being logged.

    Attributes:
        BEST_MODEL: Information about the current best model is logged.
        TRAIN_METRICS: Metrics for the training set are logged.
        EVAL_METRICS: Metrics for the validation set are logged.
        TEST_METRICS: Metrics for the test set are logged.
        CLEANUP_AFTER_CKPT_RESTORATION: Allows the logger to clean itself up after a
                                        checkpoint has been restored.
    """

    BEST_MODEL = 0
    TRAIN_METRICS = 1
    EVAL_METRICS = 2
    TEST_METRICS = 4
    CLEANUP_AFTER_CKPT_RESTORATION = 3


class TrainingIOHandler:
    """An IO handler class for the training loop.

    This handles checkpointing as well as specialized logging, e.g., to some external
    logger that a user can provide. If the config contains `None` for the local
    checkpointing directory, then this class will only do logging, but no
    checkpointing.
    """

    Config = TrainingIOHandlerConfig

    def __init__(
        self,
        config: Optional[TrainingIOHandlerConfig] = None,
        data_upload_fun: Optional[Callable[[Source], Optional[Future]]] = None,
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

        self._local_model_output_dir = None
        if self.config.local_model_output_dir is not None:
            self._local_model_output_dir = Path(config.local_model_output_dir).resolve()

        self._data_upload_fun = data_upload_fun
        self.loggers = []
        self._future = None
        self.ckpt_manager = self._configure_checkpointing()

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

    def save_dataset_info(self, dataset_info: DatasetInfo) -> None:
        """Save the dataset information class to disk in JSON format.

        Will also upload with data upload function if it exists.

        Args:
            dataset_info: The dataset information class to save.
        """
        if self._local_model_output_dir is None:
            return

        logger.debug("Saving/uploading dataset info...")

        start_time = time.perf_counter()
        local_json = self._local_model_output_dir / DATASET_INFO_FILENAME
        with local_json.open("w") as json_file:
            json.dump(json.loads(dataset_info.model_dump_json()), json_file, indent=4)
        if self._data_upload_fun is not None:
            self._data_upload_fun(local_json)

        logger.debug(
            "Dataset info was saved and possibly uploaded in %.2f sec.",
            time.perf_counter() - start_time,
        )

    def save_checkpoint(self, training_state: TrainingState, epoch_number: int) -> None:
        """Saves a model checkpoint to disk.

        Uses the data upload function as well if it exists.

        Args:
            training_state: The training state to save.
            epoch_number: The current epoch number.
        """
        if self._local_model_output_dir is None:
            return

        logger.info("Saving checkpoint at epoch %s...", epoch_number)

        ckpt = {"training_state": ocp.args.PyTreeSave(training_state)}
        if self.config.save_debiased_ema:
            ckpt["params_ema"] = ocp.args.PyTreeSave(
                get_debiased_params(training_state.ema_state, self.config.ema_decay)
            )

        if self._future is not None:
            self._future.result()

        with single_host_jax_and_orbax():
            self.ckpt_manager.save(epoch_number, args=ocp.args.Composite(**ckpt))

        if self._data_upload_fun is not None:
            self.ckpt_manager.wait_until_finished()
            logger.info("Uploading checkpoint at epoch %s...", epoch_number)
            self._future = self._data_upload_fun(self.config.local_model_output_dir)

    def restore_training_state(self, training_state: TrainingState) -> TrainingState:
        """Restores a training state from disk locally.

        Note that if one wants to restore from a remote location, first download the
        state outside of this function.

        Args:
            training_state: An instance of training state, which will serve as a
                            template for the restoration.

        Returns:
            The restored training state.

        """
        if not self.config.restore_checkpoint_if_exists:
            return training_state

        if self._local_model_output_dir is None:
            raise CheckpointRestorationError(
                "Cannot restore training state as checkpointing is disabled."
            )

        start_time = time.perf_counter()
        epoch_to_restore = self.config.epoch_to_restore
        if epoch_to_restore is None:
            epoch_to_restore = self.ckpt_manager.latest_step()

        logger.info("Restoring checkpoint from epoch %s.", epoch_to_restore)
        with single_host_jax_and_orbax():
            ckpt = self.ckpt_manager.restore(
                epoch_to_restore,
                args=ocp.args.Composite(
                    training_state=ocp.args.PyTreeRestore(training_state)
                ),
            )

        if self.config.restore_optimizer_state:
            logger.debug("Restoring params and optimizer state.")
            training_state = ckpt["training_state"]
        else:
            logger.debug("Restoring params, resetting optimizer state.")
            training_state = dataclasses.replace(
                training_state, params=ckpt["training_state"].params
            )

        logger.debug(
            "Checkpoint was restored in %.2f sec.", time.perf_counter() - start_time
        )

        return training_state

    def wait_until_finished(self) -> None:
        """Waits until the local checkpoint and `upload_fun` is finished due
        to their asynchronous nature. To be called at the end of a training run."""
        if self._local_model_output_dir is None:
            return

        self.ckpt_manager.wait_until_finished()
        if self._future is not None:
            self._future.result()

    def _configure_checkpointing(self) -> Optional[CheckpointManager]:
        if self._local_model_output_dir is None:
            return None

        options = CheckpointManagerOptions(
            save_interval_steps=1,
            max_to_keep=self.config.max_checkpoints_to_keep,
            create=True,
            cleanup_tmp_directories=True,
        )

        self._handle_already_existing_checkpoint_dir()

        item_names = ["training_state"]
        if self.config.save_debiased_ema:
            item_names.append("params_ema")

        with single_host_jax_and_orbax():
            # Orbax presumes directory is shared and so only calls mkdir on process 0.
            return CheckpointManager(
                self._local_model_output_dir / MODEL_SUBDIR_NAME,
                options=options,
                item_names=item_names,
            )

    def _handle_already_existing_checkpoint_dir(self) -> None:
        if (
            self.config.restore_checkpoint_if_exists
            and self.config.clear_previous_checkpoints
        ):
            raise CheckpointRestorationError(
                "Cannot both restore and clear previous checkpoints."
            )

        if self.config.clear_previous_checkpoints:
            if (
                self._local_model_output_dir.exists()
                and self._local_model_output_dir.is_dir()
            ):
                logger.debug("Deleting local checkpointing directory...")
                shutil.rmtree(self._local_model_output_dir)
            else:
                raise CheckpointRestorationError(
                    "Local checkpoint directory does not exist, "
                    "cannot clear previous checkpoints."
                )
        elif self.config.restore_checkpoint_if_exists:
            if (
                self._local_model_output_dir.exists()
                and self._local_model_output_dir.is_dir()
            ):
                logger.debug(
                    "Checkpointing directory exists locally and will be reused."
                )
            else:
                raise CheckpointRestorationError(
                    "Checkpoint cannot be restored because the local checkpoint "
                    "directory does not exist. Consider first downloading your "
                    "checkpoint directory from a remote storage to the local directory."
                )
