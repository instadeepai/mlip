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
import time
from functools import partial
from typing import Callable, Optional, TypeAlias

import flax
import jax
import jraph
import numpy as np
import optax

from mlip.data.helpers.data_prefetching import PrefetchIterator
from mlip.data.helpers.graph_dataset import GraphDataset
from mlip.models import ForceField
from mlip.models.loss import Loss
from mlip.training.ema import exponentially_moving_average, get_debiased_params
from mlip.training.evaluation import make_evaluation_step, run_evaluation
from mlip.training.training_io_handler import LogCategory, TrainingIOHandler
from mlip.training.training_loggers import log_metrics_to_line
from mlip.training.training_loop_config import TrainingLoopConfig
from mlip.training.training_state import TrainingState, init_training_state
from mlip.training.training_step import make_train_step
from mlip.typing import ModelParameters

Optimizer: TypeAlias = optax.GradientTransformation
GraphDatasetOrPrefetchIterator: TypeAlias = GraphDataset | PrefetchIterator
TrainingStepFun: TypeAlias = Callable[
    [TrainingState, jraph.GraphsTuple],
    tuple[TrainingState, dict],
]

logger = logging.getLogger("mlip")


class TrainingLoop:
    """Training loop class.

    It implements only the loop based on its inputs but does not construct any
    auxiliary objects within it. For example, the model, dataset, and optimizer must
    be passed to this function from the outside.

    Attributes:
        training_state: The training state.
    """

    Config = TrainingLoopConfig

    def __init__(
        self,
        train_dataset: GraphDatasetOrPrefetchIterator,
        validation_dataset: GraphDatasetOrPrefetchIterator,
        force_field: ForceField,
        loss: Loss,
        optimizer: Optimizer,
        config: TrainingLoopConfig,
        io_handler: Optional[TrainingIOHandler] = None,
        should_parallelize: bool = False,
    ) -> None:
        """Constructor.

        Args:
            train_dataset: The training dataset as either a GraphDataset or
                           a PrefetchIterator.
            validation_dataset: The validation dataset as either a GraphDataset or
                                a PrefetchIterator.
            force_field: The force field model holding at least the initial parameters
                         and a dataset info object.
            loss: The loss, which it is derived from the `Loss` base class.
            optimizer: The optimizer (based on optax).
            config: The training loop pydantic config.
            io_handler: The IO handler which handles checkpointing
                        and (specialized) logging. This is an optional argument.
                        The default is `None`, which means that a default IO handler
                        will be set up which does not include checkpointing but some
                        very basic metrics logging.
            should_parallelize: Whether to parallelize (using data parallelization)
                                across multiple devices. The default is ``False``.
        """
        self.should_parallelize = should_parallelize

        self.train_dataset = train_dataset
        self.validation_dataset = (
            validation_dataset
            if config.eval_num_graphs is None
            else validation_dataset.subset(config.eval_num_graphs)
        )
        self.total_num_graphs, self.total_num_nodes = (
            self._get_total_number_of_graphs_and_nodes_in_train_set()
        )

        self.force_field = force_field
        self.dataset_info = self.force_field.dataset_info
        self.initial_params = self.force_field.params
        self.optimizer = optimizer
        self.config = config

        self.extended_metrics = (
            True if not hasattr(loss, "extended_metrics") else loss.extended_metrics
        )
        self.io_handler = io_handler
        if self.io_handler is None:
            self.io_handler = TrainingIOHandler()
            self.io_handler.attach_logger(log_metrics_to_line)

        self.io_handler.save_dataset_info(self.dataset_info)

        loss_train = partial(loss, eval_metrics=False)
        loss_eval = partial(loss, eval_metrics=True)

        self._prepare_training_state_and_ema()
        self.training_step = make_train_step(
            force_field.predictor,
            loss_train,
            self.optimizer,
            self.ema_fun,
            config.num_gradient_accumulation_steps,
            should_parallelize,
        )
        self.metrics = None
        self.eval_step = make_evaluation_step(
            force_field.predictor,
            loss_eval,
            should_parallelize,
        )

        self.best_evaluation_step = -1
        self.best_evaluation_loss = float("inf")
        self.best_params = None

        self._should_unreplicate_train_batches = (
            not should_parallelize
        ) and isinstance(self.train_dataset, PrefetchIterator)

        self.num_batches = len(self.train_dataset)
        self.steps_per_epoch = self.num_batches
        if should_parallelize:
            self.steps_per_epoch = (
                self.num_batches // len(jax.devices())
            ) // config.num_gradient_accumulation_steps
        self.epoch_number = self._get_epoch_number_from_training_state()

        logger.debug(
            "Training loop: Number of batches has been set to: %s", self.num_batches
        )
        logger.debug(
            "Training loop: Steps per epoch has been set to: %s", self.steps_per_epoch
        )

    def run(self) -> None:
        """Runs the training loop.

        The final training state can be accessed via its member variable.
        """
        logger.info("Starting training loop...")

        # May not be zero if restored from checkpoint
        if self.epoch_number > 0:
            self.io_handler.log(
                LogCategory.CLEANUP_AFTER_CKPT_RESTORATION, {}, self.epoch_number
            )

        if self.epoch_number == 0 and self.config.run_eval_at_start:
            logger.debug("Running initial evaluation...")
            start_time = time.perf_counter()
            self._run_evaluation()
            logger.debug(
                "Initial evaluation done in %.2f sec.", time.perf_counter() - start_time
            )

        while self.epoch_number < self.config.num_epochs:
            self.epoch_number += 1
            t_before_train = time.perf_counter()
            self._run_training_epoch()
            logger.debug(
                "Parameter updates of epoch %s done, running evaluation next.",
                self.epoch_number,
            )
            t_after_train = time.perf_counter()
            self._run_evaluation()
            t_after_eval = time.perf_counter()

            logger.debug(
                "Epoch %s done. Time for parameter updates: %.2f sec.",
                self.epoch_number,
                t_after_train - t_before_train,
            )
            logger.debug("Time for evaluation: %.2f sec.", t_after_eval - t_after_train)

        self.io_handler.wait_until_finished()

        logger.info("Training loop completed.")

    def _run_training_epoch(self) -> None:
        start_time = time.perf_counter()
        metrics = []

        for batch in self.train_dataset:
            if self._should_unreplicate_train_batches:
                batch = flax.jax_utils.unreplicate(batch)
            updated_training_state, _metrics = self.training_step(
                self.training_state, batch, self.epoch_number
            )
            self.training_state = updated_training_state
            metrics.append(jax.device_get(_metrics))

        epoch_time_in_seconds = time.perf_counter() - start_time
        self._log_after_training_epoch(
            metrics, self.epoch_number, epoch_time_in_seconds
        )

    def _run_evaluation(self) -> None:
        devices = jax.devices() if self.should_parallelize else None
        eval_loss = run_evaluation(
            self.eval_step,
            self.validation_dataset,
            self._eval_params_from_current_training_state(),
            self.epoch_number,
            self.io_handler,
            devices,
        )

        if self.epoch_number == 0:
            self.best_evaluation_loss = eval_loss
            self.best_evaluation_epoch = 0

        elif eval_loss < self.best_evaluation_loss:
            logger.debug(
                "New best epoch %s has evaluation loss: %.6f",
                self.epoch_number,
                eval_loss,
            )
            self.best_evaluation_loss = eval_loss
            self.best_evaluation_epoch = self.epoch_number
            self.best_params = self._eval_params_from_current_training_state()

            self.io_handler.save_checkpoint(
                (
                    flax.jax_utils.unreplicate(self.training_state)
                    if self.should_parallelize
                    else self.training_state
                ),
                self.epoch_number,
            )

        to_log = {
            "best_loss": self.best_evaluation_loss,
            "best_epoch": self.best_evaluation_epoch,
        }
        self.io_handler.log(LogCategory.BEST_MODEL, to_log, self.epoch_number)

    def test(self, test_dataset: GraphDatasetOrPrefetchIterator) -> None:
        """Run the evaluation on the test dataset with the best parameters seen so far.

        Args:
            test_dataset: The test dataset as either a GraphDataset or
                          a PrefetchIterator.
        """
        devices = jax.devices() if self.should_parallelize else None
        run_evaluation(
            self.eval_step,
            test_dataset,
            self.best_params,
            self.epoch_number,
            self.io_handler,
            devices,
            is_test_set=True,
        )

    def _prepare_training_state_and_ema(self) -> None:
        key = jax.random.PRNGKey(self.config.random_seed)
        self.ema_fun = exponentially_moving_average(self.config.ema_decay)
        key, init_key = jax.random.split(key, 2)

        training_state = init_training_state(
            self.initial_params, init_key, self.optimizer, self.ema_fun
        )

        # The following line only restores the training state if the associated
        # setting in self.io_handler is set to true.
        training_state = self.io_handler.restore_training_state(training_state)

        training_state = jax.device_put(training_state)

        # Note: DISABLED AS IT'S MEMORY INTENSIVE AND BUT LEFT FOR VALIDATION PURPOSES.
        # assert_pytrees_match_across_hosts(training_state)
        # logger.debug(f"Training state is identical across all workers.")

        if self.should_parallelize:
            # Distribute training state
            start_time = time.perf_counter()
            training_state = flax.jax_utils.replicate(training_state)
            logger.debug(
                "Distributed training state in %.2f sec.",
                time.perf_counter() - start_time,
            )

            # Distribute keys
            start_time = time.perf_counter()
            key, key_state = jax.random.split(key, 2)
            devices = jax.local_devices()
            keys = jax.device_put_sharded(
                list(jax.random.split(key_state, len(devices))),
                devices,
            )
            training_state = training_state.replace(
                key=keys,
                acc_steps=flax.jax_utils.replicate(0),
            )
            logger.debug(
                "Distributed keys in %.2f sec.", time.perf_counter() - start_time
            )

        self.training_state = training_state

    def _get_epoch_number_from_training_state(self) -> int:
        return self._get_num_steps_from_training_state() // self.steps_per_epoch

    def _get_num_steps_from_training_state(self) -> int:
        if self.should_parallelize:
            return int(self.training_state.num_steps[0].squeeze().block_until_ready())
        return int(self.training_state.num_steps.squeeze().block_until_ready())

    def _log_after_training_epoch(
        self,
        metrics: list[dict[str, np.ndarray]],
        epoch_number: int,
        epoch_time_in_seconds: float,
    ) -> None:
        _metrics = {}
        for metric_name in metrics[0].keys():
            _metrics[metric_name] = np.mean([m[metric_name] for m in metrics])

        try:
            opt_hyperparams = jax.device_get(
                self.training_state.optimizer_state.hyperparams
            )
            if self.should_parallelize:
                opt_hyperparams = flax.jax_utils.unreplicate(opt_hyperparams)
            if self.extended_metrics:
                _metrics["learning_rate"] = float(opt_hyperparams["lr"])
        except AttributeError:
            pass

        _metrics["runtime_in_seconds"] = epoch_time_in_seconds
        if self.extended_metrics:
            _metrics["nodes_per_second"] = self.total_num_nodes / epoch_time_in_seconds
            _metrics["graphs_per_second"] = (
                self.total_num_graphs / epoch_time_in_seconds
            )

        self.io_handler.log(LogCategory.TRAIN_METRICS, _metrics, epoch_number)

        logger.debug(
            "Total number of steps after epoch %s: %s",
            epoch_number,
            self._get_num_steps_from_training_state(),
        )

    def _get_total_number_of_graphs_and_nodes_in_train_set(self) -> tuple[int, int]:
        total_num_graphs = 0
        total_num_nodes = 0

        def _batch_generator():
            if isinstance(self.train_dataset, PrefetchIterator):
                for stacked_batch in self.train_dataset:
                    for i in range(stacked_batch.n_node.shape[0]):
                        yield jax.tree.map(lambda x, idx=i: x[idx], stacked_batch)
            else:
                for batch in self.train_dataset:
                    yield batch

        for _batch in _batch_generator():
            total_num_graphs += jraph.get_graph_padding_mask(_batch).sum()
            total_num_nodes += jraph.get_node_padding_mask(_batch).sum()

        return total_num_graphs, total_num_nodes

    def _eval_params_from_current_training_state(self) -> ModelParameters:
        ema_decay = (
            self.config.ema_decay if self.config.use_ema_params_for_eval else None
        )
        devices = jax.devices() if self.should_parallelize else None

        if ema_decay is not None and self.epoch_number > 0:
            if devices is not None:
                return jax.pmap(
                    get_debiased_params,
                    axis_name="device",
                    static_broadcasted_argnums=(1,),
                )(self.training_state.ema_state, ema_decay)

            return get_debiased_params(self.training_state.ema_state, ema_decay)

        return self.training_state.params

    @property
    def best_model(self) -> ForceField:
        """Returns the current state of the force field model with the best
        parameters so far.

        Returns:
            The force field model with the best parameters so far.
        """
        return ForceField(self.force_field.predictor, self.best_params)
