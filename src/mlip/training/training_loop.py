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
import warnings
from functools import partial
from typing import Callable, Optional, TypeAlias

import jax
import jraph
import numpy as np
import optax
from jax.sharding import Mesh

from mlip.data.helpers.data_prefetching import UnsqueezeGraphDatasetWrapper
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
from mlip.typing import GraphDatasetLike, ModelParameters
from mlip.utils.multihost import (
    DATA_PARALLELISM_AXIS_NAME,
    create_device_mesh,
    create_dp_sharding,
    create_replicated_sharding,
)

Optimizer: TypeAlias = optax.GradientTransformation
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
        train_dataset: GraphDatasetLike,
        validation_dataset: GraphDatasetLike,
        force_field: ForceField,
        loss: Loss,
        optimizer: Optimizer,
        config: TrainingLoopConfig,
        io_handler: Optional[TrainingIOHandler] = None,
        mesh: Optional[Mesh] = None,
        should_parallelize: Optional[bool] = None,
    ) -> None:
        """Constructor.

        Args:
            train_dataset: The training dataset (GraphDataset or PrefetchIterator).
            validation_dataset: The validation dataset (GraphDataset or
                PrefetchIterator).
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
            mesh: The device mesh to use for training and evaluation. If not provided,
                    a device mesh will be created automatically based on the available
                    devices.
            should_parallelize: Deprecated. Use ``mesh`` instead. This parameter
                                is ignored; SPMD sharding is always enabled via
                                the device mesh.
        """
        if should_parallelize is not None:
            warnings.warn(
                "The 'should_parallelize' parameter is deprecated and will be "
                "removed in a future release. Use the 'mesh' parameter instead. "
                "SPMD sharding is now always enabled via the device mesh. "
                "This parameter is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        if mesh is None:
            mesh = create_device_mesh()
        train_dataset, mesh = self._maybe_wrap_dataset(train_dataset, mesh)
        validation_dataset, mesh = self._maybe_wrap_dataset(validation_dataset, mesh)

        self.train_dataset = train_dataset
        self.validation_dataset = (
            validation_dataset
            if config.eval_num_graphs is None
            else validation_dataset.subset(config.eval_num_graphs)
        )
        self.total_num_graphs, self.total_num_nodes = (
            self._get_total_number_of_graphs_and_nodes_in_dataset(self.train_dataset)
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

        self._loss_train = partial(loss, eval_metrics=False)
        self._loss_eval = partial(loss, eval_metrics=True)

        self.mesh: Mesh = mesh
        self.replicated_sharding = create_replicated_sharding(self.mesh)
        self.sharded_sharding = create_dp_sharding(self.mesh)

        self._prepare_training_state_and_ema()
        # Note: Because we shuffle the training data between epochs, the following
        # value may slightly fluctuate during training, however, we assume
        # it being fixed, which is a solid approximation for datasets of typical size.
        _avg_n_graphs_train = self.total_num_graphs / len(self.train_dataset)
        _out_shardings = (self.replicated_sharding, self.replicated_sharding)
        _in_shardings = (
            self.replicated_sharding,
            self.sharded_sharding,
            self.replicated_sharding,
        )

        self.training_step = make_train_step(
            force_field.predictor,
            self._loss_train,
            self.optimizer,
            self.ema_fun,
            _avg_n_graphs_train,
            num_gradient_accumulation_steps=config.num_gradient_accumulation_steps,
            in_shardings=_in_shardings,
            out_shardings=_out_shardings,
        )
        self.metrics = None
        _avg_n_graphs_validation = (
            self._get_total_number_of_graphs_and_nodes_in_dataset(
                self.validation_dataset
            )[0]
            / len(self.validation_dataset)
        )
        _eval_in_shardings = (
            self.replicated_sharding,
            self.sharded_sharding,
            self.replicated_sharding,
        )
        _eval_out_shardings = self.replicated_sharding
        self.eval_step = make_evaluation_step(
            force_field.predictor,
            self._loss_eval,
            _avg_n_graphs_validation,
            in_shardings=_eval_in_shardings,
            out_shardings=_eval_out_shardings,
        )

        self.best_evaluation_epoch = 0
        self.best_evaluation_loss = float("inf")
        self._best_params = None
        self.num_batches = len(self.train_dataset)
        # Each host stacks n_local_devices batches, so the iterator
        # yields num_batches // n_local_devices stacked batches per host.
        self.steps_per_epoch = (
            self.num_batches // len(self.mesh.local_devices)
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
            logger.debug(
                "Time for evaluation: %.2f sec.",
                t_after_eval - t_after_train,
            )

        self.io_handler.wait_until_finished()

        logger.info("Training loop completed.")

    def _run_training_epoch(self) -> None:
        start_time = time.perf_counter()
        metrics = []

        for batch in self.train_dataset:
            updated_training_state, _metrics = self.training_step(
                self.training_state, batch, self.epoch_number
            )

            self.training_state = updated_training_state
            metrics.append(_metrics)

        # Ensure all training computations are complete before measuring time
        jax.tree.map(lambda x: x.block_until_ready(), self.training_state)
        epoch_time_in_seconds = time.perf_counter() - start_time

        # Transfer all metrics to host once at the end of epoch
        logging_start_time = time.perf_counter()
        self._log_after_training_epoch(
            metrics, self.epoch_number, epoch_time_in_seconds
        )
        logging_time = time.perf_counter() - logging_start_time
        logger.debug(
            "Logging after epoch %s done in %.2f sec.", self.epoch_number, logging_time
        )

    def _run_evaluation(self) -> None:
        eval_loss = run_evaluation(
            self.eval_step,
            self.validation_dataset,
            self._eval_params_from_current_training_state(),
            self.epoch_number,
            self.io_handler,
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
            self._best_params = self._eval_params_from_current_training_state()

            if jax.process_index() == 0:
                checkpoint_state = jax.device_get(self.training_state)
                self.io_handler.save_checkpoint(checkpoint_state, self.epoch_number)

        to_log = {
            "best_loss": self.best_evaluation_loss,
            "best_epoch": self.best_evaluation_epoch,
        }
        self.io_handler.log(LogCategory.BEST_MODEL, to_log, self.epoch_number)

    def test(self, test_dataset: GraphDatasetLike) -> None:
        """Run the evaluation on the test dataset with the best parameters seen so far.

        Args:
            test_dataset: The test dataset (GraphDataset or PrefetchIterator).
        """
        test_dataset, test_mesh = self._maybe_wrap_dataset(test_dataset, self.mesh)
        replicated = create_replicated_sharding(test_mesh)
        sharded = create_dp_sharding(test_mesh)

        # The following part needs to be recomputed each time as different test
        # sets could be passed in
        avg_n_graphs = self._get_total_number_of_graphs_and_nodes_in_dataset(
            test_dataset
        )[0] / len(test_dataset)
        test_eval_step = make_evaluation_step(
            self.force_field.predictor,
            self._loss_eval,
            avg_n_graphs,
            in_shardings=(replicated, sharded, replicated),
            out_shardings=replicated,
        )

        run_evaluation(
            test_eval_step,
            test_dataset,
            self._best_params,
            self.epoch_number,
            self.io_handler,
            is_test_set=True,
        )

    @staticmethod
    def _maybe_wrap_dataset(
        dataset: GraphDatasetLike, mesh: Mesh
    ) -> tuple[GraphDatasetLike, Mesh]:
        """Wrap a GraphDataset for vmap compatibility; pass others through."""
        if not isinstance(dataset, GraphDataset):
            return dataset, mesh

        if len(mesh.devices.flat) > 1:
            logger.warning(
                "GraphDataset only supports single-device training. "
                "Auto-restricting to 1 device. "
                "Request prefetching during data processing for multi-device training."
            )
            mesh = jax.make_mesh((1,), (DATA_PARALLELISM_AXIS_NAME,))

        return UnsqueezeGraphDatasetWrapper(dataset), mesh

    def _prepare_training_state_and_ema(self) -> None:
        self.ema_fun = exponentially_moving_average(self.config.ema_decay)

        logger.debug(
            "Device mesh: %s, devices=%s, sharded_sharding=%s",
            self.mesh,
            self.mesh.devices,
            self.sharded_sharding,
        )

        # Use jit with out_shardings to replicate across ALL devices,
        # including non-addressable devices in multi-host setups.
        @partial(jax.jit, out_shardings=self.replicated_sharding)
        def _init_and_replicate_training_state():
            training_state = init_training_state(
                self.initial_params, self.optimizer, self.ema_fun
            )

            # The following line only restores the training state if the associated
            # setting in self.io_handler is set to true.
            training_state = self.io_handler.restore_training_state(training_state)
            return training_state

        start_time = time.perf_counter()
        training_state = _init_and_replicate_training_state()
        logger.debug(
            "Replicated training state in %.2f sec.",
            time.perf_counter() - start_time,
        )

        self.training_state = training_state

    def _get_epoch_number_from_training_state(self) -> int:
        return self._get_num_steps_from_training_state() // self.steps_per_epoch

    def _get_num_steps_from_training_state(self) -> int:
        return int(self.training_state.num_steps.squeeze())

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
            opt_hyperparams = self.training_state.optimizer_state.hyperparams
            if self.extended_metrics:
                _metrics["learning_rate"] = float(opt_hyperparams["lr"])
        except AttributeError:
            pass

        self.io_handler.log(LogCategory.TRAIN_METRICS, _metrics, epoch_number)

        system_metrics = {"runtime_in_seconds": epoch_time_in_seconds}
        if self.extended_metrics:
            system_metrics["nodes_per_second"] = (
                self.total_num_nodes / epoch_time_in_seconds
            )
            system_metrics["graphs_per_second"] = (
                self.total_num_graphs / epoch_time_in_seconds
            )

        self.io_handler.log(LogCategory.SYSTEM_METRICS, system_metrics, epoch_number)

        logger.debug(
            "Total number of steps after epoch %s: %s",
            epoch_number,
            self._get_num_steps_from_training_state(),
        )

    def _get_total_number_of_graphs_and_nodes_in_dataset(
        self, dataset: GraphDatasetLike
    ) -> tuple[int, int]:
        total_num_graphs = 0
        total_num_nodes = 0

        # In multi-host mode, global arrays only expose local shards.
        # Use a JIT-compiled function to count across ALL devices so
        # every host gets identical totals.
        @jax.jit
        def _count_all(stacked_batch):
            def _count_single(batch):
                return (
                    jraph.get_graph_padding_mask(batch).sum(),
                    jraph.get_node_padding_mask(batch).sum(),
                )

            g, n = jax.vmap(_count_single)(stacked_batch)
            return g.sum(), n.sum()

        for stacked_batch in dataset:
            n_graphs, n_nodes = _count_all(stacked_batch)
            total_num_graphs += int(jax.device_get(n_graphs))
            total_num_nodes += int(jax.device_get(n_nodes))

        return total_num_graphs, total_num_nodes

    def _eval_params_from_current_training_state(self) -> ModelParameters:
        ema_decay = (
            self.config.ema_decay if self.config.use_ema_params_for_eval else None
        )

        if ema_decay is not None and self.epoch_number > 0:
            return get_debiased_params(self.training_state.ema_state, ema_decay)

        return self.training_state.params

    @property
    def best_model(self) -> ForceField:
        """Returns the force field model with the best parameters seen so far.

        Returns:
            The force field model with the best parameters so far.
        """
        params = jax.device_get(self._best_params)
        return ForceField(self.force_field.predictor, params)
