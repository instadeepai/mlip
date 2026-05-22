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
from typing import Callable, TypeAlias

import jax
import numpy as np
import optax
from jax.sharding import Mesh

from mlip.data.helpers.data_prefetching import ParallelGraphDataset
from mlip.data.helpers.type_aliases import GraphDatasetLike
from mlip.graph import Graph
from mlip.models import ForceField
from mlip.models.loss import Loss
from mlip.training.ema import exponentially_moving_average, get_debiased_params
from mlip.training.evaluation import (
    EvaluationStepFun,
    make_evaluation_step,
    run_evaluation,
)
from mlip.training.training_io_handler import LogCategory, TrainingIOHandler
from mlip.training.training_loggers import log_metrics_to_line
from mlip.training.training_loop_config import TrainingLoopConfig
from mlip.training.training_state import TrainingState, init_training_state
from mlip.training.training_step import make_train_step
from mlip.typing import ModelParameters
from mlip.utils.multihost import (
    DATA_PARALLELISM_AXIS_NAME,
    create_device_mesh,
    create_dp_sharding,
    create_replicated_sharding,
)

Optimizer: TypeAlias = optax.GradientTransformation
TrainingStepFun: TypeAlias = Callable[
    [TrainingState, Graph],
    tuple[TrainingState, dict],
]
SubsetName: TypeAlias = str
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
        validation_dataset: GraphDatasetLike | dict[SubsetName, GraphDatasetLike],
        force_field: ForceField,
        loss: Loss,
        optimizer: Optimizer,
        config: TrainingLoopConfig,
        io_handler: TrainingIOHandler | None = None,
        mesh: Mesh | None = None,
    ) -> None:
        """Constructor.

        *Note:* This constructor updates the `add_atomic_energies` config field of the
        `MLIPNetwork` class of the force field, if requested via the
        `atomic_energies_removed` field in the dataset info. Hence, accessing
        `self.force_field` will possibly yield an updated force field.
        However, the method `best_model()` will return the original unmodified force
        field (but with the best parameters).

        Args:
            train_dataset: The training dataset (GraphDataset or PrefetchIterator).
            validation_dataset: The validation dataset (GraphDataset or
                PrefetchIterator). This can also be given as a dictionary of validation
                datasets instead. In that case, the metrics names during evaluation
                will be prefixed with the keys of that dictionary.
            force_field: The force field model holding at least the initial parameters
                         and a dataset info object.
            loss: The loss, which is derived from the `Loss` base class.
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
        """
        if mesh is None:
            mesh = create_device_mesh()

        mesh = self._restrict_mesh_for_graph_dataset(
            train_dataset,
            self._dataset_yields_parallel_batches(train_dataset),
            mesh,
        )
        val_sets = (
            validation_dataset.values()
            if isinstance(validation_dataset, dict)
            else [validation_dataset]
        )
        if len(mesh.devices.flat) > 1:
            for val_set in val_sets:
                if not self._dataset_yields_parallel_batches(val_set):
                    raise ValueError(
                        "Training dataset contains a ParallelGraphDataset but "
                        f"validation dataset ({type(val_set).__name__}) does "
                        "not. These must match: either both contain a "
                        "ParallelGraphDataset or neither does."
                    )

        self.train_dataset = train_dataset

        self.validation_dataset = validation_dataset
        if config.eval_num_graphs is not None:
            _n_graphs = config.eval_num_graphs
            if isinstance(self.validation_dataset, dict):
                self.validation_dataset = {
                    name: val_set.subset(_n_graphs)
                    for name, val_set in self.validation_dataset.items()
                }
            else:
                self.validation_dataset = self.validation_dataset.subset(_n_graphs)

        self.total_num_graphs = self.train_dataset.number_of_graphs()
        self.total_num_nodes = self.train_dataset.number_of_nodes()

        self.force_field = force_field
        self.dataset_info = self.force_field.dataset_info
        self.initial_params = self.force_field.params
        self.optimizer = optimizer
        self.config = config

        self._atomic_energies_removed = self.dataset_info.atomic_energies_removed
        if self._atomic_energies_removed:
            self._original_e0s_setting = self.force_field.config.add_atomic_energies
            self.force_field = self.force_field.replace_config(
                add_atomic_energies=False
            )

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
        self._replicate = jax.jit(lambda s: s, out_shardings=self.replicated_sharding)

        self._prepare_training_state_and_ema()
        self.training_step = self._make_training_step()
        self.eval_step = self._make_evaluation_step()
        self.metrics = None

        self.best_evaluation_epoch = 0
        self.best_evaluation_loss = float("inf")
        self._best_params = None
        self.num_batches = len(self.train_dataset)
        # Each host stacks n_local_devices batches, so the iterator
        # yields num_batches // n_local_devices stacked batches per host.
        self.steps_per_epoch = (
            self.num_batches // len(self.mesh.local_devices)
        ) // config.num_gradient_accumulation_steps
        self.epoch_number = self._last_completed_epoch

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

        # Prepend metrics restored from an intra-epoch checkpoint so
        # that the epoch-level averages include all steps, not just
        # those after resume.
        metrics: list = self._restored_intra_epoch_metrics
        self._restored_intra_epoch_metrics = []

        for batch in self.train_dataset:
            updated_training_state, _metrics = self.training_step(
                self.training_state, batch, self.epoch_number
            )

            self.training_state = updated_training_state
            metrics.append(_metrics)

            if self.io_handler.should_save_intra_epoch(
                int(self.training_state.num_steps)
            ):
                host_metrics = jax.device_get(metrics)
                self.io_handler.save_intra_epoch_checkpoint(
                    self.training_state,
                    step_number=int(self.training_state.num_steps),
                    epoch_number=self.epoch_number,
                    dataset_state=self._replicate(self.train_dataset.state),
                    accumulated_metrics=host_metrics,
                )

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

    def _make_training_step(self) -> Callable:
        """Function to return the training step that will then be set to
        the `self.training_step` attribute. Can be overridden to modify the default
        implementation."""
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
        return make_train_step(
            self.force_field.predictor,
            self._loss_train,
            self.optimizer,
            self.ema_fun,
            _avg_n_graphs_train,
            num_gradient_accumulation_steps=self.config.num_gradient_accumulation_steps,
            should_parallelize=self._dataset_yields_parallel_batches(
                self.train_dataset
            ),
            in_shardings=_in_shardings,
            out_shardings=_out_shardings,
        )

    def _make_evaluation_step(
        self,
    ) -> EvaluationStepFun | dict[SubsetName, EvaluationStepFun]:
        """Function to return the evaluation step that will then be set to
        the `self.eval_step` attribute. Can be overridden to modify the default
        implementation."""
        val_sets = (
            self.validation_dataset
            if isinstance(self.validation_dataset, dict)
            else {"main": self.validation_dataset}
        )

        _eval_in_shardings = (
            self.replicated_sharding,
            self.sharded_sharding,
            self.replicated_sharding,
        )
        _eval_out_shardings = self.replicated_sharding

        eval_steps = {}
        for subset_name, val_set in val_sets.items():
            _avg_n_graphs_validation = val_set.number_of_graphs() / len(val_set)
            eval_steps[subset_name] = make_evaluation_step(
                self.force_field.predictor,
                self._loss_eval,
                _avg_n_graphs_validation,
                should_parallelize=self._dataset_yields_parallel_batches(val_set),
                in_shardings=_eval_in_shardings,
                out_shardings=_eval_out_shardings,
            )

        return eval_steps if len(eval_steps) > 1 else eval_steps["main"]

    def _run_evaluation(self) -> None:
        # Using empty string for subset name if there are no subsets so that
        # in that case no prefix is added to metrics
        val_sets = (
            self.validation_dataset
            if isinstance(self.validation_dataset, dict)
            else {"": self.validation_dataset}
        )
        eval_steps = (
            self.eval_step
            if isinstance(self.validation_dataset, dict)
            else {"": self.eval_step}
        )

        eval_losses = []
        for subset_name, val_set in val_sets.items():
            _eval_loss = run_evaluation(
                eval_steps[subset_name],
                val_set,
                self._eval_params_from_current_training_state(),
                self.epoch_number,
                self.io_handler,
                is_test_set=False,
                subset_name=subset_name,
            )
            eval_losses.append(_eval_loss)

        # Combine evaluation losses, weighted by number of batches
        val_set_lengths = [len(val_set) for val_set in val_sets.values()]
        eval_loss = 0.0
        for _loss, _num_batches in zip(eval_losses, val_set_lengths):
            eval_loss += _loss * _num_batches
        eval_loss /= sum(val_set_lengths)

        if self.epoch_number == 0:
            self.best_evaluation_loss = eval_loss
            self.best_evaluation_epoch = 0
            # Seed `_best_params` so the epoch-0 checkpoint is the fallback if
            # no later epoch improves. Without this, `best_model` stays None
            # and `save_final_model_to_zip` crashes when training doesn't
            # improve from its starting point (common in FT / tiny runs).
            self._best_params = self._eval_params_from_current_training_state()

        elif eval_loss < self.best_evaluation_loss:
            logger.debug(
                "New best epoch %s has evaluation loss: %.6f",
                self.epoch_number,
                eval_loss,
            )
            self.best_evaluation_loss = eval_loss
            self.best_evaluation_epoch = self.epoch_number
            self._best_params = self._eval_params_from_current_training_state()

        if self.epoch_number > 0:
            self.io_handler.save_checkpoint(
                self.training_state,
                self.epoch_number,
                dataset_state=self._replicate(self.train_dataset.state),
                metrics={"eval_loss": eval_loss},
            )

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
        is_parallel = self._dataset_yields_parallel_batches(test_dataset)
        test_mesh = self._restrict_mesh_for_graph_dataset(
            test_dataset, is_parallel, self.mesh
        )
        replicated = create_replicated_sharding(test_mesh)
        sharded = create_dp_sharding(test_mesh)

        # The following part needs to be recomputed each time as different test
        # sets could be passed in
        avg_n_graphs = test_dataset.number_of_graphs() / len(test_dataset)
        test_eval_step = make_evaluation_step(
            self.force_field.predictor,
            self._loss_eval,
            avg_n_graphs,
            should_parallelize=is_parallel,
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
    def _dataset_yields_parallel_batches(dataset: GraphDatasetLike) -> bool:
        """Return True if the dataset yields stacked batches."""
        inner = dataset
        while hasattr(inner, "iterable"):
            inner = inner.iterable
        return isinstance(inner, ParallelGraphDataset)

    @staticmethod
    def _restrict_mesh_for_graph_dataset(
        dataset: GraphDatasetLike, is_parallel: bool, mesh: Mesh
    ) -> Mesh:
        """Force a 1-device mesh if the dataset doesn't yield device-stacked batches.

        Multi-device training requires a leading device axis on each batch,
        which is introduced by wrapping the dataset in a `ParallelGraphDataset`.
        Datasets without one (bare `GraphDataset` / `CombinedGraphDataset`)
        yield un-stacked batches and only support single-device training.
        """
        if not is_parallel and len(mesh.devices.flat) > 1:
            logger.warning(
                "%s only supports single-device training. "
                "Auto-restricting to 1 device. "
                "Requesting prefetching during data processing "
                "is suggested for multi-device training.",
                type(dataset).__name__,
            )
            return jax.make_mesh((1,), (DATA_PARALLELISM_AXIS_NAME,))
        return mesh

    def _prepare_training_state_and_ema(self) -> None:
        self.ema_fun = exponentially_moving_average(self.config.ema_decay)

        logger.debug(
            "Device mesh: %s, devices=%s, sharded_sharding=%s",
            self.mesh,
            self.mesh.devices,
            self.sharded_sharding,
        )

        self._restored_intra_epoch_metrics: list = []

        start_time = time.perf_counter()

        training_state = init_training_state(
            self.initial_params, self.optimizer, self.ema_fun
        )

        (
            training_state,
            restored_dataset_state,
            self._last_completed_epoch,
            self._restored_intra_epoch_metrics,
        ) = self.io_handler.restore_checkpoint(
            training_state,
            dataset_state=self.train_dataset.state,
            mesh=self.mesh,
        )

        self.training_state = self._replicate(training_state)

        logger.debug(
            "Initialized and replicated training state in %.2f sec.",
            time.perf_counter() - start_time,
        )

        if restored_dataset_state is not None:
            self.train_dataset.state = restored_dataset_state

    def _get_num_steps_from_training_state(self) -> int:
        return int(self.training_state.num_steps.squeeze())

    def _log_after_training_epoch(
        self,
        metrics: list[dict[str, np.ndarray]],
        epoch_number: int,
        epoch_time_in_seconds: float,
    ) -> None:
        _metrics = {}
        if not metrics:
            return
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

        if self._atomic_energies_removed:
            predictor = self.force_field.replace_config(
                add_atomic_energies=self._original_e0s_setting
            ).predictor
        else:
            predictor = self.force_field.predictor

        return ForceField(predictor, params)
