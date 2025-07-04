.. _training:

Model training
==============

To start a model training, there are the following prerequisites:

* Loading and preprocessing a training and validation dataset, as described
  :ref:`here <data_processing>`.
* Initializing a force field, as described :ref:`here <model_init>`.
* Setting up a loss (see :ref:`below <training_loss>` for details).
* Setting up an optimizer (see :ref:`below <training_optimizer>` for details).
* Creating an instance of
  :py:class:`TrainingLoopConfig <mlip.training.training_loop_config.TrainingLoopConfig>`
  (can be accessed via `TrainingLoop.Config`, too).
* Optionally (has a default): Creating an instance of
  :py:class:`TrainingIOHandler <mlip.training.training_io_handler.TrainingIOHandler>`
  (see :ref:`below <training_io_handler>` for details).

Once these objects are set up, we can create an instance of
:py:class:`TrainingLoop <mlip.training.training_loop.TrainingLoop>` and start
the training run:

.. code-block:: python

    from mlip.training import TrainingLoop

    # Prerequisites
    train_set, validation_set, dataset_info = _get_dataset()  # placeholder
    force_field = _get_force_field()  # placeholder
    loss = _get_loss()  # placeholder
    optimizer = _get_optimizer ()  # placeholder
    io_handler = _get_training_loop_io_handler()  # placeholder
    config = TrainingLoop.Config(**config_kwargs)

    # Create TrainingLoop class
    training_loop = TrainingLoop(
        train_dataset=train_set,
        validation_dataset=validation_set,
        force_field=force_field,
        loss=loss,
        optimizer=optimizer,
        config=config,
        io_handler=io_handler,  # also has a default, does not need to be set
        should_parallelize=len(jax.devices()) > 1,  # has a default of False
    )

    # Start the model training
    training_loop.run()

The final :py:class:`TrainingState <mlip.training.training_state.TrainingState>`
can be accessed after the run like this:

.. code-block:: python

    final_training_state = training_loop.training_state
    final_params = final_training_state.params

However, the final parameters are not always the ones with the best
performance on the validation set, and hence,
you can also access these with ``training_loop.best_model.params``.
Therefore, use `training_loop.best_model` to get the
:py:class:`ForceField <mlip.models.force_field.ForceField>` instance that holds
the best parameters. If you want to save a
trained force field not only via the checkpointing API described further below,
you can also use the function
:py:func:`save_model_to_zip() <mlip.models.model_io.save_model_to_zip>` to save it
as a lightweight zip archive in case you only want to use it for inference tasks later,
as this archive does not include any training state.

Note that it is also possible to run an evaluation on a test dataset after training by
using the
:py:func:`test() <mlip.training.training_loop.TrainingLoop.test>` method of the
:py:class:`TrainingLoop <mlip.training.training_loop.TrainingLoop>` instance.

In the following, we describe the prerequisites listed above in more detail.

.. _training_loss:

Loss
----

All losses must be implemented as derived classes of
:py:class:`Loss <mlip.models.loss.Loss>`. We currently implement two losses, the
Mean-Squared-Error loss (:py:class:`MSELoss <mlip.models.loss.MSELoss>`), and the
Huber loss (:py:class:`HuberLoss <mlip.models.loss.HuberLoss>`), which are both losses
that are derived from a loss that computes errors for energies, forces, and stress,
and weights them according to some weighting schedule that can depend on the epoch
number (base class: :py:class:`WeightedEFSLoss <mlip.models.loss.WeightedEFSLoss>`).

If one wants to use the MSE loss for training, simply run this code to initialize it:

.. code-block:: python

    import optax
    from mlip.models.loss import MSELoss

    # uses default weight schedules
    loss = MSELoss()

    # uses a weight flip schedule
    energy_weight_schedule = optax.piecewise_constant_schedule(1.0, {100: 25.0})
    forces_weight_schedule = optax.piecewise_constant_schedule(25.0, {100: 0.04})
    loss = MSELoss(energy_weight_schedule, forces_weight_schedule)

For our two implemented losses, we also allow for computation of more extended metrics
by setting the `extended_metrics` argument to `True` in the loss constructor.
By default, it is `False`. See the documentation of
the :py:class:`call method <mlip.models.loss.WeightedEFSLoss.__call__>` of the class
:py:class:`WeightedEFSLoss <mlip.models.loss.WeightedEFSLoss>` for more information on
the returned metrics.

Furthermore, note that even though the loss class is supposed to provide these metrics
averaged just over a given input batch, we reweight these metrics based on the number
of real (not dummy) graphs per batch in the training loop, such that the
resulting metrics that are logged during training are accurately averaged
over the whole dataset.

.. _training_optimizer:

Optimizer
---------

The optimizer provided to the
:py:class:`TrainingLoop <mlip.training.training_loop.TrainingLoop>`
can be any `Optax optimizer <https://optax.readthedocs.io/en/latest/api/optimizers.html>`_,
however, this library also has a specialized pipeline that has been inspired by
`this <https://github.com/ACEsuit/mace>`_ PyTorch MACE implementation.
It is configurable via a
:py:class:`OptimizerConfig <mlip.training.optimizer_config.OptimizerConfig>` object that
has sensible defaults set for training MLIP models. However, we suggest to also check
out `our white paper <https://arxiv.org/abs/2505.22397>`_ for recommendations for
sensible ways to adapt the defaults for specific models, for instance, ViSNet and
NequIP seem to be more prone to NaNs with the default learning rate and benefit from
using a smaller one such as ``1e-4``.

The default MLIP optimizer can be set up like this:

.. code-block:: python

    from mlip.training import get_default_mlip_optimizer, OptimizerConfig

    # with default config
    optimizer = get_default_mlip_optimizer()

    # with modified config
    optimizer = get_default_mlip_optimizer(OptimizerConfig(**config_kwargs))

See the API reference for
:py:func:`get_default_mlip_optimizer <mlip.training.optimizer.get_default_mlip_optimizer>`
and
:py:class:`OptimizerConfig <mlip.training.optimizer_config.OptimizerConfig>`
for further details on how this MLIP optimizer works internally.

.. _training_io_handler:

IO handling and logging
-----------------------

During training, we want to allow for checkpointing of the training state and logging
of metrics. The
:py:class:`TrainingIOHandler <mlip.training.training_io_handler.TrainingIOHandler>`
class manages these tasks. It comes with its own config, the
:py:class:`TrainingIOHandlerConfig <mlip.training.training_io_handler.TrainingIOHandlerConfig>`,
which like most other configs in the library can be accessed
via `TrainingIOHandler.Config`. The IO handler uses
`Orbax Checkpointing <https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax_checkpoint_101.html>`_
to save and restore model checkpoints. Also, for loading a trained model for simulations or
other inference tasks, this library relies on loading these model checkpoints
(see :py:func:`load_parameters_from_checkpoint() <mlip.models.params_loading.load_parameters_from_checkpoint>`).
The local checkpointing location can be set in the config, however, uploading these checkpoints
to remote storage locations can be achieved via a provided data upload function:

.. code-block:: python

    import os
    from mlip.training import TrainingIOHandler

    io_config = TrainingIOHandler.Config(**config_kwargs)

    def remote_storage_sync_fun(source: str | os.PathLike) -> None:
        """Makes sure local data in source is uploaded to remote storage"""
        pass  # placeholder

    io_handler = TrainingIOHandler(io_config, remote_storage_sync_fun)

Locally, after the training run has started,
the checkpointing location will contain a ``dataset_info.json`` file with
the saved :py:class:`DatasetInfo <mlip.data.dataset_info.DatasetInfo>`
object, and a ``model`` subdirectory with all the model checkpoints, one for
each epoch that had the best model up to that point judging by validation set loss.
In this location, it is recommended to also save other metadata manually,
such as the applied model config.

For advanced logging, e.g., to an experiment tracking platform (such as
`Neptune <https://neptune.ai>`_), one can also attach custom logging functions to the
IO handler:

.. code-block:: python

    mlip.training.training_io_handler import LogCategory

    def train_logging_fun(
        category: LogCategory, to_log: dict[str, Any], epoch_number: int
    ) -> None:
    """Advanced logging function"""
        pass  # placeholder

    io_handler.attach_logger(train_logging_fun)

See the documentation of
:py:class:`LogCategory <mlip.training.training_io_handler.LogCategory>`
for more details on what type of data can be logged with such a logger during training.
Furthermore, this library provides built-in logging functions that can be attached
to the IO handler,
:py:func:`log_metrics_to_table() <mlip.training.training_loggers.log_metrics_to_table>`,
which prints the training metrics to the console in a nice table format (using
`Rich tables <https://rich.readthedocs.io/en/stable/tables.html>`_), or
:py:func:`log_metrics_to_line() <mlip.training.training_loggers.log_metrics_to_line>`,
which logs the metrics in a single line.

These logging functions automatically convert any MSE metrics to RMSE for easier
interpretation. Internally, we only keep track of MSE instead of RMSE because we must
ensure that the square root is taken at the very end and not before any averaging
across batches or devices happens. If one desires to do the same conversion in their
custom logging function, see
:py:func:`convert_mse_to_rmse_in_logs() <mlip.training.training_loggers.convert_mse_to_rmse_in_logs>`,
which is a helper function we provide for this task.

Note that it is possible to omit the `io_handler` argument in the
:py:class:`TrainingLoop <mlip.training.training_loop.TrainingLoop>` class. In that case,
a default IO handler is set up internally and used. This IO handler does not include
checkpointing, but it does have the
:py:func:`log_metrics_to_line() <mlip.training.training_loggers.log_metrics_to_line>`
logging function attached by default.
