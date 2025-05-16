.. _model_finetuning:

Model fine-tuning
=================

.. note::

   Currently, fine-tuning is only available for MACE models.

A common use case is fine-tuning a pre-trained MLIP foundation model
on additional data to improve its accuracy for specific types of chemical systems.

In the following, we describe how to fine-tune an MLIP model with this library. We
recall that an MLIP model can be trained using multiple read-out heads. Note that
currently, this is just implemented for the MACE architecture. The number of read-out
heads can be set via ``num_readout_heads`` in
:py:func:`MaceConfig <mlip.models.mace.config.MaceConfig>`.
By default, one trains a foundation model with only one read-out head. However, it does
not matter for this fine-tuning step whether a model already has *N* read-out heads,
it can be fine-tuned by adding more heads and optimizing their associated weights only.
Note that the final energy prediction of a model is obtained by summing the outputs
of the *N* read-out heads.

To fine-tune a given model, set up the new model with at least one more read-out head
than the foundation model you already have pre-trained.

.. code-block:: python

    from mlip.models import Mace, ForceField

    foundation_model_params = _get_params_for_pretrained_model()  # placeholder

    # Make sure the new model you create has at least one more read-out head
    mace = Mace(Mace.Config(num_readout_heads=2), dataset_info)
    initial_force_field = ForceField.from_mlip_network(mace)

Now, we can transfer the pre-trained parameters into the new parameter object by using
the function
:py:func:`transfer_params() <mlip.models.params_transfer.transfer_params>`:

.. code-block:: python

    from mlip.models.params_transfer import transfer_params

    transferred_params, finetuning_blocks = transfer_params(
        foundation_model_params,
        initial_force_field.params,
        scale_factor=0.1,
    )

As shown above, you have the option to rescale the randomly initialized
parameters of the additional heads by setting the keyword argument
``scale_factor`` accordingly. Rescaling the yet untrained
parameters to values close to zero can potentially aid model learning, as it initializes
the model to be close to the pre-trained model at the start of the training.
As part of our initial testing, we have found that models that are trained with a
scale factor of 1.0 could sometimes not be optimized back to the quality of the
pre-trained model. At the same time, a scale factor of 0.0 keeps all untrained
weights at zero at the start of the training which prevents proper gradient flow.

Therefore, **we recommend to apply a non-zero scale factor of about 0.1 or lower** which
worked well in our initial tests. However, we also encourage users
to experiment with this hyperparameter themselves.

The resulting ``transferred_params`` have the shape of your new model, but the new
heads are not yet optimized. The other parameters are taken from the pre-trained
foundation model. The second output of the function ``finetuning_blocks`` holds a list
of module names inside the parameters that correspond to the blocks of untrained
parameters. This list will be needed for the subsequent step.

In the final step of preparing a model fine-tuning, we need to mask the optimizer to
only update the untrained parameters. This can be easily done with the utility function
:py:func:`mask_optimizer_for_finetuning() <mlip.training.finetuning_utils.mask_optimizer_for_finetuning>`:

.. code-block:: python

    from mlip.training.finetuning_utils import mask_optimizer_for_finetuning

    optimizer = _set_up_optimizer_like_for_normal_model_training()  # placeholder

    masked_optimizer = mask_optimizer_for_finetuning(
        optimizer, transferred_params, finetuning_blocks
    )

    # Go on to set up a normal training with the masked_optimizer
    # and the transferred_params

Subsequently, fine-tuning this model works exactly like the normal model training.
All code can be reused. Creating a force field that is needed for training with
the transferred parameters works like this:

.. code-block:: python

    from mlip.models import ForceField

    force_field = ForceField(initial_force_field.predictor, transferred_params)


**To summarize, there are only three additional steps that are**
**required for fine-tuning in contrast to a regular model training:**

* Loading the original foundation model parameters *and* setting up a new model that
  has the same configuration but with one or more additional read-out heads.
* Transfer the parameters using the function
  :py:func:`transfer_params() <mlip.models.params_transfer.transfer_params>`.
* Mask the optimizer using the function
  :py:func:`mask_optimizer_for_finetuning() <mlip.training.finetuning_utils.mask_optimizer_for_finetuning>`.

**Additional note:** When fine-tuning on datasets that are quite different to the
original dataset which the foundation model was trained on, we recommend to add a subset
of the original dataset to the dataset the fine-tuning is performed on. The proportion
to which the original dataset should extend the new data points (e.g., 50:50 or
90:10 ratio) is a hyperparameter to experiment with and the optimal choice
may depend on how chemically different the new data is from the original data.
