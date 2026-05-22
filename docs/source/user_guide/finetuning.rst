.. _model_finetuning:

Model fine-tuning
=================

This guide describes how to run multi-head fine-tuning (MHFT) with the *mlip* library.

.. note::

   As of version 0.2.0, the fine-tuning methodology has been revised and is now
   available for all implemented architectures: MACE, NequIP, ViSNet, and eSEN.

There are the following steps to the process:

* Load the pre-trained force field.
* Process the dataset consisting of replay data (from the original dataset) and fine-tuning
  dataset(s).
  Important: reuse the :py:class:`DatasetInfo <mlip.data.dataset_info.DatasetInfo>` from
  the pre-trained model for the replay dataset.
* Initialize a new force field with multiple readout heads and transfer pre-trained
  parameters into it.
* Train the model.
* Run inference with the model with a specific readout head.

Below, the details for each step are explained.

First, we load the pre-trained force field, e.g., MACE:

.. code-block:: python

    from mlip.models import Mace
    from mlip.models.model_io import load_model_from_zip

    pretrained_force_field = load_model_from_zip(Mace, "path/to/model.zip")


Then, we process the dataset with the `BuilderMode.MULTI` mode as described in the
data processing guide :ref:`here <multi_builder_mode>`. Make sure to pass the
`pretrained_force_field.dataset_info` of the pre-trained model to the graph dataset
builder, because for the replay dataset, the dataset info will be reused from the
pre-trained model.

Next, we instantiate the new force field with multiple readout heads. The rule
is one head per dataset key passed to the `MULTI` builder (i.e. `len(readers)`):
for example, two heads if we fine-tune on one dataset while still having the
replay data. To inspect how many heads a pretrained model has, use
:py:func:`count_readout_heads() <mlip.models.params_transfer.count_readout_heads>`.

.. code-block:: python

    from mlip.models import Mace, ForceField

    dataset_info = graph_dataset_builder.dataset_info

    mace = Mace(Mace.Config(num_readout_heads=2), dataset_info)
    force_field = ForceField.from_mlip_network(mace)

Transferring the pre-trained parameters to this new force field can be done like this:

.. code-block:: python

    from dataclasses import replace
    from mlip.models.params_transfer import transfer_params

    transferred_params = transfer_params(
        pretrained_force_field.params,
        force_field.params,
    )

    force_field = replace(force_field, params=transferred_params)

The above example uses the function
:py:func:`transfer_params() <mlip.models.params_transfer.transfer_params>`, see its
API reference for details. By default, newly added readout heads are warm-started
by deep-copying the pretrained head-0 weights, so fine-tuning begins from
pretrained readout values rather than a random init. Pass `scale_factor=0.0` to
fall back to a scaled random init for new blocks instead.

As a next step, we train the model as is described in the
:ref:`model training user guide <training>`.

Running inference with the trained model, for example, after it has been saved to zip
format, is straightforward via the
:py:class:`InferenceContext <mlip.models.inference_context.InferenceContext>`
concept. In the example below, we assume that the name given to the fine-tuning dataset
was `"ft"` and the model was saved to `"path/to/model.zip"`. Note that the
`dataset_name` value must match one of the keys passed to `readers` when building
the dataset (it is resolved against `DatasetInfo.dataset_name`):

.. code-block:: python

    from mlip.models import InferenceContext, Mace
    from mlip.models.model_io import load_model_from_zip

    force_field_ft = load_model_from_zip(
        Mace,
        "path/to/model.zip",
        inference_context=InferenceContext(dataset_name="ft"),
    )

    graph = _get_graph_from_somewhere()  # placeholder
    prediction = force_field_ft(graph)

Of course, technically, the force field can also be loaded with the context
`dataset_name="replay"`. After being loaded like above, in addition to single graph
inference, this model can also be used in simulations and batched inference.
