.. _batched_inference:

Batched Inference
=================

The function
:py:func:`run_batched_inference() <mlip.inference.batched_inference.run_batched_inference>`
can be used to efficiently compute model predictions on a collection of structures.

As in :ref:`simulation <simulations>` workflows, a trained model is passed via the
:py:class:`ForceField <mlip.models.force_field.ForceField>` container, wrapping
the `nn.Module` predictor with its learned parameters.
See :py:func:`load_model_from_zip() <mlip.models.model_io.load_model_from_zip>` for more
details on our recommended model loading strategy.

The :py:func:`run_batched_inference() <mlip.inference.batched_inference.run_batched_inference>`
function returns a list of :py:class:`Prediction <mlip.typing.prediction.Prediction>`
objects.

Data processing
---------------

There are two ways a user can provide input data:

* a list of `ase.Atoms` can be passed for convenience, e.g., for interoperability
  with simulation workflows.
* a :py:class:`GraphDataset <mlip.data.graph_dataset.GraphDataset>` or
  :py:class:`PrefetchIterator <mlip.data.helpers.data_prefetching.PrefetchIterator>`
  can alternatively be passed
  to support a variety of data sources via the
  :py:class:`SingleGraphDatasetBuilder <mlip.data.single_graph_dataset_builder.SingleGraphDatasetBuilder>`
  data processing pathway described :ref:`here <single_split_builder>`.

See the :ref:`data processing <data_processing>` section of this documentation
for more information on graph dataset creation.
Note that internally, passing `list[ase.Atoms]` will also reuse the
:py:class:`SingleGraphDatasetBuilder <mlip.data.single_graph_dataset_builder.SingleGraphDatasetBuilder>`
pathway, and that the `max_n_node`, `max_n_edge` and `batch_size` arguments
are only used when `list[ase.Atoms]` are given.

Example
-------

.. code-block:: python

    import ase.io
    from mlip.inference import run_batched_inference
    from mlip.models import Visnet
    from mlip.models.model_io import load_model_from_zip

    # 1. Load model
    force_field = load_model_from_zip(Visnet, "/path/to/visnet_model.zip")

    # 2. Prepare input data
    structures = ase.io.read("/path/to/molecule.xyz")

    # 3. Run batched inference
    predictions = run_batched_inference(
        structures,
        force_field,
        batch_size=8,
    )

    # Example: Get energy and forces for 4-th structure (indexing starts at 0)
    energy = predictions[3].energy
    forces = predictions[3].forces


For models that require graph-level inference metadata, pass it at loading time:

.. code-block:: python

    from mlip.models import InferenceContext, Visnet
    from mlip.models.model_io import load_model_from_zip

    force_field = load_model_from_zip(
        Visnet,
        "/path/to/visnet_model.zip",
        inference_context=InferenceContext(dataset_name="organics")
    )

See the :ref:`model user guide <models>` for details on the concept of the
:py:class:`InferenceContext <mlip.models.inference_context.InferenceContext>` class.

If additional properties are required that are not among the defaults
of the :py:class:`Properties <mlip.typing.properties.Properties>` class (e.g.,
atomic partial charges or Hessians), make sure to request those properties when
loading the model:

.. _code_block_req_properties:

.. code-block:: python

    from mlip.models import Visnet
    from mlip.models.model_io import load_model_from_zip

    force_field = load_model_from_zip(
        Visnet,
        "/path/to/visnet_model.zip",
        required_properties=Properties(partial_charges=True, hessian=True)
    )

Note that the required properties must be among the `available_properties` stored in
the model, otherwise an exception will be raised.
