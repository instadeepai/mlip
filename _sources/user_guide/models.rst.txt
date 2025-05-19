.. _models:

Models
======

.. _model_init:

Create a model and force field
--------------------------------

This section discusses how to initialize an MLIP model for subsequent training.
If you are just interested in loading a pretrained model for application in simulations,
please see the dedicated section :ref:`below <load_zip_model>`.

Our MLIP models exist in two abstraction levels:

* On the one hand, we have the pure neural networks,
  which are classes derived from
  :py:class:`MLIPNetwork <mlip.models.mlip_network.MLIPNetwork>`. As a general rule,
  these raw models take in as input a graph's edge vectors and node representations and
  output a vector of node energies.

* On the other hand, we wrap these models into force
  fields which take care of computing properties such as total energy, forces, or stress
  from the MLIP network's output and themselves take a `jraph.GraphsTuple` object
  from the `jraph <https://jraph.readthedocs.io/en/latest/>`_
  library as input. The flax module that implements this is
  :py:class:`ForceFieldPredictor <mlip.models.predictor.ForceFieldPredictor>`, however,
  we recommend to mostly interact with the class
  :py:class:`ForceField <mlip.models.force_field.ForceField>` which makes handling of a
  force field as one object (that is aware of its parameters) easier and is the main
  class for passing a model between training and simulation.

The library currently interfaces three MLIP model architectures, i.e., MLIP network
implementations:

* `MACE <https://arxiv.org/abs/2206.07697>`_
  (class: :py:class:`Mace <mlip.models.mace.models.Mace>`),
* `NequIP <https://www.nature.com/articles/s41467-022-29939-5>`_
  (class: :py:class:`Nequip <mlip.models.nequip.models.Nequip>`), and
* `ViSNet <https://www.nature.com/articles/s41467-023-43720-2>`_
  (class: :py:class:`Visnet <mlip.models.visnet.models.Visnet>`).

These networks can be created from their configuration
(:py:class:`MaceConfig <mlip.models.mace.config.MaceConfig>`,
:py:class:`NequipConfig <mlip.models.nequip.config.NequipConfig>`, or
:py:class:`VisnetConfig <mlip.models.visnet.config.VisnetConfig>`) and a
:py:class:`DatasetInfo <mlip.data.dataset_info.DatasetInfo>` object
that one obtained after the :ref:`data processing step <get_dataset_info>`. For the
sake of simplified usage, the config objects can be directly accessed from the network
classes via their `.Config` attribute (see example below).

For example, to create a force field that uses MACE, one can simply execute:

.. code-block:: python

    from mlip.models import Mace, ForceField

    dataset_info = _get_from_data_processing()  # placeholder

    # with default config
    mace = Mace(Mace.Config(), dataset_info)
    force_field = ForceField.from_mlip_network(mace)

    # with modified config
    mace = Mace(Mace.Config(num_channels=64), dataset_info)
    force_field = ForceField.from_mlip_network(mace)

The :py:class:`ForceField <mlip.models.force_field.ForceField>` class stores the
parameters of the model (random parameters after initialization) and acts as the input
to all downstream tasks. However, it is also possible for advanced users to interact
with the underlying flax modules directly.
We recommend to visit the `flax documentation <https://flax.readthedocs.io/>`_
for more details on how to work with
`flax modules <https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html>`_.

Make predictions
----------------

We can run a prediction with an MLIP force field like this:

.. code-block:: python

    graph = _get_jraph_graph_from_somewhere()  # placeholder
    prediction = force_field(graph)

The ``prediction`` includes several properties and is a dataclass of type
:py:class:`Prediction <mlip.typing.prediction.Prediction>`. The properties other than
energy and forces are only predicted optionally
(see ``predict_stress`` argument of `ForceField.from_mlip_network`).

If the input ``graph`` object (type: ``jraph.GraphsTuple``) contains multiple subgraphs,
for example, if it represents a batch, we can get the energy and forces of the ``i``-th
subgraph like this:

.. code-block:: python

    # For i-th energy
    energy_i = float(prediction.energy[i])

    # For i-th forces
    num_nodes_before_i = sum(graph.n_node[j] for j in range(0, i))
    forces_i = prediction.forces[num_nodes_before_i : num_nodes_before_i + graph.n_node[i]]


.. _load_zip_model:

Load a model from a zip archive
-------------------------------

To load a model (e.g., MACE) from our lightweight zip format that we ship our
foundation models with, you can use the function
:py:func:`load_model_from_zip <mlip.models.model_io.load_model_from_zip>`:

.. code-block:: python

    from mlip.models import Mace
    from mlip.models.model_io import load_model_from_zip

    force_field = load_model_from_zip(Mace, "path/to/model.zip")

Subsequently, you can use the returned force field
(type: :py:class:`ForceField <mlip.models.force_field.ForceField>`) for
any downstream tasks.

.. _load_trained_model:

Load a trained model from an Orbax checkpoint
---------------------------------------------

To load a trained model from an `orbax <https://orbax.readthedocs.io/en/latest/>`_
checkpoint, one can use the
:py:func:`load_parameters_from_checkpoint() <mlip.models.params_loading.load_parameters_from_checkpoint>`
helper function:

.. code-block:: python

    from mlip.models import ForceField
    from mlip.models.params_loading import load_parameters_from_checkpoint

    initial_force_field = _create_initial_force_field()  # placeholder

    # Load parameters
    loaded_params = load_parameters_from_checkpoint(
        local_checkpoint_dir="path/to/checkpoint/directory",  # must be local
        initial_params=initial_force_field.params,
        epoch_to_load=157,
        load_ema_params=False,
    )

    # Create new force field with those loaded parameters
    force_field = ForceField(initial_force_field.predictor, loaded_params)
