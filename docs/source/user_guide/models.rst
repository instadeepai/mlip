.. _models:

Models
======

.. _model_init:

Create a model and force field
--------------------------------

This section discusses how to initialize an MLIP model for subsequent training.
If you are just interested in loading a pre-trained model for application in simulations,
please see the dedicated section :ref:`below <load_zip_model>`.

Our MLIP models exist in two abstraction levels:

* On the one hand, we have the **pure neural networks**,
  which are classes derived from
  :py:class:`MLIPNetwork <mlip.models.mlip_network.MLIPNetwork>`. These models take in
  as input a :py:class:`Graph <mlip.graph.Graph>` and output a
  :py:class:`Graph <mlip.graph.Graph>`. In the networks implemented in the library, we
  populate at least the `"energy"` field in the `Graph.nodes.features` dictionary
  with the node energies, optionally additional fields for other property predictions,
  such as atomic partial charges. However, the `Graph`-to-`Graph` signature is designed
  in a general way so that a newly added
  :py:class:`MLIPNetwork <mlip.models.mlip_network.MLIPNetwork>` can decide to populate
  other fields, as long as the next abstraction level described below is adapted to
  handle this downstream. Note that many of the layers and blocks inside the
  networks are implemented with the `Graph`-to-`Graph` signature.

* On the other hand, we **wrap these models into force fields**
  which take care of computing properties such as total energy, forces, stress,
  Hessians, or atomic partial charges
  from the MLIP network's output. These also take a :py:class:`Graph <mlip.graph.Graph>`
  object as input and can output either an output :py:class:`Graph <mlip.graph.Graph>`
  or :py:class:`Prediction <mlip.typing.prediction.Prediction>` (more details later).
  The flax module that implements this is
  :py:class:`ForceFieldPredictor <mlip.models.predictors.predictor.ForceFieldPredictor>`
  (or actually, its derived classes), however,
  we recommend interacting with the class
  :py:class:`ForceField <mlip.models.force_field.ForceField>`, which makes handling a
  force field as one object (that is aware of its parameters) easier and is the main
  class for passing a model between training and simulation. More information on
  how the
  :py:class:`ForceFieldPredictor <mlip.models.predictors.predictor.ForceFieldPredictor>`
  classes work internally can be found in a dedicated section
  :ref:`below <predictor_details>`.

The library currently interfaces four MLIP model architectures, i.e., MLIP network
implementations:

* `MACE <https://arxiv.org/abs/2206.07697>`_
  (class: :py:class:`Mace <mlip.models.mace.network.Mace>`),
* `NequIP <https://www.nature.com/articles/s41467-022-29939-5>`_
  (class: :py:class:`Nequip <mlip.models.nequip.network.Nequip>`),
* `ViSNet <https://www.nature.com/articles/s41467-023-43720-2>`_
  (class: :py:class:`Visnet <mlip.models.visnet.network.Visnet>`), and
* `eSEN <https://arxiv.org/abs/2502.12147>`_
  (class: :py:class:`Esen <mlip.models.esen.network.Esen>`).

These networks can be created from their configuration
(:py:class:`MaceConfig <mlip.models.mace.config.MaceConfig>`,
:py:class:`NequipConfig <mlip.models.nequip.config.NequipConfig>`,
:py:class:`VisnetConfig <mlip.models.visnet.config.VisnetConfig>`, or
:py:class:`EsenConfig <mlip.models.esen.config.EsenConfig>`) and a
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

    graph = _get_graph_from_somewhere()  # placeholder

    # Option 1: output a prediction
    prediction = force_field(graph)

    # Option 2: output a prediction graph
    output_graph = force_field.calculate(graph)

For option 1, the ``prediction`` includes several properties and is a dataclass of type
:py:class:`Prediction <mlip.typing.prediction.Prediction>`.

Which properties are predicted depends on the ones requested via the
`required_properties` attribute of the
:py:class:`ForceFieldPredictor <mlip.models.predictors.predictor.ForceFieldPredictor>`.
By default, this includes energies and forces, but additional required properties
can be passed either when creating a force field via the
:py:meth:`ForceField.from_mlip_network <mlip.models.force_field.ForceField.from_mlip_network>`
method or when loading an already trained force field
(see :ref:`below <load_zip_model>`). Required properties are passed and stored as
a :py:class:`Properties <mlip.typing.properties.Properties>` dataclass.

**Important caveat:** For Hessian matrix predictions, it is *not* sufficient to
set `Properties(hessian=True)` for the required properties, but additionally, one must
call :py:meth:`Graph.request_full_hessian <mlip.graph.Graph.request_full_hessian>` to
obtain an updated graph before running a prediction on it. This only applies when
predicting on a graph directly, it is not applicable to the training workflow, and is
handled automatically when running :ref:`batched inference <batched_inference>`. See
the `Hessian tutorial notebook <https://github.com/instadeepai/mlip/blob/main/tutorials/hessian_model_training_tutorial.ipynb>`_
for an explicit example.

If the input `Graph` object contains multiple subgraphs,
for example, if it represents a batch, we can get the energy and forces of the `i`-th
subgraph like this:

.. code-block:: python

    # For i-th energy
    energy_i = float(prediction.energy[i])

    # For i-th forces
    num_nodes_before_i = sum(graph.n_node[j] for j in range(0, i))
    forces_i = prediction.forces[num_nodes_before_i : num_nodes_before_i + graph.n_node[i]]

In option 2, the `calculate()` method yields a prediction `Graph` that stores the
resulting properties in its attributes. Note that a `Prediction` can be created from
a `Graph` easily via
:py:meth:`Graph.to_prediction <mlip.graph.Graph.to_prediction>`.

**Easiest way to create a single input graph from an XYZ file:**

The following example demonstrates how to create a simple `Graph` object for a molecule
stored in the common XYZ file format:

.. code-block:: python

    import ase.io
    from mlip.data import ChemicalSystem
    from mlip.graph import Graph

    molecule = ase.io.read("/path/to/molecule.xyz")

    chem_system = ChemicalSystem.from_ase_atoms(molecule)
    graph = Graph.from_chemical_system(chem_system, graph_cutoff_angstrom=5.0)

.. _load_zip_model:

Load a model from a zip archive
-------------------------------

To load a model (e.g., MACE) from our lightweight zip format that we ship our
pre-trained models with, you can use the function
:py:func:`load_model_from_zip <mlip.models.model_io.load_model_from_zip>`:

.. code-block:: python

    from mlip.models import Mace
    from mlip.models.model_io import load_model_from_zip

    force_field = load_model_from_zip(Mace, "path/to/model.zip")

The required properties can also be passed to
:py:func:`load_model_from_zip <mlip.models.model_io.load_model_from_zip>` as
a :py:class:`Properties <mlip.typing.properties.Properties>` dataclass. Note that
by default, the required properties are only energy and forces.

If the model needs graph-level metadata during inference, pass an
:py:class:`InferenceContext <mlip.models.inference_context.InferenceContext>` while
loading. The returned force field stores the resolved context. For
Mixture-of-Experts (MoE) models, the
loader also contracts experts for that fixed context. See example code below:

.. code-block:: python

    from mlip.models import InferenceContext, Mace
    from mlip.models.model_io import load_model_from_zip

    force_field = load_model_from_zip(
        Mace,
        "path/to/model.zip",
        inference_context=InferenceContext(dataset_name="organics"),
    )

Subsequently, you can use the returned force field
(type: :py:class:`ForceField <mlip.models.force_field.ForceField>`) for
any downstream tasks (e.g., MD simulations or batched inference).

.. _load_trained_model:

Load a trained model from an Orbax checkpoint
---------------------------------------------

To load a trained model from an `Orbax <https://orbax.readthedocs.io/en/latest/>`_
checkpoint, one can use the
:py:func:`load_parameters_from_checkpoint() <mlip.models.params_loading.load_parameters_from_checkpoint>`
function:

.. code-block:: python

    from mlip.models import ForceField
    from mlip.models.params_loading import load_parameters_from_checkpoint

    initial_force_field = _create_initial_force_field()  # placeholder

    # Load parameters
    loaded_params = load_parameters_from_checkpoint(
        checkpoint_dir="path/to/checkpoint/directory",
        initial_params=initial_force_field.params,
        epoch_to_load=157,
        load_ema_params=False,
    )

    # Create new force field with those loaded parameters
    force_field = ForceField(initial_force_field.predictor, loaded_params)

In the final line of the example above, it is assumed that the
:py:class:`InferenceContext <mlip.models.inference_context.InferenceContext>` is
`None`.

.. _predictor_details:

Details on `ForceFieldPredictor`
--------------------------------

This section reports additional details on the design of the
:py:class:`ForceFieldPredictor <mlip.models.predictors.predictor.ForceFieldPredictor>`
class and its derived classes. While it is not necessary to understand the design as
an applied user that interacts mostly with the
:py:class:`ForceField <mlip.models.force_field.ForceField>` directly, it can still be
useful, and furthermore, it is absolutely crucial to understand for users that aim to
develop new models and plan to implement their own derived predictor classes.

The purpose of the
:py:class:`ForceFieldPredictor <mlip.models.predictors.predictor.ForceFieldPredictor>`
is to convert the raw output of a
:py:class:`MLIPNetwork <mlip.models.mlip_network.MLIPNetwork>` to a prediction
:py:class:`Graph <mlip.graph.Graph>` that contains all required properties in the
intended fields, for example, forces in `Graph.nodes.forces` and energy in
`Graph.globals.energy`. This conversion typically contains two parts, (1)
**differentiation** and (2) **applying an energy head** that may compute the final graph
energy differently based on the raw node energies (e.g., with or without long-range
interactions).

While the base
:py:class:`ForceFieldPredictor <mlip.models.predictors.predictor.ForceFieldPredictor>`
class contains a `compute_energy` implementation (it calls the MLIP network, applies
the energy head, and evaluates the final energy and optionally partial charges),
it delegates the implementation of
its `compute_forces_and_stress` method, which remains abstract because it is one of the
two custom behaviors of a predictor implementation (see (1) above). For (2), the
energy head is an attribute of the predictor
(`energy_head: Callable[[Graph], Array] | None = None`) with `None` resulting in a
default energy head. The energy head is a unit that takes in a `Graph` and returns
an energy array (i.e., total energy for each graph in a potentially batched graph).

The current state of the library contains **two predictor implementations**:

* The :py:class:`ConservativePredictor <mlip.models.predictors.conservative_predictor.ConservativePredictor>`
  that computes energies, forces, and stress conservatively, i.e., the forces are
  the derivative of the energy with respect to the atomic coordinates. The name hints
  at the fact that future versions of the library are planned to include a
  direct force predictor.

* The :py:class:`HessianPredictor <mlip.models.predictors.hessian_predictor.HessianPredictor>`
  that, in addition to energies, forces, and stress, also computes the Hessian matrix
  (or, during training, a subset of rows subsampled from the Hessian).

Two energy heads are included in *mlip* currently: the
**standard energy head** and a **long-range interaction variant** that applies an
additional Coulomb potential.

**Important:** When interacting with our models via the
:py:class:`ForceField <mlip.models.force_field.ForceField>` class, as is typically
the case, the force field automatically takes care of selecting the correct predictor
class and energy head that matches the required properties and MLIP network config.

However, user-defined energy heads can be useful, for instance, for adding surrogate
potentials to a force field. A custom force field with a custom energy head can easily
be defined via inheritance by overriding the `get_energy_head` method, like this:

.. code-block:: python

    from mlip.models import ForceField

    def custom_energy_head(graph):
        # custom energy head implementation

    class CustomForceField(ForceField):

        @classmethod
        def get_energy_head(cls, *args, **kwargs):
            return custom_energy_head
