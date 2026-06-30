.. _migration_guide:

Migration guide: v1 to v2
=========================

This migration guide aims at

* describing the :ref:`API changes <migration_api>` of *mlip* introduced as part
  of the v0.2.0 release, and
* explaining how to use the :ref:`new features <migration_new_feat>` of *mlip* v2.

While the former is crucial for understanding how to update existing *mlip* v1 code
to achieve the same functionality with v2, we recommend to check out the latter as
well to familiarize yourself with the new recently added capabilities of *mlip*.

.. _migration_api:

API updates
-----------

This section explains API updates for :ref:`data processing <migration_data>`,
:ref:`models <migration_models>`, :ref:`training <migration_training>`,
and :ref:`simulations <migration_simulations>`.

.. _migration_data:

1. Data processing
^^^^^^^^^^^^^^^^^^

The main update regarding the training of models concerns data processing. While in v1,
a single reader class covers all splits (train, validation, and test), in v2, the
:py:class:`ChemicalSystemsReader <mlip.data.chemical_systems_readers.chemical_systems_reader.ChemicalSystemsReader>`
classes only handle one split at a time. As a result, we do *not* provide a single
reader to the
:py:class:`GraphDatasetBuilder <mlip.data.graph_dataset_builder.GraphDatasetBuilder>`
anymore, but instead a dictionary of readers. Moreover, the reader class does not have a
pydantic config anymore and just receives its few arguments directly in its constructor.
Lastly, executing the graph building process is now a one-step process by calling
`get_datasets()` on the builder while before two separate functions had to be called.
The returned splits are not a tuple anymore in v2, but instead a dictionary with the
same keys as in the provided dictionary `readers`.
Code example with comparison is shown below:

.. code-block:: python

    from mlip.data import GraphDatasetBuilder, BuilderMode, ExtxyzReader

    # In v1: just one reader with a config
    # ------------------------------------
    # reader_config = ExtxyzReader.Config(
    #     train_dataset_paths = "/path/to/train.xyz",
    #     valid_dataset_paths = "/path/to/validation.xyz",
    #     test_dataset_paths = "/path/to/test.xyz",
    # )
    # reader = ExtxyzReader(reader_config)

    # In v2:
    readers = {
        "train": ExtxyzReader("/path/to/train.xyz"),
        "validation": ExtxyzReader("/path/to/validation.xyz"),
        "test": ExtxyzReader("/path/to/test.xyz"),
    }

    # For the builder initialization, nothing changes
    builder_config = GraphDatasetBuilder.Config(
        graph_cutoff_angstrom=5.0,
        batch_size=32,
    )
    graph_dataset_builder = GraphDatasetBuilder(
        readers,
        builder_config,
    )

    # Calling the builder is different
    # --------------------------------
    # In v1:
    # graph_dataset_builder.prepare_datasets()
    # splits = graph_dataset_builder.get_splits()
    # train_set, validation_set, test_set = splits

    # In v2:
    splits = graph_dataset_builder.get_datasets()
    train_set, validation_set, test_set = (
        splits["train"], splits["validation"], splits["test"]
    )

.. _migration_models:

2. Models and the `Graph` object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When initializing a force field, either via the `ForceField.from_mlip_network` function
or loading it from disk via
:py:func:`load_model_from_zip <mlip.models.model_io.load_model_from_zip>`, one now
passes the required properties that the model should output via an instance of the
:py:class:`Properties <mlip.typing.properties.Properties>` dataclass. This replaces the
`predict_stress` flag in v1. See :ref:`this code block <code_block_req_properties>`
for an example.

Note that while the `__call__` function of the force field still outputs a
:py:class:`Prediction <mlip.typing.prediction.Prediction>` class, in *mlip* v2, the
force field has an additional `calculate()` method that returns an output graph
(type: :py:class:`Graph <mlip.graph.Graph>`) instead.

The class :py:class:`Graph <mlip.graph.Graph>` is new in v2 and replaces the previous
`jraph.GraphsTuple` object as the representation for a graph. As an applied user,
this change is most notable if one wants to run inference on a single system. The
difference between v1 and v2 is demonstrated in the example below:

.. code-block:: python

    import ase.io
    from mlip.data import ChemicalSystem
    from mlip.graph import Graph

    # In v1: manually set up chemical system;
    # then convert to graph (that includes dummy)
    # -------------------------------------------
    # system = ChemicalSystem(
    #     atomic_numbers = np.array([1, 8, 1]),
    #     atomic_species = np.array([0, 3, 0]),
    #     positions = np.array([[-.5, .0, .0], [.0, .2, .0], [.5, .0, .0]]),
    # )
    #
    # graph = create_graph_from_chemical_system(
    #     chemical_system=system,
    #     distance_cutoff_angstrom=5.0,
    #     batch_it_with_minimal_dummy=True,
    # )

    # In v2:
    molecule = ase.io.read("/path/to/molecule.xyz")
    chem_system = ChemicalSystem.from_ase_atoms(molecule)
    graph = Graph.from_chemical_system(chem_system, graph_cutoff_angstrom=5.0)

Note that above, there is no requirement anymore to batch with a dummy graph for
single-system inference.

The concept of atomic species, i.e., indices
starting at zero for each element type that a model can handle, has been removed
entirely from the data processing pipelines and is now handled inside the models
via the :py:class:`SpeciesAssignmentBlock <mlip.models.blocks.SpeciesAssignmentBlock>`.

**For developers of new models,**
we point out that the model implementations have been fully refactored to
use a `Graph -> Graph` signature for the
:py:class:`MLIPNetwork <mlip.models.mlip_network.MLIPNetwork>` class and many layers
and blocks used by the models. We refer to the
`model addition tutorial <https://github.com/instadeepai/mlip/blob/main/tutorials/model_addition_tutorial.ipynb>`_
and the source code (module: `models`) for details.

**Backwards compatibility for models trained with v1:**
Due to the extensive refactoring inside the models and the model interface, models
trained with v1 cannot be used for inference with the v2 model implementations.
However, we include a `models_v1` module with the old implementations and have
integrated these with the v2
:py:class:`ForceFieldPredictor <mlip.models.predictors.predictor.ForceFieldPredictor>`
API, such that v1-trained models are still usable with *mlip* v2, for example,
by loading a v1 zip like this for MACE:

.. code-block:: python

    from mlip.models_v1 import Mace as MaceV1
    from mlip.models.model_io import load_model_from_zip

    force_field = load_model_from_zip(
        MaceV1,
        "path/to/model_mace_v1.zip",
    )

However, note that support for this is planned to be terminated
eventually with upcoming *mlip* versions.

.. _migration_training:

3. Loss and training
^^^^^^^^^^^^^^^^^^^^

The API for the :py:class:`TrainingLoop <mlip.training.training_loop.TrainingLoop>`
remains mostly stable between v1 and v2. We discuss a subtle difference in the
following. While when using the standard losses
:py:class:`MSELoss <mlip.models.loss.MSELoss>` or
:py:class:`HuberLoss <mlip.models.loss.HuberLoss>`, the imports and code looks exactly
the same in v2 compared to v1, under the hood, the
:py:class:`Loss <mlip.models.loss.Loss>` interface has changed: its `__call__` method
does not receive as arguments an reference graph and a model-generated
:py:class:`Prediction <mlip.typing.prediction.Prediction>` anymore, but instead
takes in two graphs, one for the reference and one for the prediction (output graph of
:py:class:`ForceFieldPredictor <mlip.models.predictors.predictor.ForceFieldPredictor>`).
While this change is transparent to users utilizing the library's built-in loss
classes, those implementing custom losses must update
their code to ensure compatibility.

In v2, customizing your losses and adding new ones is made more convenient compared to
v1, which is explained in the :ref:`loss section <training_loss>` of the training user
guide.

.. _migration_simulations:

4. Simulations
^^^^^^^^^^^^^^

For running simulations with the JAX-MD or ASE backends, users should not expect any
breaking API changes. However, note that implementation details might have changed,
which can be relevant for advanced users who have implemented their own derived classes
of our :py:class:`SimulationEngine <mlip.simulation.simulation_engine.SimulationEngine>`.


.. _migration_new_feat:

New features
------------

Below, please find a list of new features introduced with *mlip* v2 with either a brief
explanation of how they can be used or a link to the relevant resource in this
documentation.

* The models MACE and NequIP are now powered by our **open-source library for
  equivariant operations** `e3j <https://github.com/instadeepai/e3j>`_, significantly
  accelerating their inference compared to *mlip* v1.

* The **eSEN model architecture**, see :py:class:`Esen <mlip.models.esen.network.Esen>`.
  Optionally, with a **Mixture-of-Experts (MoE) formalism**, requested via the `moe`
  field in the
  :py:class:`EsenConfig <mlip.models.esen.config.EsenConfig>`. Details, especially
  regarding how to prepare an MoE model for efficient inference, can be found in the
  `MoE training and inference tutorial notebook <https://github.com/instadeepai/mlip/blob/main/tutorials/moe_training_and_inference_tutorial.ipynb>`_.

* MACE now supports the use of **Gaunt tensor products (GTP)**. GTP can be used as a
  replacement of Clebsch-Gordan tensor products in the message passing block by setting
  `use_gaunt_tp_message_passing = True` in the
  :py:class:`MaceConfig <mlip.models.mace.config.MaceConfig>`. They can also be
  used as a backend for the symmetric contraction block by setting
  `symmetric_contraction_backend = "gaunt_tp"`. Note that our implementation uses a
  direct estimation of the integrals defining the GTP through spherical designs.

* For all architectures: **global charge conditioning, partial charge predictions, and
  long-range interactions** via a PhysNet-inspired Coulomb term. These features are
  available in each model config, e.g., see
  :py:class:`VisnetConfig <mlip.models.visnet.config.VisnetConfig>` (fields:
  `use_total_charge_embedding`, `predict_partial_charges`, and `use_coulomb_term`). Note
  that at inference time, partial charges must be requested via the
  `required_properties` argument upon force field initialization.

* **Transition state search** with the
  :py:class:`NEBSimulationEngine <mlip.simulation.ts_search.neb_simulation_engine.NEBSimulationEngine>`,
  see :ref:`here <neb_ts_search>` for its documentation.

* **NPT ensemble MD simulations**, requested via the new `md_integrator` field of the
  :py:class:`SimulationConfig <mlip.simulation.configs.simulation_config.SimulationConfig>`.
  It is implemented for both the JAX-MD and ASE backends. See our
  `tutorial notebook on advanced simulation <https://github.com/instadeepai/mlip/blob/main/tutorials/advanced_simulation_tutorial.ipynb>`_
  for an in-depth introduction.

* **Training on Hessian labels** and prediction Hessians at inference, see the
  `Hessian tutorial notebook <https://github.com/instadeepai/mlip/blob/main/tutorials/hessian_model_training_tutorial.ipynb>`_
  and :ref:`this <hessian_training>` section of the documentation for more information.

* **Multi-head fine-tuning**, see :ref:`this <model_finetuning>` dedicated user guide
  for more information.

In addition to these essential new features, there are many small feature additions
to the code base in *mlip* v2 that we believe will improve the overall user experience.
We encourage everyone to check out the relevant guides and API reference for the
functionalities they are using to familiarize themselves with the updated library.
