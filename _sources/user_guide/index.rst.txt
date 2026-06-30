.. _user_guide:

User guide
==========

.. _getting_started:

Getting started
---------------

The *mlip* library consists of multiple submodules targeted towards different
parts of a complete MLIP pipeline:

* ``graph``: Code related to the :py:class:`Graph <mlip.graph.Graph>` object, central
  to all parts of the *mlip* code base, as it defines the core data structure for
  inputs and outputs of MLIP neural networks, and to some extent, the network
  layers and blocks.

* ``data``: Code related to dataset reading and preprocessing. Its main purpose is
  to go from datasets stored on a file system to instances of
  :py:class:`GraphDataset <mlip.data.graph_dataset.GraphDataset>`
  that can be directly used for training or batched inference tasks.

* ``models``: Code related to the MLIP models, which can be wrapped as
  :py:class:`ForceField <mlip.models.ForceField>` objects for easy interfacing with
  other ``mlip`` submodules like ``training`` or ``simulation``.
  This module also contains the loss definition and utilities for
  parameter loading of trained models.

* ``models_v1``: Code related to the legacy v1 MLIP model implementations, which can
  be used via the v2 :py:class:`ForceField <mlip.models.ForceField>`
  interface. These are still included in the library such that models trained with
  the v1 version of this library remain usable, however, support for this is
  planned to be terminated eventually with upcoming *mlip* versions.

* ``training``: Code related to training MLIP models. The main class for this task
  is the :py:class:`TrainingLoop <mlip.training.training_loop.TrainingLoop>`.

* ``simulation``: Code related to running MD simulations, energy minimizations, or
  nudged elastic band transition state searches with
  MLIP models. We support the `JAX-MD <https://jax-md.readthedocs.io/>`_
  and `ASE <https://wiki.fysik.dtu.dk/ase/>`_ backends.

* ``inference``: Code related to running batched inference.
  See :ref:`this <batched_inference>` section for more information.

* ``utils`` and ``typing``: Utility functions and type definitions
  used in different modules.

**Each of these modules is designed to allow a user to set up their own experiment
scripts or notebooks with minimal effort, while also supporting customization,
especially for topics such as logging (e.g., to a remote storage location
like S3 or GCS) or adding new losses, MLIP model architectures, or dataset
readers for customized data preprocessing.**

We provide some example Jupyter notebooks as tutorials
to help you with the onboarding process to the library.
Furthermore, we provide in-depth tutorials for each of the four main modules of the
library, along with some other more advanced topics.

If you have been a user of *mlip* prior to the v0.2.0 release, check out
our detailed :ref:`migration guide <migration_guide>` for breaking API changes
introduced with v0.2.0.

.. _notebook_tutorials:

Jupyter Notebook Tutorials
--------------------------

We provide Jupyter notebooks with example code that may serve as templates
to build your own more complex MLIP pipelines. These can be used alongside the
deep-dive tutorials :ref:`below <tutorials>` to help you with getting onboarded to the
*mlip* library. These tutorials can be found in the GitHub repository:

* `Inference and simulation <https://github.com/instadeepai/mlip/blob/main/tutorials/simulation_tutorial.ipynb>`_
* `Model training <https://github.com/instadeepai/mlip/blob/main/tutorials/model_training_tutorial.ipynb>`_
* `MoE training and inference <https://github.com/instadeepai/mlip/blob/main/tutorials/moe_training_and_inference_tutorial.ipynb>`_
* `Addition of new models <https://github.com/instadeepai/mlip/blob/main/tutorials/model_addition_tutorial.ipynb>`_
* `Training on Hessian labels <https://github.com/instadeepai/mlip/blob/main/tutorials/hessian_model_training_tutorial.ipynb>`_
* `Advanced simulation <https://github.com/instadeepai/mlip/blob/main/tutorials/advanced_simulation_tutorial.ipynb>`_
* `Metadynamics <https://github.com/instadeepai/mlip/blob/main/tutorials/metadynamics_tutorial.ipynb>`_

To run the tutorials, install Jupyter notebooks via pip and launch it from
a directory that contains the notebooks:

.. code-block:: bash

    pip install notebook && jupyter notebook

The installation of *mlip* itself is included within the notebooks. We recommend
running these notebooks with GPU acceleration enabled.

.. _tutorials:

Deep-dive Tutorials
-------------------

Follow the links below for more in-depth tutorials for each of the available
tasks supported by the library.

.. toctree::
   :maxdepth: 1

   data_processing
   models
   training
   inference
   simulations
   finetuning
