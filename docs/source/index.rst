.. MLIP documentation master file:
   You can adapt this file completely to your liking,
   but it should at least
   contain the root `toctree` directive.

Welcome to the documentation of MLIP!
=====================================

*mlip* is a Python library for **Machine Learning Interatomic Potentials (MLIP)**
in JAX. It contains the following features:

* Multiple model architectures (for now: MACE, NequIP and ViSNet)
* Dataset preprocessing
* Training of MLIP models
* Batched inference with trained MLIP models
* MD simulations with MLIP models using multiple simulation backends
* Energy minimizations with MLIP models using multiple simulation backends
* Batched MD simulations and energy minimizations with the JAX-MD simulation backend
* Fine-tuning of pre-trained MLIP models

As a first step, we recommend that you check out our page on :ref:`installation`
and our :ref:`user_guide` which contains several tutorials on how to use the library.
Furthermore, we also provide an :ref:`api_reference`.

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation <installation/index>
   User guide <user_guide/index>
   API reference <api_reference/index>
