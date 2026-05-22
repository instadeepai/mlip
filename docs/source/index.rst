.. raw:: html

   <div style="text-align:center; margin-top: 1.5rem; margin-bottom: 2rem;">
     <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
       MLIP
     </h1>
   </div>

================================================================================

*mlip* is a Python library for building, training, and deploying
machine learning interatomic potentials in JAX.

It provides tools for:

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 🧠 Models
      :link: user_guide/models
      :link-type: doc

      Multiple architectures including **MACE**, **NequIP**, **ViSNet** and **eSEN**;
      built in a modular way that makes building new models easy

   .. grid-item-card:: 📦 Data
      :link: user_guide/data_processing
      :link-type: doc

      Highly customizable dataset preprocessing for training and inference

   .. grid-item-card:: 🚀 Training
      :link: user_guide/training
      :link-type: doc

      Train or fine-tune MLIP models, on a single or multiple accelerators in parallel
      (even scalable across hosts)

   .. grid-item-card:: ⚛️ Simulations
      :link: user_guide/simulations
      :link-type: doc

      Molecular dynamics, energy minimization, and transition state search with
      multiple backends

   .. grid-item-card:: ⚡ State-of-the-art speed

      Ultra-fast (batched) inference and MD simulations enabled by JAX-MD backend

   .. grid-item-card:: 🧪 Advanced descriptors

      Global charge conditioning, treatment of long-range interactions, and training
      on Hessian labels

--------------------------------------------------------------------------------

Getting Started
---------------

If you're new to *mlip*, we recommend starting here:

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: ⚙️ Installation
      :link: installation/index
      :link-type: doc

      Set up *mlip* and dependencies.

   .. grid-item-card:: 📘 User Guide
      :link: user_guide/index
      :link-type: doc

      Tutorials and practical workflows.

   .. grid-item-card:: 🧩 API Reference
      :link: api_reference/index
      :link-type: doc

      Detailed API documentation.

   .. grid-item-card:: 🔀 Migration Guide
      :link: migration_guide/index
      :link-type: doc

      Upgrade from v1 to v2.

.. note::

   The *mlip* library is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation <installation/index>
   User guide <user_guide/index>
   Migration guide: v1 to v2 <migration_guide/index>
   API reference <api_reference/index>
