.. _installation:

Installation
============

The *mlip* library can be installed via pip:

.. code-block:: bash

    pip install mlip

However, this command **only installs the regular CPU version** of JAX.
We recommend that the library is run on GPU.
Use this command instead to install the GPU-compatible version:

.. code-block:: bash

    pip install "mlip[cuda13]"

**This command installs the CUDA 13 version of JAX.**

Note that the alias `cuda` can be used instead of `cuda13` for backwards compatibility
with *mlip* v1.

To install the CUDA 12 version instead, run:

.. code-block:: bash

    pip install "mlip[cuda12]"

For any other custom versions, please
install *mlip* without the `cuda` flag and install the desired JAX version via pip.

We also support installation of the TPU version of JAX via this command:

.. code-block:: bash

    pip install "mlip[tpu]"

Note that using the TPU version of JAX for *mlip* has not been tested
with the same thoroughness as the GPU version, and should therefore be considered
an experimental feature.
