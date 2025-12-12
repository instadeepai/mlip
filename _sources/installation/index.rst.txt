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

    pip install "mlip[cuda]"

**This command installs the CUDA 12 version of JAX.** For different versions, please
install *mlip* without the `cuda` flag and install the desired JAX version via pip.

Note that using the TPU version of JAX is, in principle, also supported by
this library. You need to install it separately via pip. However, it has not been
thoroughly tested and should therefore be considered an experimental feature.
