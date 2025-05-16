.. _installation:

Installation
============

The *mlip* library can be installed via pip:

.. code-block:: bash

    pip install mlip

However, this command **only installs the regular CPU version** of JAX.
We recommend that the library is run on GPU.
This requires also installing the necessary versions
of `jaxlib <https://pypi.org/project/jaxlib/>`_ which can also be installed via pip. See
the `installation guide of JAX <https://docs.jax.dev/en/latest/installation.html>`_ for
more information.
At time of release, the following install command is supported:

.. code-block:: bash

    pip install -U "jax[cuda12]"

Note that using the TPU version of *jaxlib* is, in principle, also supported by
this library. However, it has not been thoroughly tested and should therefore be
considered an experimental feature.

Also, some tasks in *mlip* will
require `JAX-MD <https://github.com/jax-md/jax-md>`_ as a dependency. As the newest
version of JAX-MD is not available on PyPI yet, this dependency will not
be shipped with *mlip* automatically and instead must be installed
directly from the GitHub repository, like this:

.. code-block:: bash

    pip install git+https://github.com/jax-md/jax-md.git
