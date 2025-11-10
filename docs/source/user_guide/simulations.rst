.. _simulations:

Simulations
===========

This library supports :ref:`two types of simulations <simulation_enums_type>`,
MD and energy minimizations, with
:ref:`two types of backends <simulation_enums_backend>`, JAX-MD and ASE. Simulations are
handled with simulation engine classes, which are implementations of the abstract
base class
:py:class:`SimulationEngine <mlip.simulation.simulation_engine.SimulationEngine>`.
One can either use our two implemented engines
(:py:class:`JaxMDSimulationEngine <mlip.simulation.jax_md.jax_md_simulation_engine.JaxMDSimulationEngine>`
and
:py:class:`ASESimulationEngine <mlip.simulation.ase.ase_simulation_engine.ASESimulationEngine>`),
or implemented custom ones. Each engine comes with its own pydantic config that
inherits from
:py:class:`SimulationConfig <mlip.simulation.configs.simulation_config.SimulationConfig>`.

**Important note on units**: The system of units for the inputs and outputs of all
simulation types is the
`ASE unit system <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.

**Important note on logging**: There is a subtle difference in which steps the JAX-MD
and ASE backends log. While both engines run for *n* steps, JAX-MD logs *N* snapshots,
the first of which corresponds to the initial (zero-th) state
and the last snapshot corresponds to the *N-1*-th logging step. In contrast,
ASE logs *N+1* snapshots, the first of which corresponds to the initial (zero-th) state
and the last snapshot corresponds to the *N*-th logging step.

Simulations with JAX-MD
-----------------------

To run a simulation (for example, an MD) with the JAX-MD backend, one can use the
following code:

.. code-block:: python

    from ase.io import read as ase_read
    from mlip.simulation.jax_md import JaxMDSimulationEngine

    atoms = ase_read("/path/to/xyz/or/pdb/file")
    force_field = _get_a_trained_force_field_from_somewhere()  # placeholder
    md_config = JaxMDSimulationEngine.Config(**config_kwargs)

    md_engine = JaxMDSimulationEngine(atoms, force_field, md_config)
    md_engine.run()

Note that in the example above, ``_get_a_trained_force_field_from_somewhere()`` is a
placeholder for a function that loads a trained force field, as described either
:ref:`here <load_zip_model>` (Option 1) or :ref:`here <load_trained_model>` (Option 2).
The config class for JAX-MD simulations is
:py:class:`JaxMDSimulationConfig <mlip.simulation.configs.jax_md_config.JaxMDSimulationConfig>`
and can also be accessed via `JaxMDSimulationEngine.Config` for the sake of needing
fewer imports. The format for the input structure is the commonly used ``ase.Atoms``
class (see the ASE docs `here <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_).

The result of the simulation is stored in the
:py:class:`SimulationState <mlip.simulation.state.SimulationState>`, which can
be accessed like this:

.. code-block:: python

    md_state = md_engine.state

    # Print some data from the simulation:
    print(md_state.positions)
    print(md_state.temperature)
    print(md_state.compute_time_seconds)

Also, we recommend that you take note of the units
of the computed properties as described in the
:py:class:`SimulationState <mlip.simulation.state.SimulationState>` reference. See
our Jupyter notebook on simulations :ref:`here <notebook_tutorials>` for
more information on how to convert these raw numpy arrays into file
formats that can be read by popular MD visualization tools.

Energy minimizations can be run in exactly the same way, possibly using slightly
different settings. See the documentation of the
:py:class:`JaxMDSimulationConfig <mlip.simulation.configs.jax_md_config.JaxMDSimulationConfig>`
class for more details. Most importantly, the `simulation_type` needs to be set to
`SimulationType.MINIMIZATION` (see
:py:class:`SimulationType <mlip.simulation.enums.SimulationType>`).

.. note::

    The default timestep of 1.0 fs that is common for MD simulations may not be optimal
    for energy minimizations. We recommend to set this value to 0.1 fs when using the
    `SimulationType.MINIMIZATION` mode with the JAX-MD backend.

**Algorithms**: For MD, the NVT-Langevin algorithm is used
(see `here <https://jax-md.readthedocs.io/en/main/jax_md.simulate.html#jax_md.simulate.nvt_langevin>`_).
For energy minimization, the FIRE algorithm is used
(see `here <https://jax-md.readthedocs.io/en/main/jax_md.minimize.html#jax_md.minimize.fire_descent>`_).
We plan to provide more options in future versions of the library.

Furthermore, for MD simulations, we support running them in a **batched manner**.
See :ref:`this <batched_simulations>` section below for more information.

.. note::

   A special feature of the JAX-MD backend is that a simulation is divided into
   multiple episodes. Within one episode, the simulation runs in a fully jitted way.
   After each episode, the neighbor lists can be reallocated, the simulation state can
   be populated and :ref:`loggers <advanced_logging_simulations>` can be called.

Simulations with ASE
--------------------

With ASE, running MD simulations and energy minimizations works in an analogous way
as described above. The following code can be used:

.. code-block:: python

    from ase.io import read as ase_read
    from mlip.simulation.ase.ase_simulation_engine import ASESimulationEngine

    atoms = ase_read("/path/to/xyz/or/pdb/file")
    force_field = _get_a_trained_force_field_from_somewhere()  # placeholder
    md_config = ASESimulationEngine.Config(**config_kwargs)

    md_engine = ASESimulationEngine(atoms, force_field, md_config)
    md_engine.run()

The config class for ASE simulations is
:py:class:`ASESimulationConfig <mlip.simulation.configs.ase_config.ASESimulationConfig>`
(accessible via `ASESimulationEngine.Config`).
As in the JAX-MD case, the format for the input structure is the ``ase.Atoms`` class
(see the ASE docs `here <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_).

The results of the simulation are stored in the
:py:class:`SimulationState <mlip.simulation.state.SimulationState>` object as
described in the JAX-MD case above. Also, we recommend that you take note of the units
of the computed properties as described in the
:py:class:`SimulationState <mlip.simulation.state.SimulationState>` reference.

For the settings required for energy minimizations, check out the documentation of the
:py:class:`ASESimulationConfig <mlip.simulation.configs.ase_config.ASESimulationConfig>`
class. Most importantly, the `simulation_type` needs to be set to
`SimulationType.MINIMIZATION` (see
:py:class:`SimulationType <mlip.simulation.enums.SimulationType>`).

**Algorithms**: For MD, the NVT-Langevin algorithm is used
(see `here <https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md.langevin>`_).
For energy minimization, the BFGS algorithm is used
(see `here <https://wiki.fysik.dtu.dk/ase/ase/optimize.html#ase.optimize.BFGS>`_).
We plan to provide more options in future versions of the library.

Temperature Scheduling
----------------------

It is also possible to add a temperature schedule to both simulation engines,
check out the documentation of the
:py:class:`TemperatureScheduleConfig <mlip.simulation.configs.simulation_config.TemperatureScheduleConfig>`
class for more details. This is done by creating an instance of
:py:class:`TemperatureScheduleConfig <mlip.simulation.configs.simulation_config.TemperatureScheduleConfig>`
and passing it under the variable name ``temperature_schedule_config`` to either
:py:class:`ASESimulationConfig <mlip.simulation.configs.ase_config.ASESimulationConfig>`
or :py:class:`JaxMDSimulationConfig <mlip.simulation.configs.jax_md_config.JaxMDSimulationConfig>`.
By default, the method is ``CONSTANT``, which means the target temperature is set at the
start of the simulation and kept constant throughout its entirety.
However, other methods are available: ``LINEAR`` and ``TRIANGLE``.
If you want to use a temperature schedule, you can set the ``method``
attribute to an instance of the
:py:class:`TemperatureScheduleMethod <mlip.simulation.enums.TemperatureScheduleMethod>`
class and ensure that any other required parameters for the different methods
have been set appropriately.
The temperature schedule methods
are described :ref:`here <temperature_scheduling>` for more information.

Below we provide an example of how to use a linear schedule
that will heat the system from 300 K to 600 K when using the JAX-MD simulation backend:

.. code-block:: python

    from mlip.simulation.configs import TemperatureScheduleConfig
    from mlip.simulation.jax_md import JaxMDSimulationEngine
    from mlip.simulation.enums import TemperatureScheduleMethod

    temp_schedule_config = TemperatureScheduleConfig(
        method=TemperatureScheduleMethod.LINEAR,
        start_temperature=300.0,
        end_temperature=600.0
    )
    md_config = JaxMDSimulationEngine.Config(
        temperature_schedule_config=temp_schedule_config,
        **config_kwargs
    )

    # Go on to initialize a simulation with this config


.. _advanced_logging_simulations:

Advanced logging
----------------

The :py:class:`SimulationEngine <mlip.simulation.simulation_engine.SimulationEngine>`
allows to attach custom loggers to a simulation:

.. code-block:: python

    from mlip.simulation.state import SimulationState

    def logging_fun(state: SimulationState) -> None:
        """You can do anything with the given state here"""
        _log_something()  # placeholder

    md_engine.attach_logger(logging_fun)

The logger must be attached before starting the simulation.
In ASE, this logging function will be called depending on the logging interval set,
and in JAX-MD, it will be called after every episode.

.. _batched_simulations:

Batched simulations with JAX-MD
-------------------------------

For MD simulations or energy minimizations with JAX-MD, we support running them in a
batched manner for multiple systems. The API for this is straightforward,
instead of passing a single `ase.Atoms` object to the engine, we pass a list of them.
After the simulation, the simulation state will contain lists of properties,
for example, a list of position arrays (i.e., the trajectories) instead of a single
position array. Note that it is also supported that the input molecules have
varying sizes. See example code below:

.. code-block:: python

    from ase.io import read as ase_read
    from mlip.simulation.jax_md import JaxMDSimulationEngine

    systems = []
    for path in ["/path/to/mol_1", "/path/to/mol_2", "/path/to/mol_3"]:
        atoms = ase_read(path)
        systems.append(atoms)

    force_field, md_config = _get_from_somewhere()  # placeholder
    md_engine = JaxMDSimulationEngine(systems, force_field, md_config)
    md_engine.run()

    # Fetch results:
    # Get trajectory and temperatures for "/path/to/mol_2" (indexing starts at 0)
    md_state = md_engine.state
    print(md_state.positions[1])
    print(md_state.temperature[1])

    # Compute time, for example, is not a list
    print(md_state.compute_time_seconds)

The example above works for both energy minimizations and MD simulations in the same
way.

.. _batched_inference:

Batched inference
-----------------

Instead of running MD simulations or energy minimizations,
we also provide the function
:py:func:`run_batched_inference() <mlip.inference.batched_inference.run_batched_inference>`
that allows to input a list of `ase.Atoms` objects and returns a list of
:py:class:`Prediction <mlip.typing.prediction.Prediction>` objects like this:

.. code-block:: python

    from mlip.inference import run_batched_inference

    structures = _get_list_of_ase_atoms_from_somewhere()  # placeholder
    force_field = _get_a_trained_force_field_from_somewhere()  # placeholder
    predictions = run_batched_inference(structures, force_field, batch_size=8)

    # Example: Get energy and forces for 7-th structure (indexing starts at 0)
    energy = predictions[7].energy
    forces = predictions[7].forces
