.. _simulations:

Simulations
===========

This library supports :ref:`three types of simulations <simulation_enums_type>`,

* MD (NVT and NPT ensemble),
* energy minimizations, and
* transition state searches,

with :ref:`two types of backends <simulation_enums_backend>`, JAX-MD and ASE.

**MD and energy minimization:**
Simulations are handled with simulation engine classes, which are implementations
of the abstract base class
:py:class:`SimulationEngine <mlip.simulation.simulation_engine.SimulationEngine>`.
One can either use our two implemented engines
(:py:class:`JaxMDSimulationEngine <mlip.simulation.jax_md.jax_md_simulation_engine.JaxMDSimulationEngine>`
and
:py:class:`ASESimulationEngine <mlip.simulation.ase.ase_simulation_engine.ASESimulationEngine>`),
or implement custom ones. Each engine comes with its own pydantic config that
inherits from
:py:class:`SimulationConfig <mlip.simulation.configs.simulation_config.SimulationConfig>`.

**Transition state search:**
While the two classes mentioned above handle MD simulations and energy minimizations,
the class
:py:class:`NEBSimulationEngine <mlip.simulation.ts_search.neb_simulation_engine.NEBSimulationEngine>`
handles transition state searches via the nudged elastic band (NEB) method.
Hence, the NEB functionality is documented separately in :ref:`this <neb_ts_search>`
section below.

A few important notes
---------------------

**On units:** The system of units for the inputs and outputs of all
simulation types is the
`ASE unit system <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.

**On logging:** There is a subtle difference in which steps the JAX-MD
and ASE backends log. While both engines run for *n* steps, JAX-MD logs *N* snapshots,
the first of which corresponds to the initial (zero-th) state
and the last snapshot corresponds to the *N-1*-th logging step. In contrast,
ASE logs *N+1* snapshots, the first of which corresponds to the initial (zero-th) state
and the last snapshot corresponds to the *N*-th logging step.

**On early stopping:** If a simulation is unstable, it may "explode",
meaning that its temperature becomes ``nan`` or larger than ``1e6``.
In this case, the simulation will be stopped early, and the
simulation state will be logged before exiting. For the ASE backend, the simulation
stops immediately, for JAX-MD, after the current episode.

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
class (see the ASE docs `here <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`__).

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

Energy minimizations can be run in exactly the same way, using slightly
different settings. See the documentation of the
:py:class:`JaxMDSimulationConfig <mlip.simulation.configs.jax_md_config.JaxMDSimulationConfig>`
class for more details. Most importantly, the `simulation_type` needs to be set to
`SimulationType.MINIMIZATION` (see
:py:class:`SimulationType <mlip.simulation.enums.SimulationType>`).

.. note::

    The default timestep of 1.0 fs that is common for MD simulations may not be optimal
    for energy minimizations. We recommend to set this value to 0.1 fs when using the
    `SimulationType.MINIMIZATION` mode with the JAX-MD backend.

**Algorithms**: For energy minimization, the FIRE algorithm is used
(see `here <https://jax-md.readthedocs.io/en/main/jax_md.minimize.html#jax_md.minimize.fire_descent>`__).
We plan to provide more options in future versions of the library.
For MD, the integrator/ensemble can be set via the `md_integrator` attribute
(see :py:class:`MDIntegrator <mlip.simulation.enums.MDIntegrator>`),
to use either the NVT-Langevin algorithm
(see `here <https://jax-md.readthedocs.io/en/main/jax_md.simulate.html#jax_md.simulate.nvt_langevin>`__)
or the NPT-MC-Langevin algorithm, which uses Langevin dynamics with a Monte-Carlo Barostat
(see `here <https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#montecarlobarostat>`__).

For more information on NPT simulations, in particular, we refer to our
`advanced simulation tutorial notebook <https://github.com/instadeepai/mlip/blob/main/tutorials/advanced_simulation_tutorial.ipynb>`_.

For MD simulations, we support running them in a **batched manner**.
See :ref:`this <batched_simulations>` section below for more information.

.. note::

   A special feature of the JAX-MD backend is that a simulation is divided into
   multiple episodes. Within one episode, the simulation runs in a fully jitted way.
   After each episode, the neighbor lists can be reallocated, the simulation state can
   be populated and :ref:`loggers <advanced_logging_simulations>` can be called.

.. _simulations_ase_user_guide:

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
(see the ASE docs `here <https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`__).

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

**Algorithms**: For energy minimization, the BFGS algorithm is used
(see `here <https://wiki.fysik.dtu.dk/ase/ase/optimize.html#ase.optimize.BFGS>`__).
For MD, the integrator/ensemble can be set via the `md_integrator` attribute
(see :py:class:`MDIntegrator <mlip.simulation.enums.MDIntegrator>`),
to use either the NVT-Langevin algorithm
(see `here <https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md.langevin>`__).
or the NPT-MC-Langevin algorithm, which uses Langevin dynamics with a Monte-Carlo Barostat
(see `here <https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html#montecarlobarostat>`__).

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

With JAX-MD, we support running NVT-Langevin and NPT-MC-Langevin MD simulations
as well as energy minimizations in a batched manner for multiple systems.
The API for this is straightforward,
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

The example above works for both energy minimizations and NVT-Langevin MD simulations in the same
way.

Periodic Boundary Conditions
----------------------------

If the `ase.Atoms` object has periodic boundary conditions (PBCs), the simulation engine will
use them by default. Note that non-orthorhombic (non-diagonal) cells are currently supported by the
:py:class:`ASESimulationEngine <mlip.simulation.ase.ase_simulation_engine.ASESimulationEngine>`,
but not by the
:py:class:`JaxMDSimulationEngine <mlip.simulation.jax_md.jax_md_simulation_engine.JaxMDSimulationEngine>`.
We intend to support non-orthorhombic PBCs with Jax-MD in future versions.

If the `ase.Atoms` object does not have PBCs set, the `box` attribute of the
:py:class:`SimulationConfig <mlip.simulation.configs.simulation_config.SimulationConfig>`
is used to set them. This attribute can either be `None` (no PBCs), a float (cubic PBCs),
or a list of three floats (orthorhombic PBCs).

.. _neb_ts_search:

Transition state search with the NEB method
-------------------------------------------

Transition state searches can be conducted with the
:py:class:`NEBSimulationEngine <mlip.simulation.ts_search.neb_simulation_engine.NEBSimulationEngine>`,
which wraps
`ASE's nudged elastic band implementation <https://ase-lib.org/ase/neb.html>`_. Instead
of a single
``ase.Atoms`` object, the engine takes a list of images: (a) two entries are
interpreted as the initial and final state and are interpolated via the
`IDPP <https://ase-lib.org/examples_generated/tutorials/neb_idpp.html>`_
method up to ``num_images``, (b) three entries treat the middle one as a transition
state guess and interpolate on either side, and (c) more than three entries are used
as is.

The optimizer (BFGS or FIRE), spring constant ``neb_k``, climbing image
option ``climb``, and tangent formulation ``neb_method`` are set on
:py:class:`NEBSimulationConfig <mlip.simulation.configs.neb_config.NEBSimulationConfig>`
(also accessible via ``NEBSimulationEngine.Config``). The simulation runs until
either ``num_steps`` is reached or the maximum atomic force drops below
``max_force_convergence_threshold``. Results are stored in a
:py:class:`NEBSimulationState <mlip.simulation.state.NEBSimulationState>`. The additional
``forces_real`` field holds the physical forces on each image
before the band-tangent projection and spring forces are applied. See an example of
usage below.

.. code-block:: python

    from ase.io import read as ase_read
    from mlip.simulation.ts_search import NEBSimulationEngine

    initial = ase_read("/path/to/reactant.xyz")
    final = ase_read("/path/to/product.xyz")
    force_field = _get_ff_from_somewhere()  # placeholder

    neb_config = NEBSimulationEngine.Config()  # all defaults
    neb_engine = NEBSimulationEngine([initial, final], force_field, neb_config)
    neb_engine.run()

    neb_state = neb_engine.state
    print(neb_state.positions.shape)       # (num_images, num_atoms, 3)
    print(neb_state.potential_energy[-1])  # energies along the band per snapshot
    print(neb_state.forces_real[-1])       # unprojected per-image forces

Note that the NEB method assumes the endpoints are already relaxed local
minima. If they are not, run an energy minimization on each first as described
in the :ref:`ASE section <simulations_ase_user_guide>` above.
