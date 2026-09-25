.. _enhanced_sampling:

Enhanced Sampling
=================

This library supports two enhanced sampling methods,

* Metadynamics, and
* Free Energy Perturbation (FEP),

both using the `JAX-MD <https://jax-md.readthedocs.io/>`_ simulation backend.

.. _metadynamics:

Metadynamics
------------

Metadynamics is an enhanced-sampling technique that adds a history-dependent
bias potential along one or more *collective variables* (CVs) to help the system
escape free-energy basins and explore configuration space more efficiently.
The implemented variant is **well-tempered metadynamics**
(`Barducci et al., PRL 2008 <https://doi.org/10.1103/PhysRevLett.100.020603>`_):
Gaussian hills are deposited periodically and their heights are rescaled by a
factor that depends on the accumulated bias, preventing the bias from growing
without bound. Setting `bias_factor=None` disables the rescaling and recovers
plain (untempered) metadynamics, equivalent to the γ → ∞ limit.

The
:py:class:`JaxMDMetadynamicsSimulationEngine <mlip.simulation.metadynamics.jax_md_metad_engine.JaxMDMetadynamicsSimulationEngine>`
extends the JAX-MD simulation engine and is configured via
:py:class:`MetadynamicsConfig <mlip.simulation.metadynamics.config.MetadynamicsConfig>`
embedded inside
:py:class:`JaxMDMetadynamicsSimulationConfig <mlip.simulation.metadynamics.config.JaxMDMetadynamicsSimulationConfig>`.

Note that batched simulations with metadynamics are not currently supported.

For a worked end-to-end example, we refer to our
`metadynamics tutorial notebook <https://github.com/instadeepai/mlip/blob/main/tutorials/metadynamics_tutorial.ipynb>`_.

**Minimal example** (distance CV, upper wall):

.. code-block:: python

    from ase.io import read as ase_read
    from mlip.simulation.metadynamics.jax_md_metad_engine import (
        JaxMDMetadynamicsSimulationEngine,
    )
    from mlip.simulation.metadynamics.config import MetadynamicsConfig
    from mlip.simulation.metadynamics.potential_terms import (
        DistanceCVConfig,
        DistanceWallConfig,
    )
    from mlip.simulation.enums import SimulationType, MDIntegrator

    atoms = ase_read("/path/to/structure.xyz")
    force_field = _get_a_trained_force_field_from_somewhere()  # placeholder

    metad_config = MetadynamicsConfig(
        bias_cvs=[DistanceCVConfig(atom_indices_1=[10], atom_indices_2=[30])],
        bias_sigmas=[0.2],                  # Å
        bias_factor=15.0,                   # well-tempered γ
        deposition_interval=500,            # steps between hill depositions
        initial_height=0.02,                # eV
        walls=[
            DistanceWallConfig(
                atom_indices_1=[10], atom_indices_2=[30], upper=3.5, kappa=50.0, exp=2
            )
        ],
    )

    engine_config = JaxMDMetadynamicsSimulationEngine.Config(
        metadynamics_config=metad_config,
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator.NVT_LANGEVIN,
        num_steps=500_000,
        snapshot_interval=10,
        num_episodes=500,
        timestep_fs=1.0,
        temperature_kelvin=300.0,
    )

    engine = JaxMDMetadynamicsSimulationEngine(atoms, force_field, engine_config)
    engine.run()

    state = engine.state
    print(state.bias_cv_values.shape)   # (n_snapshots, num_cvs) — bias CV trajectory
    print(state.bias_potential.shape)   # (n_snapshots,) — bias energy at each snapshot

Collective variables
~~~~~~~~~~~~~~~~~~~~

One or more collective variables (CVs) can be included in the bias potential via the
`bias_cvs` list; the corresponding `bias_sigmas` list must have the same length. Note
that it is recommended to use 1-2 CVs as the volume of CV space grows exponentially
with dimensionality, making it harder to convergence.

Collective variables are specified using their config classes:

.. code-block:: python

    import math
    from mlip.simulation.metadynamics.potential_terms import (
        AngleCVConfig,
        DistanceCVConfig,
    )

    bias_cvs = [
        DistanceCVConfig(atom_indices_1=[10], atom_indices_2=[30]),
        AngleCVConfig(atom_indices=[5, 10, 30]),
    ]

The available CV config classes are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Config class
     - Description
   * - :py:class:`DistanceCVConfig <mlip.simulation.metadynamics.potential_terms.DistanceCVConfig>`
     - Pairwise distance between the centroids of two atom groups `a = [i, j, k, ...]`
       and `b = [l, m, n, ...]` in Å, where
       each group contains one or more atoms. Set `atom_indices_1=a` and `atom_indices_2=b`.
   * - :py:class:`AngleCVConfig <mlip.simulation.metadynamics.potential_terms.AngleCVConfig>`
     - Bond angle for a triplet *p–q–r* where *q* is the vertex (radians). Set `atom_indices=(p, q, r)`.
   * - :py:class:`DihedralCVConfig <mlip.simulation.metadynamics.potential_terms.DihedralCVConfig>`
     - Dihedral angle for a quadruplet *i–j–k–l*. Set `atom_indices=(i, j, k, l)`.
   * - :py:class:`CoordinationNumberCVConfig <mlip.simulation.metadynamics.potential_terms.CoordinationNumberCVConfig>`
     - Differentiable coordination number of a central atom with respect to a
       neighbor element, computed via a rational switching function. Set
       `central_idx` and `element` (element symbol, e.g. `"N"`).

Walls
~~~~~

Wall potentials confine a CV to a desired range without affecting the bias.
They are one-sided potentials: `V = kappa * max(s - upper, 0)^exp` or
`V = kappa * max(lower - s, 0)^exp`. Pass a list of wall configs via `walls`:

.. code-block:: python

    import math
    from mlip.simulation.metadynamics.potential_terms import (
        AngleWallConfig,
        DistanceWallConfig,
    )

    walls = [
        # Keep distance (atoms 10, 30) below 3.5 Å
        DistanceWallConfig(
            atom_indices_1=[10], atom_indices_2=[30], upper=3.5, kappa=50.0
        ),
        # Keep angle (atoms 5, 10, 30) above 80° and below 150°
        AngleWallConfig(
            atom_indices=(5, 10, 30),
            lower_rad=math.radians(80.0),
            upper_rad=math.radians(150.0),
            kappa=100.0,
        ),
    ]

Available wall config classes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Config class
     - Description
   * - :py:class:`DistanceWallConfig <mlip.simulation.metadynamics.potential_terms.DistanceWallConfig>`
     - Wall potential on a distance CV (Å). Accepts `lower` and/or `upper` thresholds.
   * - :py:class:`AngleWallConfig <mlip.simulation.metadynamics.potential_terms.AngleWallConfig>`
     - Wall potential on a bond-angle CV. Thresholds set in radians via `lower_rad` / `upper_rad`.

Positional restraints
~~~~~~~~~~~~~~~~~~~~~

Positional restraints apply a harmonic penalty `V = 0.5 * kappa * Σ |r_i - r0_i|²`
to keep a set of atoms near their initial positions. This is useful for example to keep a
solvent shell or spectator atoms from drifting while a reactive fragment is
biased. Pass a list of
:py:class:`PositionalRestraintConfig <mlip.simulation.metadynamics.potential_terms.PositionalRestraintConfig>`
objects via `restraints`:

.. code-block:: python

    from mlip.simulation.metadynamics.potential_terms import PositionalRestraintConfig

    restraints = [
        # Restrain atoms 0–19 with kappa = 100 eV/Å²
        PositionalRestraintConfig(atom_indices=list(range(20)), kappa=100.0)
    ]

Alternatively, set `start_atom_index` instead of `atom_indices` and the engine
will automatically identify the restrained fragment via BFS over an implicit
bond graph (added for all pairs with distance 0.1–1.8 Å) starting from `start_atom_index`.

Simulation state
~~~~~~~~~~~~~~~~

After running, `engine.state` is a
:py:class:`MetadynamicsSimulationState <mlip.simulation.metadynamics.states.MetadynamicsSimulationState>`
that extends the standard
:py:class:`SimulationState <mlip.simulation.state.SimulationState>` with:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - `bias_cv_values`
     - Values of the CVs used by the bias potential at each logged snapshot.
   * - `bias_potential`
     - Total bias energy (eV) at each logged snapshot.
   * - `gaussian_centers`
     - Positions of all deposited Gaussian hills along the bias potential CVs.
   * - `gaussian_heights`
     - Heights (eV) of all deposited Gaussian hills.


.. _fep_simulations:

FEP Simulations
---------------

Free Energy Perturbation (FEP) is a statistical mechanics method used to compute the
free-energy difference between two "end-states" of a system. Instead of simulating
physical pathways, it relies on "alchemical" transitions, such as artificially vanishing
a solute so it is fully decoupled from its solvent, or morphing one molecule into
another while holding the surrounding environment
fixed (`Zwanzig, R. W. JCP 1954 <https://doi.org/10.1063/1.1740409>`_,
for a modern review
`Mey, Antonia SJ et al., Liv. J. Comput. Mol. Sci. 2020 <https://doi.org/10.33011/livecoms.2.1.18378>`_).

FEP utilizes a series of simulations mapped along a non-physical alchemical path
between two end-states. By using auxiliary "lambda" (:math:`\lambda`) parameters to scale
the system’s Hamiltonian, we smoothly transition between these states computationally.
The total free-energy difference is then recovered by pooling the energy changes sampled
at each intermediate lambda window.

Our current implementation supports a single alchemical transformation pathway: the
gradual decoupling of a selected set of atoms from the rest of the system. This is
achieved by scaling the interaction edges of the MLIP graph. To transition from the
fully interacting (State A) to the fully decoupled (State B) state, as required for
solvation free energy calculations, we employ a three-stage pathway
(:math:`A \to R \to B`) inspired by
`Xie et al., JCTC 2026 <https://doi.org/10.1021/acs.jctc.5c02019>`_:

- State A (Fully Coupled): The target atoms interact normally with their surroundings.
- State R (Repulsive): Directly removing interaction edges can allow clashes between
  atoms before the edge is completely removed, which can cause spikes in the MLIP
  energy. To prevent this, we introduce a softcore repulsive potential along the
  alchemical path to keep the two subsystems physically separated while the MLIP graph
  edges are being scaled down, which is at maximal strength at this intermediate stage.
- State B (Fully Decoupled): The target atoms are entirely isolated from the rest of the
  system.

In practice, the calculation is performed by running multiple parallel replicas of the
system, each mapped to a specific coordinate (a "lambda window") along this path.
The final free energy is then estimated from the potential energy fluctuations between
adjacent windows using estimators like
BAR (`Bennett C.H., JCP 1976 <https://doi.org/10.1016/0021-9991(76)90112-2>`_)
or MBAR
(`Shirts, M. R., & Chodera, J. D. JCP 2006, <https://doi.org/10.1063/1.2978177>`_).

The
:py:class:`FEPSimulationSampler <mlip.fep.sampler.fep_jaxmd_sampler.FEPSimulationSampler>`
orchestrates one JAX-MD simulation engine per lambda window, dispatches them
in parallel across local devices (and across hosts in multi-host TPU
deployments), and optionally performs Hamiltonian Replica Exchange (HREX)
between adjacent windows. It is configured via
:py:class:`FEPSimulationSamplerConfig <mlip.fep.sampler.fep_sampler_config.FEPSimulationSamplerConfig>`,
which embeds a standard
:py:class:`JaxMDSimulationConfig <mlip.simulation.configs.jax_md_config.JaxMDSimulationConfig>`
describing the MD settings shared by every window.

Note that, unlike the :ref:`batched simulations <batched_simulations>`
described in :ref:`simulation section <simulations>`, each lambda window runs a
single (non-batched) system; the parallelism here is across lambda windows rather
than across independent systems in one batch. Also note that Hessian predictions
are not supported when running FEP simulations.

For a worked end-to-end example, we refer to our
`FEP tutorial notebook <https://github.com/instadeepai/mlip/blob/main/tutorials/fep_tutorial.ipynb>`_.

**Minimal example** (hydration free energy of a solute in a water box):

.. code-block:: python

    from ase.io import read as ase_read
    import numpy as np

    from mlip.simulation.fep.sampler import FEPSimulationSampler
    from mlip.simulation.configs.jax_md_config import JaxMDSimulationConfig
    from mlip.simulation.enums import SimulationType, MDIntegrator

    atoms = ase_read("/path/to/solute_in_water_box.pdb")
    molecule_indices = np.load("path/to/system_molecule_indices.npy")
    force_field = _get_a_trained_force_field()  # placeholder

    # The first 20 atoms are the solute; these are the atoms to be decoupled
    # from the rest of the system (the solvent).
    alchemical_atom_indices = list(range(20))

    # Lambda schedule along the path:
    # A (edge=1, rep=0) -> R (edge=0, rep=1) -> B (edge=0, rep=0)
    lambda_edge_values = [1.00, 0.71, 0.43, 0.14, 0.00, 0.00, 0.00, 0.00, 0.00]
    lambda_repulsion_values = [0.00, 0.29, 0.57, 0.86, 1.00, 0.75, 0.50, 0.25, 0.00]

    engine_config = JaxMDSimulationConfig(
        simulation_type=SimulationType.MD,
        md_integrator=MDIntegrator.NPT_MC_LANGEVIN,
        num_steps=270_000,
        num_episodes=540,
        snapshot_interval=50,
        timestep_fs=1.0,
        temperature_kelvin=298.15,
        molecule_indices=molecule_indices,
        pressure_bar=1.01325,
        barostat_update_interval=25,
    )

    sampler_config = FEPSimulationSampler.Config(
        simulation_config=engine_config,
        alchemical_atom_indices=alchemical_atom_indices,
        lambda_edge_values=lambda_edge_values,
        lambda_repulsion_values=lambda_repulsion_values,
        use_alchemical_mlip=True,
        use_replica_exchange=True,
        num_equilibration_episodes=40,
        checkpoint_interval_episodes=20,
    )

    sampler = FEPSimulationSampler(atoms, force_field, sampler_config)
    sampler.run()

Note that in the example above, `_get_a_trained_force_field()` is a placeholder
for a function that loads a trained force field, as described either
:ref:`here <load_zip_model>` (Option 1) or :ref:`here <load_trained_model>` (Option 2).
Instead of a single `ase.Atoms` object, one may also pass a list with one
entry per lambda window (all describing the same system, e.g., differing only
in starting positions or velocities) to `FEPSimulationSampler`.

Alchemical atoms and the lambda schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`alchemical_atom_indices` selects the atoms to be treated as alchemical
(typically the solute). Edges connecting these atoms to the rest of the
system are the ones gradually switched off along the simulation.
`lambda_edge_values` and `lambda_repulsion_values` are equal-length lists, one
entry per lambda window, where:

* `lambda_edge_values` is the **edge weight**, scaling the message-passing edges
  between alchemical and non-alchemical atoms (1.0 = fully connected, 0.0 = fully
  decoupled).
* `lambda_repulsion_values` is the **repulsive potential weight**, scaling the strength
  of the softcore repulsive potential between alchemical and non-alchemical atoms. This
  is used to prevent clashes while the alchemical atoms are not fully decoupled.

Computing the alchemical MLIP potential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For cases where the edges of the MLIP graph are partially on, it is necessary
to compute the "alchemical potential" using a pretrained MLIP model. There are
two ways to compute the alchemical energy/forces from a trained MLIP, controlled
by `use_alchemical_mlip` on
:py:class:`FEPSimulationSamplerConfig <mlip.fep.sampler.fep_sampler_config.FEPSimulationSamplerConfig>`:

* **Native alchemical MLIP** (`use_alchemical_mlip=True`, default): the edge weight
  is applied directly inside the model's message-passing equations to compute
  an alchemical energy with no computational overhead (see
  `Moore et al., arXiv 2024 <https://arxiv.org/abs/2405.18171>`_ and
  `Nam et al., arXiv 2024 <https://arxiv.org/abs/2404.10746>`_).
  We provide alchemical variants of all four supported architectures in our library,
  which can be used directly with any pretrained model. Note that this approach results
  in a non-linear relationship between the edge weight and the alchemical potential,
  which may make the lambda schedule more difficult to tune.

* **Linear combination** (`use_alchemical_mlip=False`): duplicates the graph into
  its two end-states (fully-coupled and fully-decoupled) and batches them
  together before the forward pass; the two resulting energies are then linearly
  combined, weighted by the edge weight. This works with any `MLIPNetwork`, including
  legacy v1 checkpoints, at the cost of a more expensive forward pass. For lambda
  values where the graph is either fully-coupled or fully-decoupled (i.e. for edge
  weights `1.0` or `0.0`), we do not duplicate the graph before the forward pass.
  Note that this approach results in a linear relationship between the edge weight and
  the alchemical potential, which may make the lambda schedule easier to tune.

`FEPSimulationSampler` converts the standard base input force field into the requested
alchemical variant automatically, depending on `use_alchemical_mlip`.

Softcore repulsive potentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the edge weight is being ramped down, a softcore repulsive potential
between alchemical and non-alchemical atoms is ramped up (via the repulsive
potential weight) to prevent the decoupled solute from overlapping with its
environment. Two variants are available, both parameterized per-element via
`sigma_map` / `epsilon_map` (defaulting to representative GAFF-2.1 values):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Config class
     - Description
   * - :py:class:`SoftcoreLennardJonesPotential <mlip.fep.models.repulsive_potentials.SoftcoreLennardJonesPotential>`
     - Softcore Lennard-Jones potential (default).
   * - :py:class:`SoftcoreWCAPotential <mlip.fep.models.repulsive_potentials.SoftcoreWCAPotential>`
     - Softcore Weeks–Chandler–Andersen potential (default): a repulsion-only
       equivalent of the softcore Lennard-Jones potential; shifted and truncated to
       remove the attractive region.

Pass an instance via the `repulsive_potential` attribute of
:py:class:`FEPSimulationSamplerConfig <mlip.fep.sampler.fep_sampler_config.FEPSimulationSamplerConfig>`:

.. code-block:: python

    from mlip.fep.models import SoftcoreWCAPotential

    sampler_config = FEPSimulationSampler.Config(
        repulsive_potential=SoftcoreWCAPotential(),
        **config_kwargs,
    )

Hamiltonian replica exchange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When `use_replica_exchange=True` (default), Hamiltonian Replica Exchange
(HREX, `Liu, Pu et al. PNAS 2005 <https://doi.org/10.1073/pnas.0506346102>`_)
is performed between adjacent
lambda windows after every episode, swapping their configurations according to a
Metropolis criterion. This
improves sampling by letting configurations diffuse across the whole lambda
path. The number of steps per episode (and therefore the HREX frequency)
is `simulation_config.num_steps / simulation_config.num_episodes`.

`num_equilibration_episodes` sets the number of initial episodes to run
before HREX attempts begin. Setting `use_replica_exchange=False` runs every
lambda window fully independently.

.. _fep_checkpointing:

Checkpointing
~~~~~~~~~~~~~

Setting `checkpoint_dir` on
:py:class:`FEPSimulationSamplerConfig <mlip.fep.sampler.fep_sampler_config.FEPSimulationSamplerConfig>`
saves a checkpoint (per-lambda simulation state, replica exchange PRNG key,
and completed episode count) to that local directory every
`checkpoint_interval_episodes` episodes.
To also upload the checkpoint to remote storage after each save, pass a
`checkpoint_dir_upload_fun` callable to the sampler's constructor:

.. code-block:: python

    from pathlib import Path

    def upload_checkpoint_dir(checkpoint_dir: Path) -> None:
        """Upload the checkpoint directory to remote storage."""
        _upload_directory_to_remote_storage(  # placeholder
            checkpoint_dir, "s3://my-bucket/checkpoints/run_1"
        )

    sampler = FEPSimulationSampler(
        atoms,
        force_field,
        sampler_config,  # sampler_config.checkpoint_dir must be set
        checkpoint_dir_upload_fun=upload_checkpoint_dir,
    )
    sampler.run()

To resume a run, call `sampler.restore_checkpoint(checkpoint_dir)` with a local
directory before calling `sampler.run()`; if the checkpoint directory is in remote
storage, download it first:

.. code-block:: python

    local_checkpoint_dir = Path("/path/to/local/checkpoint_dir")
    _download_directory_from_remote_storage(
        "s3://my-bucket/checkpoints/run_1", local_checkpoint_dir
    )

    sampler = FEPSimulationSampler(atoms, force_field, sampler_config)
    sampler.restore_checkpoint(local_checkpoint_dir)  # must be called before run()
    sampler.run()


Simulation state and logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As with the standard engines, loggers can be attached to a
:py:class:`FEPSimulationSampler <mlip.fep.sampler.fep_jaxmd_sampler.FEPSimulationSampler>`
via `attach_logger`. Unlike the single-engine case, the logging function must
take two arguments: the
:py:class:`FEPSimulationState <mlip.fep.sampler.states.FEPSimulationState>`,
and an integer index identifying which lambda window it belongs to (or `None`
if it is the shared global state, which only carries `replica_exchange_log`).

Note also that due to the long simulation times required, if a logger performs
expensive operations (e.g. uploading to remote storage) then you may want to
limit the logging frequency:

.. code-block:: python

    from mlip.fep.sampler.states import FEPSimulationState

    def make_logger():
        episode_counts: dict[int | None, int] = {}

        def logging_fun(state: FEPSimulationState, engine_index: int | None) -> None:
            episode_counts[engine_index] = episode_counts.get(engine_index, 0) + 1
            if episode_counts[engine_index] % log_interval_episodes != 0:
                return

            _log_something(state, engine_index)  # placeholder, e.g. an s3 upload

        return logging_fun

    sampler.attach_logger(make_logger())

:py:class:`FEPSimulationState <mlip.fep.sampler.states.FEPSimulationState>` extends
:py:class:`SimulationState <mlip.simulation.state.SimulationState>` with:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - `per_lambda_energies`
     - The energy of this window's configuration evaluated at every lambda
       value in the schedule, at each logged snapshot. Shape
       `(n_snapshots, num_lambdas)`. This is the quantity needed as input to
       free-energy estimators such as BAR or MBAR.
   * - `replica_exchange_log`
     - The log of replica exchange attempts and outcomes. Only populated on
       the shared global state (`engine_index=None`).
   * - `final_positions`, `final_velocities`, `final_cell`
     - The final positions, velocities and cell of this window's simulation.
