# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit, lax, random
from jax_md import dataclasses, simulate
from jax_md.dataclasses import dataclass as jax_compatible_dataclass
from jax_md.simulate import Array, ShiftFn

from mlip.simulation.jax_md.auxiliary_properties import AuxiliaryProperties
from mlip.simulation.jax_md.jaxmd_utils import _NVTLangevinState, batched_nvt_langevin
from mlip.simulation.montecarlo_barostat import (
    INITIAL_MAX_DELTA_VOLUME_FRACTION,
    MonteCarloBarostatState,
    accept_volume_change,
    propose_volume_change,
    sanitize_molecule_indices,
    tune_barostat,
)
from mlip.utils.jax_utils import TupleLeaf, segment_sum


@jax_compatible_dataclass
class NPTLangevinState:
    """
    Dataclass holding the state of an `NPT_MC_LANGEVIN` simulation.

    Attributes:
        langevin_state: The state of the Langevin dynamics.
        box: The box dimensions.
        barostat_state: The state of the Monte Carlo Barostat.
        step_count: The number of steps taken.
    """

    langevin_state: _NVTLangevinState

    # NPT specific fields
    box: Array
    barostat_state: MonteCarloBarostatState
    step_count: int

    @property
    def position(self):
        return self.langevin_state.position

    @property
    def momentum(self):
        return self.langevin_state.momentum

    @property
    def force(self):
        return self.langevin_state.force

    @property
    def mass(self):
        return self.langevin_state.mass

    @property
    def rng(self):
        return self.langevin_state.rng

    @property
    def velocity(self):
        return self.langevin_state.velocity

    @property
    def aux_properties(self) -> AuxiliaryProperties:
        return self.langevin_state.aux_properties


def apply_montecarlo_barostat(
    state: NPTLangevinState,
    energy_fn: Callable,
    force_fn: Callable,
    kT: float,  # noqa: N803
) -> NPTLangevinState:
    """Apply the Monte Carlo Barostat step to the NPT Langevin simulation.

    Returns a new NPTLangevinState with the updated positions and box, by proposing
    a volume change, and accepting or rejecting based on a Metropolis criterion:
        acceptance_probability = exp(-Delta_W / kB * T)
        where Delta_W = Delta_E + P * Delta_V - N_mol * kB * T * ln(V_new / V_old)

    See the OpenMM documentation for more details:
    docs.openmm.org/latest/userguide/theory/02_standard_forces.html#montecarlobarostat

    Args:
        state: The current state of the simulation.
        energy_fn: The energy function.
        force_fn: The force function used to recompute forces after an accepted move.
        kT: The temperature in energy units (kB * T).

    Returns:
        The updated state of the simulation with the new box and positions.
    """
    is_mc_bs = lambda x: isinstance(x, MonteCarloBarostatState)  # noqa: E731
    is_tl = lambda x: isinstance(x, TupleLeaf)  # noqa: E731

    # Propose volume changes — one call per system via tree.map
    proposals = jax.tree.map(
        lambda bs, box, pos: TupleLeaf(propose_volume_change(bs, box, pos)),
        state.barostat_state,
        state.box,
        state.position,
        is_leaf=is_mc_bs,
    )
    new_barostat_state = jax.tree.map(lambda p: p[0], proposals, is_leaf=is_tl)
    vol_old = jax.tree.map(lambda p: p[1], proposals, is_leaf=is_tl)
    vol_new = jax.tree.map(lambda p: p[2], proposals, is_leaf=is_tl)
    box_new = jax.tree.map(lambda p: p[3], proposals, is_leaf=is_tl)
    pos_new = jax.tree.map(lambda p: p[4], proposals, is_leaf=is_tl)

    # Energy at current and proposed positions (one call each)
    energy_old = energy_fn(state.position, box=state.box)
    energy_new = energy_fn(pos_new, box=box_new)

    # Accept / reject and tune per system.
    accept_results = jax.tree.map(
        lambda bs, e_old, e_new, v_old, v_new: TupleLeaf(
            accept_volume_change(
                bs, e_old, e_new, v_old, v_new, kT, bs.mol_counts.shape[0]
            )
        ),
        new_barostat_state,
        energy_old,
        energy_new,
        vol_old,
        vol_new,
        is_leaf=is_mc_bs,
    )
    new_barostat_state = jax.tree.map(lambda r: r[0], accept_results, is_leaf=is_tl)
    accepted = jax.tree.map(lambda r: r[1], accept_results, is_leaf=is_tl)

    new_barostat_state = jax.tree.map(
        tune_barostat, new_barostat_state, vol_old, is_leaf=is_mc_bs
    )

    # Select final positions and boxes
    final_positions = jax.tree.map(
        lambda pos_p, pos_o, acc: jnp.where(acc, pos_p, pos_o),
        pos_new,
        state.position,
        accepted,
    )
    final_boxes = jax.tree.map(
        lambda box_p, box_o, acc: jnp.where(acc, box_p, box_o),
        box_new,
        state.box,
        accepted,
    )

    # Forces at proposed positions; selected via jnp.where (one batched call)
    forces_proposed, _ = force_fn(pos_new, box=box_new)
    final_forces = jax.tree.map(
        lambda f_new, f_old, acc: jnp.where(acc, f_new, f_old),
        forces_proposed,
        state.force,
        accepted,
    )

    new_langevin = dataclasses.replace(
        state.langevin_state, position=final_positions, force=final_forces
    )
    return dataclasses.replace(
        state,
        langevin_state=new_langevin,
        box=final_boxes,
        barostat_state=new_barostat_state,
    )


def npt_montecarlo_langevin(
    langevin_force_fn: Callable[..., Array],
    barostat_energy_fn: Callable[..., Array],
    shift_fn: ShiftFn,
    dt: float,
    kT: float,  # noqa: N803
    pressure: float,
    molecule_indices: Array | list[Array],
    gamma: float = 0.1,
    barostat_interval: int = 25,
) -> simulate.Simulator:
    """Simulation in the NPT ensemble using Langevin dynamics + Monte Carlo Barostat.

    Works for both single-system and batched simulations.  For batched simulations,
    pass a list of position arrays and a list of boxes; per-system
    `MonteCarloBarostatState` objects are initialised automatically.

    Args:
        langevin_force_fn: The force function for the Langevin dynamics.
        barostat_energy_fn: The energy function for the Monte-Carlo Barostat.
        shift_fn: The shift function.
        dt: The duration of each timestep.
        kT: The temperature in energy units (kB * T).
        pressure: The external pressure.
        molecule_indices: Array specifying which molecule each atom belongs to.
        gamma: The friction coefficient for the Langevin dynamics.
        barostat_interval: The interval for the Monte-Carlo Barostat in timesteps.
    """
    # Infer static molecular topology information for each system.
    _mol_indices_list = (
        molecule_indices if isinstance(molecule_indices, list) else [molecule_indices]
    )
    _mol_indices_list = [sanitize_molecule_indices(m) for m in _mol_indices_list]
    _num_molecules_list = [int(jnp.max(m)) + 1 for m in _mol_indices_list]
    _mol_counts_list = [
        segment_sum(
            jnp.ones(m.shape[0]),
            segment_ids=m,
            num_segments=n,
            deterministic=True,
        )
        for m, n in zip(_mol_indices_list, _num_molecules_list)
    ]

    # batched_nvt_langevin handles both single-system (array) and batched (list) inputs
    nvt_init_fn, nvt_step_fn = batched_nvt_langevin(
        langevin_force_fn, shift_fn, dt, kT, gamma
    )

    @jit
    def init_fn(key, positions, box, mass=1.0, **kwargs):
        langevin_state = nvt_init_fn(key, positions, mass=mass, box=box, **kwargs)

        # Normalise to lists so the same setup code works for single and batched
        _positions = positions if isinstance(positions, list) else [positions]
        _box = box if isinstance(box, list) else [box]
        N = len(_positions)

        assert len(_mol_indices_list) == N, (
            f"Expected one set of molecule_indices per system ({N}), "
            f"got {len(_mol_indices_list)}."
        )

        def _make_barostat_state(i):
            box_i = jnp.asarray(_box[i])
            vol = (
                box_i ** _positions[i].shape[1] if box_i.ndim == 0 else jnp.prod(box_i)
            )
            return MonteCarloBarostatState(
                target_pressure=pressure,
                max_delta_volume=vol * INITIAL_MAX_DELTA_VOLUME_FRACTION,
                num_attempted=0,
                num_accepted=0,
                num_attempted_since_tune=0,
                num_accepted_since_tune=0,
                mol_counts=_mol_counts_list[i],
                mol_indices=_mol_indices_list[i],
                rng=random.fold_in(key[i] if isinstance(key, list) else key, i),
            )

        bs_list = [_make_barostat_state(i) for i in range(N)]
        # Store as a plain object for single-system to preserve the existing state shape
        barostat_state = bs_list if isinstance(positions, list) else bs_list[0]

        return NPTLangevinState(
            langevin_state=langevin_state,
            box=box,
            barostat_state=barostat_state,
            step_count=0,
        )

    @jit
    def step_fn(state: NPTLangevinState, **kwargs):
        # Allow dynamic overrides
        _dt = kwargs.pop("dt", dt)
        _kT = kwargs.pop("kT", kT)  # noqa: N806

        # Run Langevin step and update the state
        new_langevin = nvt_step_fn(
            state.langevin_state, dt=_dt, kT=_kT, box=state.box, **kwargs
        )
        state = dataclasses.replace(
            state, langevin_state=new_langevin, step_count=state.step_count + 1
        )

        # Run Monte Carlo Barostat step if needed
        is_mc_step = (state.step_count % barostat_interval) == 0

        def barostat_wrapper(s):
            def _energy_fn(pos, box):
                return barostat_energy_fn(pos, box=box, **kwargs)

            def _force_fn(pos, box):
                return langevin_force_fn(pos, box=box, **kwargs)

            return apply_montecarlo_barostat(s, _energy_fn, _force_fn, _kT)

        return lax.cond(is_mc_step, barostat_wrapper, lambda s: s, state)

    return init_fn, step_fn
