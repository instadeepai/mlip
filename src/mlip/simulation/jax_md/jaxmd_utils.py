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

"""Contents of this file:
----------------------
This file copies a few functions from
https://github.com/jax-md/jax-md/blob/main/jax_md/simulate.py and modifies a few lines
inside them to enable batched simulations.
"""

# ruff: noqa: N803, N806, B008

from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit, random
from jax_md.simulate import (
    Array,
    Normal,
    ShiftFn,
    canonicalize_mass,
    dataclasses,
    dispatch_by_state,
    f32,
    momentum_step,
    position_step,
    tree_flatten,
    tree_unflatten,
)

from mlip.simulation.jax_md.auxiliary_properties import AuxiliaryProperties


@dataclasses.dataclass
class _NVTLangevinState:
    """Overriding to use `tree_map` in `velocity` property."""

    position: Array
    momentum: Array
    force: Array
    mass: Array
    rng: Array
    aux_properties: AuxiliaryProperties

    @property
    def velocity(self) -> Array:
        return jax.tree.map(lambda mom, mass: mom / mass, self.momentum, self.mass)


@dataclasses.dataclass
class _NVEState:
    """Overriding to use `jax.tree.map` in the `velocity` property."""

    position: Array
    momentum: Array
    force: Array
    mass: Array
    aux_properties: AuxiliaryProperties

    @property
    def velocity(self) -> Array:
        return jax.tree.map(lambda mom, mass: mom / mass, self.momentum, self.mass)


@dispatch_by_state
def _initialize_momenta(
    state: _NVTLangevinState | _NVEState, key: Array, kT: float
) -> _NVTLangevinState | _NVEState:
    """Overriding method to use tree_map for batched momentum initialisation."""
    R, mass = state.position, state.mass

    R, treedef = tree_flatten(R)
    mass, _ = tree_flatten(mass)
    keys, _ = tree_flatten(key)
    if len(keys) < len(R):
        keys = keys * len(R)

    def initialize_fn(k, r, m):
        p = jnp.sqrt(m * kT) * random.normal(k, r.shape, dtype=r.dtype)
        # If simulating more than one particle, center the momentum.
        if r.shape[0] > 1:
            p = p - jnp.mean(p, axis=0, keepdims=True)
        return p

    P = [initialize_fn(k, r, m) for k, r, m in zip(keys, R, mass)]

    return state.set(momentum=tree_unflatten(treedef, P))


@dispatch_by_state
def _stochastic_step(state, dt, kT, gamma):
    """Stochastic step with tree-mapping for batched NVT simulations."""

    c1 = jnp.exp(-gamma * dt)
    c2 = jnp.sqrt(kT * (1 - c1**2))

    state_keys = jax.tree.map(lambda key: random.split(key, 2), state.rng)
    new_key = jax.tree.map(lambda keys: keys[0], state_keys)
    momentum_key = jax.tree.map(lambda keys: keys[1], state_keys)

    new_momentum = jax.tree.map(
        lambda momentum, mass, key: Normal(c1 * momentum, c2**2 * mass).sample(key),
        state.momentum,
        state.mass,
        momentum_key,
    )

    return state.set(momentum=new_momentum, rng=new_key)


def batched_nvt_langevin(force_fn, shift_fn, dt, kT, gamma=0.1):
    """Copy of `jax_md.simulate.nvt_langevin` using tree-map to enable batched sims.

    Exact copy of original method, with these changes:
        - Replaces `NVTLangevinState` with `_NVTLangevinState` which uses
          tree-map in `velocity` and allows to set auxiliary properties.
        - Initializes momenta using the same RNG key for each system in the batch.
        - Uses a tree-mapped stochastic step.
        - Accepts only a force function, not an energy function.
        - Forwards auxiliary properties to state.

    See
    https://jax-md.readthedocs.io/en/main/jax_md.simulate.html#jax_md.simulate.nvt_langevin
    for the documentation of the original function.
    """

    @jit
    def init_fn(key, R, mass=f32(1.0), **kwargs):
        _kT = kwargs.pop("kT", kT)
        two_keys = jax.tree.map(random.split, key)
        state_key = jax.tree.map(lambda x: x[0], two_keys)
        momentum_init_key = jax.tree.map(lambda x: x[1], two_keys)
        if isinstance(R, list) and not isinstance(state_key, list):
            state_key = [state_key] * len(R)
            momentum_init_key = [momentum_init_key] * len(R)
        force, aux_prop = force_fn(R, **kwargs)
        state = _NVTLangevinState(R, None, force, mass, state_key, aux_prop)
        state = canonicalize_mass(state)
        return _initialize_momenta(state, momentum_init_key, _kT)

    @jit
    def step_fn(state, **kwargs):
        _dt = kwargs.pop("dt", dt)
        _kT = kwargs.pop("kT", kT)
        dt_2 = _dt / 2

        state = momentum_step(state, dt_2)
        state = position_step(state, shift_fn, dt_2, **kwargs)

        # This is the only modified line:
        state = _stochastic_step(state, _dt, _kT, gamma)

        state = position_step(state, shift_fn, dt_2, **kwargs)
        force, aux_prop = force_fn(state.position, **kwargs)
        state = state.set(force=force, aux_properties=aux_prop)
        state = momentum_step(state, dt_2)

        return state

    return init_fn, step_fn


def _velocity_verlet(
    force_fn: Callable[..., Array],
    shift_fn: ShiftFn,
    dt: float,
    state: _NVEState,
    **kwargs,
) -> _NVEState:
    """Apply a single step of velocity Verlet integration to a state.

    Copy of `simulate.velocity_verlet` in JAX-MD with only updating that the auxiliary
    properties are forwarded to the state.
    """
    dt = f32(dt)
    dt_2 = f32(dt / 2)

    state = momentum_step(state, dt_2)
    state = position_step(state, shift_fn, dt, **kwargs)
    force, aux_prop = force_fn(state.position, **kwargs)
    state = state.set(force=force, aux_properties=aux_prop)
    state = momentum_step(state, dt_2)

    return state


def batched_nve_velocity_verlet(force_fn, shift_fn, dt=1e-3, **sim_kwargs):
    """Copy of `jax_md.simulate.nve` using tree-map to enable batched sims.

    Exact copy of original method, with these changes:
        - Replaces `NVEState` with `_NVEState` which uses tree-map in `velocity`
          and allows to set auxiliary properties.
        - Initializes momenta using the same RNG key for each system in the batch.
        - Removes `kT` kwarg inside `step_fn` to prevent it being passed to the FF.
        - Accepts only a force function, not an energy function.
        - Forwards auxiliary properties to state. For that the `velocity_verlet`
          function is replaced by the local `_velocity_verlet` adaptation.

    See https://jax-md.readthedocs.io/en/main/jax_md.simulate.html#jax_md.simulate.nve
    for the documentation of the original function.
    """

    @jit
    def init_fn(key, R, kT, mass=f32(1.0), **kwargs):
        force, aux_prop = force_fn(R, **kwargs)
        state = _NVEState(R, None, force, mass, aux_prop)
        state = canonicalize_mass(state)
        return _initialize_momenta(state, key, kT)

    @jit
    def step_fn(state, **kwargs):
        _dt = kwargs.pop("dt", dt)

        # Remove `kT`: passed by the engine but not used by `velocity_verlet`.
        kwargs.pop("kT", None)

        return _velocity_verlet(force_fn, shift_fn, _dt, state, **kwargs)

    return init_fn, step_fn
