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

"""
Contents of this file:
----------------------
This file copies a few functions from
https://github.com/jax-md/jax-md/blob/main/jax_md/simulate.py and modifies a few lines
inside them to enable batched simulations.
"""

# flake8: noqa=N803,N806,B008

import jax.numpy as jnp
from jax import jit, random
from jax.tree_util import tree_leaves, tree_map
from jax_md.simulate import (
    Array,
    Normal,
    canonicalize_mass,
    dataclasses,
    dispatch_by_state,
    f32,
    momentum_step,
    position_step,
    quantity,
    tree_flatten,
    tree_unflatten,
)

from mlip.utils.jax_utils import TupleLeaf


@dispatch_by_state
def _stochastic_step(state, dt, kT, gamma):
    """New stochastic step, that can be used in a batched simulation."""

    def _step_impl(momentum, mass):
        c1 = jnp.exp(-gamma * dt)
        c2 = jnp.sqrt(kT * (1 - c1**2))
        momentum_dist = Normal(c1 * momentum, c2**2 * mass)
        key, split = random.split(state.rng)
        return TupleLeaf([momentum_dist.sample(split), key])

    step_result = tree_map(_step_impl, state.momentum, state.mass)
    new_momentum = tree_map(lambda x: x[0], step_result)
    new_rng = tree_leaves(tree_map(lambda x: x[1], step_result))[0]

    return state.set(momentum=new_momentum, rng=new_rng)


@dataclasses.dataclass
class _NVTLangevinState:
    """Overriding because the original velocity property computation does
    not use `tree_map`.
    """

    position: Array
    momentum: Array
    force: Array
    mass: Array
    rng: Array

    @property
    def velocity(self) -> Array:
        return tree_map(lambda mom, mass: mom / mass, self.momentum, self.mass)


@dispatch_by_state
def _initialize_momenta(
    state: _NVTLangevinState, key: Array, kT: float
) -> _NVTLangevinState:
    """Overriding this to use the same key for each system in batch."""
    R, mass = state.position, state.mass

    R, treedef = tree_flatten(R)
    mass, _ = tree_flatten(mass)
    keys = [key] * len(R)  # This line is different from the original implementation

    def initialize_fn(k, r, m):
        p = jnp.sqrt(m * kT) * random.normal(k, r.shape, dtype=r.dtype)
        # If simulating more than one particle, center the momentum.
        if r.shape[0] > 1:
            p = p - jnp.mean(p, axis=0, keepdims=True)
        return p

    P = [initialize_fn(k, r, m) for k, r, m in zip(keys, R, mass)]

    return state.set(momentum=tree_unflatten(treedef, P))


def batched_nvt_langevin(energy_or_force_fn, shift_fn, dt, kT, gamma=0.1):
    """Exact copy of JAX-MD's `jax_md.simulate.nvt_langevin`,
    but with a tree-mapped stochastic step, such that we can use this for batched
    simulations. See
    https://jax-md.readthedocs.io/en/main/jax_md.simulate.html#jax_md.simulate.nvt_langevin
    for the documentation of this function.

    Furthermore, the initialization of the momenta and the `NVTLangevinState` are
    slightly modified versions of the original as well.
    """
    force_fn = quantity.canonicalize_force(energy_or_force_fn)

    @jit
    def init_fn(key, R, mass=f32(1.0), **kwargs):
        _kT = kwargs.pop("kT", kT)
        key, split = random.split(key)
        force = force_fn(R, **kwargs)
        state = _NVTLangevinState(R, None, force, mass, key)
        state = canonicalize_mass(state)
        return _initialize_momenta(state, split, _kT)

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
        state = state.set(force=force_fn(state.position, **kwargs))
        state = momentum_step(state, dt_2)

        return state

    return init_fn, step_fn
