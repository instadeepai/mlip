# Code converted from https://github.com/facebookresearch/fairchem
# Some parts of the code may remain identical, distributed under:
#
#     MIT License- Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Copyright 2026 InstaDeep Ltd and Zhongguancun Academy
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

"""Wigner D matrices via hybrid approach (fastest method per l).

Selects methods from the `wigner_custom.py` (l<5) or `wigner_general.py` (l>=5) files.

This module provides Wigner D computation using the optimal method for each l:
    - l=0: Trivial (identity).
    - l=1: Direct quaternion to rotation matrix.
    - l=2: Degree-4 tensor contraction via einsum.
    - l=3,4: Batched polynomial matmul (single kernel dispatch for both).
    - l>=5: Ra/Rb polynomial (`wigner_general.py`), run in float64 for stability.

Entry point:
    - axis_angle_wigner_hybrid: Main function using real arithmetic throughout.
"""

import jax
import jax.numpy as jnp

from mlip.models.esen.quaternion.utils import (
    gamma_quaternion_multiply,
    quaternion_edge_to_y_stable,
)
from mlip.models.esen.quaternion.wigner_custom import (
    build_custom_kernels,
    quaternion_to_rotation_matrix,
    quaternion_to_wigner_d_l2_einsum,
    quaternion_to_wigner_d_l3l4_batched,
    quaternion_to_wigner_d_matmul,
)
from mlip.models.esen.quaternion.wigner_general import (
    wigner_d_matrix_real,
    wigner_d_pair_to_real,
)


def wigner_d_from_quaternion_hybrid(
    q: jax.Array, jd_list: list[jax.Array], lmax: int
) -> jax.Array:
    """Compute Wigner D matrices from quaternion using hybrid approach.

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion to Wigner D via degree-4 polynomial einsum
    - l=3: Quaternion matmul (used when lmax=3)
    - l=3,4: Batched quaternion matmul (used when lmax>=4)
    - l>=5: Ra/Rb polynomial

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        jd_list: J_y generators (angular momentum operators) in the |ℓ,m⟩ basis
        lmax: Maximum angular momentum

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)^2
    """
    size = (lmax + 1) ** 2
    d = jnp.zeros((q.shape[0], size, size), dtype=q.dtype)

    d = d.at[:, 0, 0].set(1.0)

    if lmax < 1:
        return d

    d = d.at[:, 1:4, 1:4].set(quaternion_to_rotation_matrix(q))

    if lmax < 2:
        return d

    with jax.ensure_compile_time_eval():
        c_l2, c_l3, c_l3l4, monomials_l3, monomials_l4 = build_custom_kernels()
    d = d.at[:, 4:9, 4:9].set(quaternion_to_wigner_d_l2_einsum(q, c_l2))

    if lmax < 3:
        return d

    if lmax >= 4:
        d_l3, d_l4 = quaternion_to_wigner_d_l3l4_batched(q, c_l3l4, monomials_l4)
        d = d.at[:, 16:25, 16:25].set(d_l4)
    else:
        d_l3 = quaternion_to_wigner_d_matmul(q, 3, c_l3, monomials_l3)

    d = d.at[:, 9:16, 9:16].set(d_l3)

    if lmax < 5:
        return d

    d_re, d_im = wigner_d_matrix_real(q, lmin=5, lmax=lmax)
    d_range = wigner_d_pair_to_real(d_re, d_im, jd_list, lmin=5, lmax=lmax)

    d = d.at[:, 25:, 25:].set(d_range)

    return d


def axis_angle_wigner_hybrid(
    edge_distance_vec: jax.Array,
    lmax: int,
    jd_list: list[jax.Array],
    key: jax.Array | None = None,
) -> jax.Array:
    """Compute Wigner D using hybrid approach (optimal method per l).

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion einsum tensor contraction
    - l=3,4: Batched quaternion matmul (single kernel dispatch)
    - l>=5: Ra/Rb polynomial

    Combines the edge->Y and gamma rotations into a single quaternion before
    computing the Wigner D, avoiding the overhead of computing two separate
    Wigner D matrices and multiplying them.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        jd_list: J_y generators (angular momentum operators) in the |ℓ,m⟩ basis
        key: optional PRNGKey. If provided, used to sample random γ (roll).
             If None, γ is set to zeros (deterministic).

    Returns:
        wigner_d: Wigner D of shape (N, size, size) and size = (lmax+1)^2.
    """

    xyz = edge_distance_vec / (
        jnp.linalg.norm(edge_distance_vec + 1e-12, axis=-1, keepdims=True) + 1e-12
    )
    xyz = jnp.clip(xyz, -1.0, 1.0)

    q_edge_to_y = quaternion_edge_to_y_stable(xyz)

    if key is not None:
        half_gamma = jax.random.uniform(key, shape=(xyz.shape[0],)) * jnp.pi
        q_edge_to_y = gamma_quaternion_multiply(half_gamma, q_edge_to_y)

    wigner = wigner_d_from_quaternion_hybrid(q_edge_to_y, jd_list, lmax)

    return wigner
