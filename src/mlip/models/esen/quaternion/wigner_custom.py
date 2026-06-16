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

"""Custom Wigner D computation kernels for l = 1, 2, 3, 4.

Defines kernels used by `wigner_hybrid.py` to accelerate angular momentum blocks.

This module contains specialized, optimized kernels for computing Wigner D matrices
for small angular momentum values, using coefficients in `custom_coefficients.npz`.

Primary kernels:
    - l=1: quaternion_to_rotation_matrix - direct quaternion to 3x3 rotation.
    - l=2: quaternion_to_wigner_d_l2_einsum - degree-4 tensor contraction via einsum.
    - l=3 (standalone): quaternion_to_wigner_d_matmul - polynomial coefficient matmul.
    - l=3,4 (batched): quaternion_to_wigner_d_l3l4_batched - single matmul for both.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_COEFFICIENTS_FILE = Path(__file__).parent / "custom_coefficients.npz"


def _generate_monomials(n_vars: int, total_degree: int) -> list[tuple[int, ...]]:
    """Generate all monomials of given degree in n_vars variables.

    Returns a list of tuples (a, b, c, d) representing w^a * x^b * y^c * z^d
    where a + b + c + d = total_degree.
    """
    monomials: list[tuple[int, ...]] = []

    def generate(remaining_vars: int, remaining_deg: int, current: list[int]) -> None:
        if remaining_vars == 1:
            monomials.append(tuple(current + [remaining_deg]))
            return
        for i in range(remaining_deg + 1):
            generate(remaining_vars - 1, remaining_deg - i, current + [i])

    generate(n_vars, total_degree, [])
    return monomials


def _load_and_decompress_coefficients() -> tuple[jax.Array, jax.Array, jax.Array]:
    """Load and decompress precomputed coefficients from file.

    Coefficients are stored in palette-compressed format for smaller file size.
    Each matrix is stored as (palette, indices, shape) and decompressed here.
    """
    raw = np.load(_COEFFICIENTS_FILE)

    result = []
    for ell in [2, 3, 4]:
        key = f"C_l{ell}"
        palette = raw[f"{key}_palette"]
        indices = np.asarray(raw[f"{key}_indices"], dtype=np.int64)
        shape = tuple(raw[f"{key}_shape"].tolist())
        result.append(jnp.asarray(palette[indices].reshape(shape)))

    return result[0], result[1], result[2]


def build_custom_kernels() -> tuple[
    jax.Array, jax.Array, jax.Array, list[tuple[int, ...]], list[tuple[int, ...]]
]:
    """Precompute required coefficient tensors for l=2,3,4 Wigner D kernels.

    Returns:
        (c_l2, c_l3, c_l3l4, monomials_l3, monomials_l4), where:
            c_l2: Coefficient tensor of shape (5, 5, 4, 4, 4, 4) for l=2 einsum.
            c_l3: Coefficient matrix of shape (49, 84) for standalone l=3 matmul.
            c_l3l4: Combined coefficient matrix of shape (130, 165) for
                batched l=3,4 computation.
            monomials_l3: List of 84 degree-6 monomial tuples (device-independent).
            monomials_l4: List of 165 degree-8 monomial tuples (device-independent).
    """

    c_l2, c_l3, c_l4 = _load_and_decompress_coefficients()

    monomials_l3 = _generate_monomials(4, 6)
    monomials_l4 = _generate_monomials(4, 8)

    mono8_to_idx = {m: i for i, m in enumerate(monomials_l4)}
    c_l3_lifted = jnp.zeros((c_l3.shape[0], len(monomials_l4)), dtype=c_l3.dtype)

    for j, (a, b, c, d) in enumerate(monomials_l3):
        lifted = [
            (a + 2, b, c, d),
            (a, b + 2, c, d),
            (a, b, c + 2, d),
            (a, b, c, d + 2),
        ]
        for mono8 in lifted:
            idx = mono8_to_idx[mono8]
            c_l3_lifted = c_l3_lifted.at[:, idx].add(c_l3[:, j])

    c_l3l4 = jnp.concat([c_l3_lifted, c_l4], axis=0)

    return c_l2, c_l3, c_l3l4, monomials_l3, monomials_l4


def quaternion_to_rotation_matrix(q: jax.Array) -> jax.Array:
    """Convert quaternion directly to 3x3 rotation matrix (l=1 Wigner D).

    This is the recommended method for l=1 as it uses pure polynomial
    arithmetic without requiring axis-angle extraction or matrix operations.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention

    Returns:
        Rotation matrices of shape (N, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot = jnp.stack(
        [
            jnp.stack([1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            jnp.stack([2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)], axis=-1),
            jnp.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)], axis=-1),
        ],
        axis=-2,
    )

    return rot


def quaternion_to_wigner_d_l2_einsum(q: jax.Array, c_l2: jax.Array) -> jax.Array:
    """Convert quaternion to 5x5 l=2 Wigner D matrix using einsum tensor contraction.

    Expresses D as a tensor contraction:
        D[i,j] = C[i,j,a,b,c,d] * q[a] * q[b] * q[c] * q[d]

    where C is a precomputed (5, 5, 4, 4, 4, 4) coefficient tensor.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        c_l2: Coefficient tensor of shape (5, 5, 4, 4, 4, 4)

    Returns:
        Wigner D matrices of shape (N, 5, 5) for l=2
    """
    c_l2 = jnp.asarray(c_l2, dtype=q.dtype)

    q2 = q[:, :, None] * q[:, None, :]
    q4 = q2[:, :, :, None, None] * q2[:, None, None, :, :]

    d = jnp.einsum("nabcd,ijabcd->nij", q4, c_l2)

    return d


def _build_wigner_d_matrix_l3l4(
    q: jax.Array, c: jax.Array, monomials: list[tuple[int, ...]], max_power: int
) -> jax.Array:
    """Build Wigner D matrix for l=3, l=4 or l=3&4 batched computation.

    Uses optimal multiplication tree: p[i] = p[i//2] * p[(i+1)//2]
    which minimizes the number of multiplications.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        c: Coefficient matrix of shape ((2*ell+1)^2, n_monomials) or (130, 165)
        monomials: List of monomial exponent tuples
        max_power: Maximum power to compute

    Returns:
        Wigner D matrices.
    """

    c = jnp.asarray(c, dtype=q.dtype)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    def powers_for_var(var: jax.Array) -> dict[int, jax.Array]:
        p = {0: jnp.ones_like(var), 1: var}
        for i in range(2, max_power + 1):
            p[i] = p[i // 2] * p[(i + 1) // 2]
        return p

    powers = {
        0: powers_for_var(w),
        1: powers_for_var(x),
        2: powers_for_var(y),
        3: powers_for_var(z),
    }

    m = jnp.stack(
        [
            powers[0][a] * powers[1][b] * powers[2][_c] * powers[3][d]
            for a, b, _c, d in monomials
        ],
        axis=1,
    )

    d_flat = m @ c.T

    return d_flat


def quaternion_to_wigner_d_matmul(
    q: jax.Array, ell: int, c: jax.Array, monomials: list[tuple[int, ...]]
) -> jax.Array:
    """Matmul-based Wigner D computation for l=3 or l=4.

    Computes D = M @ C^T where:
    - M[n, k] = product of quaternion powers for monomial k
    - C[ij, k] = coefficient of monomial k in D[i,j]

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        ell: Angular momentum (3 or 4)
        c: Coefficient matrix of shape ((2*ell+1)^2, n_monomials)
        monomials: List of monomial exponent tuples

    Returns:
        Wigner D matrices of shape (N, 7, 7) for l=3 or (N, 9, 9) for l=4
    """

    d_flat = _build_wigner_d_matrix_l3l4(q, c, monomials, 2 * ell)

    size = 2 * ell + 1

    return d_flat.reshape(q.shape[0], size, size)


def quaternion_to_wigner_d_l3l4_batched(
    q: jax.Array, c_combined: jax.Array, monomials_l4: list[tuple[int, ...]]
) -> tuple[jax.Array, jax.Array]:
    """Compute l=3 and l=4 Wigner D matrices in a single matmul.

    Builds degree-8 monomials once and multiplies by the combined (130, 165)
    coefficient matrix to get both D_l3 and D_l4 from one kernel dispatch.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        c_combined: Combined coefficient matrix of shape (130, 165)
        monomials_l4: List of 165 degree-8 monomial tuples

    Returns:
        Tuple of (D_l3, D_l4) with shapes (N, 7, 7) and (N, 9, 9)
    """

    d_flat = _build_wigner_d_matrix_l3l4(q, c_combined, monomials_l4, 8)

    n = q.shape[0]
    d_l3 = d_flat[:, :49].reshape(n, 7, 7)
    d_l4 = d_flat[:, 49:].reshape(n, 9, 9)

    return d_l3, d_l4
