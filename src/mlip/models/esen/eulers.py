# Code converted from https://github.com/facebookresearch/fairchem
# Some parts of the code may remain identical, distributed under:
#
#     MIT License- Copyright (c) Meta Platforms, Inc. and affiliates.
#
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

from functools import partial

import jax
import jax.numpy as jnp


@jax.custom_jvp
def safe_acos(x: jax.Array, eps: float = 1e-7) -> jax.Array:
    x_c = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    return jnp.arccos(x_c)


@safe_acos.defjvp
def _safe_acos_jvp(primals, tangents):
    x, eps = primals
    (x_dot, eps_dot) = tangents

    x_c = jnp.clip(x, -1.0 + eps, 1.0 - eps)
    y = jnp.arccos(x_c)

    denom = jnp.sqrt(jnp.maximum(1.0 - x_c * x_c, eps))
    y_dot = -x_dot / denom

    return y, y_dot


@jax.custom_jvp
def safe_atan2(y: jax.Array, x: jax.Array, eps: float = 1e-7) -> jax.Array:
    return jnp.arctan2(y, x)


@safe_atan2.defjvp
def _safe_atan2_jvp(primals, tangents):
    y, x, eps = primals
    y_dot, x_dot, eps_dot = tangents

    out = jnp.arctan2(y, x)

    denom = jnp.maximum(x * x + y * y, eps)

    out_dot = (x / denom) * y_dot + (-y / denom) * x_dot

    return out, out_dot


def init_edge_rot_euler_angles(
    edge_distance_vec: jax.Array,
    key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Args:
        edge_distance_vec: [E, 3] — edge direction vectors.
        key: optional PRNGKey. If provided, used to sample random γ (roll).
             If None, γ is set to zeros (deterministic).

    Returns:
        (gamma, beta, alpha): each [E], Euler angles in radians.
    """

    xyz = edge_distance_vec / (
        jnp.linalg.norm(edge_distance_vec + 1e-12, axis=-1, keepdims=True) + 1e-12
    )
    xyz = jnp.clip(xyz, -1.0, 1.0)

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    beta = safe_acos(y)

    alpha = safe_atan2(x, z)

    if key is not None:
        gamma = jax.random.uniform(key, shape=alpha.shape) * (2 * jnp.pi)
    else:
        gamma = jnp.zeros_like(alpha)

    return -gamma, -beta, -alpha


@partial(jax.jit, static_argnames=("start_l_max", "end_l_max"))
def eulers_to_wigner(
    eulers: tuple[jax.Array, jax.Array, jax.Array],
    start_l_max: int,
    end_l_max: int,
    jd: list[jax.Array],
) -> jax.Array:
    """
    Returns Wigner: [E, K, K], K = (end_l_max+1)^2 - (start_l_max)^2
    Block-diagonal concat of Wigner-D(l) for l in [start_l_max..end_l_max].
    """
    alpha, beta, gamma = eulers

    e_shape = jnp.broadcast_shapes(alpha.shape, beta.shape, gamma.shape)
    assert len(e_shape) == 1, "alpha/beta/gamma must be 1D (broadcastable to [E])"
    e = e_shape[0]

    sizes = [2 * l_number + 1 for l_number in range(start_l_max, end_l_max + 1)]
    k = (end_l_max + 1) ** 2 - (start_l_max) ** 2

    w = jnp.zeros((e, k, k), dtype=alpha.dtype)

    start = 0
    for l_number, s in zip(range(start_l_max, end_l_max + 1), sizes):
        blk = wigner_d_block(l_number, alpha, beta, gamma, jd)
        w = w.at[:, start : start + s, start : start + s].set(blk)
        start += s

    return w


@partial(jax.jit, static_argnames=("lv",))
def wigner_d_block(
    lv: int,
    alpha: jax.Array,
    beta: jax.Array,
    gamma: jax.Array,
    jd: tuple[jax.Array, ...],
) -> jax.Array:
    """
    Returns the Wigner-D block for degree lv.
    alpha,beta,gamma broadcast to the same shape [...]=[E], result [..., s, s].
    """
    alpha, beta, gamma = jnp.broadcast_arrays(alpha, beta, gamma)
    j = jnp.asarray(jd[lv], dtype=alpha.dtype)

    x_a = _z_rot_mat(alpha, lv)
    x_b = _z_rot_mat(beta, lv)
    x_c = _z_rot_mat(gamma, lv)

    return (((x_a @ j) @ x_b) @ j) @ x_c


def _z_rot_mat(angle: jax.Array, lv: int) -> jax.Array:
    """
    Build the Z-rotation block for level `lv`.
    angle: [...], returns [..., s, s] with s = 2*lv+1
    Diagonal = cos(m*angle), anti-diagonal = sin(m*angle), m = lv..-lv
    """
    s = 2 * lv + 1
    freqs = jnp.arange(lv, -lv - 1, -1, dtype=angle.dtype)
    ang = angle[..., None] * freqs
    cosv = jnp.cos(ang)
    sinv = jnp.sin(ang)

    eye = jnp.eye(s, dtype=angle.dtype)
    anti = eye[:, ::-1]

    diag = jnp.einsum("...i,ij->...ij", cosv, eye)
    anti_mat = jnp.einsum("...i,ij->...ij", sinv, anti)
    return diag + anti_mat
