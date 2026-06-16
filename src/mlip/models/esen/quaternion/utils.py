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

import jax
import jax.numpy as jnp

# Blend region parameters for two-chart quaternion computation. Blend region
# is ey in [BLEND_START, BLEND_START + BLEND_WIDTH] = [-0.9, 0.9]
BLEND_START = -0.9
BLEND_WIDTH = 1.8


def _smooth_step_cinf(t: jax.Array) -> jax.Array:
    """C-infinity smooth step function based on the classic bump function.

    Uses f(x) = exp(-1/x) for x > 0 (0 otherwise), then:
    step(t) = f(t) / (f(t) + f(1-t)) = sigmoid((2t-1)/(t*(1-t)))

    Properties:
    - C-infinity smooth everywhere
    - All derivatives are exactly zero at t=0 and t=1
    - Values: f(0)=0, f(1)=1
    - Symmetric: f(t) + f(1-t) = 1

    Args:
        t: Input tensor, will be clamped to [0, 1]

    Returns:
        Smooth step values in [0, 1]
    """
    t_clamped = t.clip(0, 1)
    eps = jnp.finfo(t.dtype).eps

    numerator = 2.0 * t_clamped - 1.0
    denominator = t_clamped * (1.0 - t_clamped)
    denom_safe = denominator.clip(min=eps)
    arg = numerator / denom_safe
    result = jax.nn.sigmoid(arg)

    result = jnp.where(t_clamped < eps, jnp.zeros_like(result), result)
    result = jnp.where(t_clamped > 1 - eps, jnp.ones_like(result), result)

    return result


def gamma_quaternion_multiply(half_gamma: jax.Array, q2: jax.Array) -> jax.Array:
    """Multiply quaternion of gamma with q2.

    Uses Hamilton product convention: (w, x, y, z).

    Args:
        half_gamma: Half of rotation angles of shape (N,)
        q2: Second quaternion of shape (N, 4) or (4,)

    Returns:
        Product quaternion of shape (N, 4)
    """
    w1 = jnp.cos(half_gamma)
    y1 = jnp.sin(half_gamma)

    # x1 = z1 = 0
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    q2_y = jnp.stack([-y2, z2, w2, -x2], axis=-1)

    return w1[:, None] * q2 + y1[:, None] * q2_y


def quaternion_nlerp(q1: jax.Array, q2: jax.Array, t: jax.Array) -> jax.Array:
    """Normalized linear interpolation between quaternions.

    nlerp(q1, q2, t) = normalize((1-t) * q1 + t * q2)

    Args:
        q1: First quaternion, shape (..., 4)
        q2: Second quaternion, shape (..., 4)
        t: Interpolation parameter, shape (...)

    Returns:
        Interpolated quaternion, shape (..., 4)
    """
    dot = (q1 * q2).sum(axis=-1, keepdims=True)
    q1_aligned = jnp.where(dot < 0, -q1, q1)

    t_expanded = t[..., None] if t.ndim < q1.ndim else t
    result = (1.0 - t_expanded) * q1_aligned + t_expanded * q2
    result = result / (jnp.linalg.norm(result + 1e-12, axis=-1, keepdims=True) + 1e-12)

    return result


def _quaternion_chart(
    w: jax.Array, x: jax.Array, y: jax.Array, z: jax.Array
) -> jax.Array:
    """Combined quaternion for two-chart edge -> +Y.

    Standard quaternion: edge -> +Y directly. Singular at edge = -Y.
    Uses the half-vector formula:
        q = normalize(1 + ey, -ez, 0, ex)

    Alternative quaternion: edge -> +Y via -Y. Singular at edge = +Y.
    Path: edge -> -Y -> +Y (compose with 180 deg about X)

    Args:
        w, x, y, z: Constructed quaternion components

    Returns:
        Quaternions of shape (..., 4) in (w, x, y, z) convention
    """

    q = jnp.stack([w, x, y, z], axis=-1)
    q_sq = jnp.sum(q**2, axis=-1, keepdims=True)
    eps = jnp.finfo(w.dtype).eps
    norm = jnp.sqrt(jnp.clip(q_sq, min=eps))

    return q / norm


def quaternion_edge_to_y_stable(edge_vec: jax.Array) -> jax.Array:
    """Compute quaternion for edge -> +Y using two charts with NLERP blending.

    Uses two quaternion charts to avoid singularities:
    - Chart 1: q = normalize(1+ey, -ez, 0, ex) - singular at -Y
    - Chart 2: q = normalize(-ez, 1-ey, ex, 0) - singular at +Y

    NLERP blend in ey in [-0.9, 0.9]:
    - Uses Chart 2 when near -Y (stable there)
    - Uses Chart 1 when near +Y (stable there)
    - Smoothly interpolates in between

    Args:
        edge_vec: Edge vectors of shape (N, 3), assumed normalized

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) convention
    """
    ex = edge_vec[..., 0]
    ey = edge_vec[..., 1]
    ez = edge_vec[..., 2]

    zeros = jnp.zeros_like(ex)
    q_chart1 = _quaternion_chart(1.0 + ey, -ez, zeros, ex)
    q_chart2 = _quaternion_chart(-ez, 1.0 - ey, ex, zeros)

    t = (ey - BLEND_START) / BLEND_WIDTH
    t_smooth = _smooth_step_cinf(t)

    q = quaternion_nlerp(q_chart2, q_chart1, t_smooth)

    return q
