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

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3nn_jax import IrrepsArray, spherical_harmonics

import mlip
from mlip.models.esen.eulers import eulers_to_wigner, init_edge_rot_euler_angles
from mlip.models.esen.quaternion.wigner_hybrid import axis_angle_wigner_hybrid

JD_FILE_PATH = Path(mlip.__file__).parent / "models" / "esen" / "Jd.npz"


@jax.jit(static_argnums=(1,))
def quaternion_wigner_d(edge_vectors, l_max):
    jd_data = np.load(JD_FILE_PATH, allow_pickle=True)
    jd_list = list(jd_data["Jd"])
    with jax.ensure_compile_time_eval():
        jd_buffers = [
            jnp.asarray(jd_list[l_number], dtype=edge_vectors.dtype)
            for l_number in range(l_max + 1)
        ]

    return axis_angle_wigner_hybrid(
        edge_vectors,
        l_max,
        jd_buffers,
        key=jax.random.PRNGKey(42),
    )


@jax.jit(static_argnums=(1,))
def euler_wigner_d(edge_vectors, l_max):
    jd_data = np.load(JD_FILE_PATH, allow_pickle=True)
    jd_list = list(jd_data["Jd"])
    with jax.ensure_compile_time_eval():
        jd_buffers = [
            jnp.asarray(jd_list[l_number], dtype=edge_vectors.dtype)
            for l_number in range(l_max + 1)
        ]

    euler_angles = init_edge_rot_euler_angles(edge_vectors, key=jax.random.PRNGKey(42))
    return eulers_to_wigner(
        eulers=euler_angles, start_l_max=0, end_l_max=l_max, jd=jd_buffers
    )


@pytest.mark.parametrize("l_max", [3, 6])  # l_max=6 covers all l cases.
def test_quaternion_matches_eulers(setup_system, l_max):
    _, graph = setup_system
    edge_vectors = graph.edge_vectors()

    wigner_quaternion = quaternion_wigner_d(edge_vectors, l_max)
    wigner_eulers = euler_wigner_d(edge_vectors, l_max)

    sph = spherical_harmonics(
        range(l_max + 1), IrrepsArray("1o", edge_vectors), normalize=True
    ).array

    rotated_quaternion = jnp.einsum("bji,bi->bj", wigner_eulers, sph)
    rotated_eulers = jnp.einsum("bji,bi->bj", wigner_quaternion, sph)

    assert np.allclose(
        np.array(rotated_quaternion), np.array(rotated_eulers), atol=1e-5
    )
