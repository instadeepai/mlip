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

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mlip.utils.jax_utils import scatter_sum, segment_sum

N_ITEMS = 20
N_FEATURES = 8
N_SEGMENTS = 4


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def segment_sum_inputs(key):
    """Data and kwargs for segment_sum tests."""
    k1, k2 = jax.random.split(key)
    data = jax.random.normal(k1, (N_ITEMS, N_FEATURES))
    segment_ids = jnp.sort(jax.random.randint(k2, (N_ITEMS,), 0, N_SEGMENTS))
    return data, segment_ids, N_SEGMENTS


@pytest.mark.parametrize("deterministic", [True, False])
def test_segment_sum_compiles_with_static_deterministic(
    segment_sum_inputs, deterministic
):
    """Verifies segment_sum compiles under jax.jit."""
    data, segment_ids, num_segments = segment_sum_inputs

    fn = jax.jit(
        partial(segment_sum, num_segments=num_segments, deterministic=deterministic)
    )
    result = fn(data, segment_ids)

    expected = jax.ops.segment_sum(data, segment_ids, num_segments=num_segments)
    np.testing.assert_allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize("deterministic", [True, False])
def test_segment_sum_matches_baseline(segment_sum_inputs, deterministic):
    """
    Verifies _deterministic_segment_sum gives the same values as jax.ops.segment_sum.
    """
    data, segment_ids, num_segments = segment_sum_inputs

    expected = jax.ops.segment_sum(data, segment_ids, num_segments=num_segments)
    actual = segment_sum(data, segment_ids, num_segments, deterministic=deterministic)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_segment_sum_high_dimensional_inputs(key):
    """
    Verifies that the segment_sum function handles high-dimensional tensors.
    """
    data = jax.random.normal(key, (100, 8, 4))
    segment_ids = jnp.sort(jax.random.randint(key, (100,), 0, 10))
    num_segments = 10

    result = segment_sum(
        data, segment_ids, num_segments=num_segments, deterministic=True
    )

    assert result.shape == (10, 8, 4)
    expected_0 = jnp.sum(data[segment_ids == 0], axis=0)
    np.testing.assert_allclose(result[0], expected_0, atol=1e-5)


def test_segment_sum_gradient_flow_irreps(segment_sum_inputs):
    """Ensures gradients are consistent for IrrepsArray inputs."""
    data, segment_ids, num_segments = segment_sum_inputs
    irreps = e3nn.Irreps(f"{N_FEATURES}x0e")

    def loss_fn(x_arr):
        ia = e3nn.IrrepsArray(irreps, x_arr)
        sums = segment_sum(
            ia, segment_ids, num_segments=num_segments, deterministic=True
        )
        return jnp.sum(sums.array**2)

    def baseline_loss_fn(x_arr):
        sums = segment_sum(
            x_arr, segment_ids, num_segments=num_segments, deterministic=False
        )
        return jnp.sum(sums**2)

    grad_det = jax.grad(loss_fn)(data)
    grad_base = jax.grad(baseline_loss_fn)(data)

    np.testing.assert_allclose(grad_det, grad_base, atol=1e-5)


@pytest.fixture
def scatter_sum_inputs(key):
    """Data and kwargs for scatter_sum tests."""
    k1, _ = jax.random.split(key)
    data = jax.random.normal(k1, (N_ITEMS, N_FEATURES))
    num_elements_per_segment = jnp.array([N_ITEMS // N_SEGMENTS] * N_SEGMENTS)
    return data, num_elements_per_segment


@pytest.mark.parametrize("deterministic", [True, False])
def test_scatter_sum_matches_baseline(scatter_sum_inputs, deterministic):
    """Verifies internal `scatter_sum` matches `e3nn.scatter_sum`."""
    data, num_elements_per_segment = scatter_sum_inputs
    expected = e3nn.scatter_sum(data, nel=num_elements_per_segment)
    actual = scatter_sum(data, num_elements_per_segment, deterministic=deterministic)
    np.testing.assert_allclose(actual, expected, atol=1e-5)
