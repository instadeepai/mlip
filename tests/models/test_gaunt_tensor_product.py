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

"""Rotation equivariance for `GauntTensorProduct` (Gaunt Tensor Product).

For a random rotation `R` we check
`GauntTensorProduct(R x1, R x2) = R GauntTensorProduct(x1, x2)`
via `IrrepsArray.transform_by_matrix`. Inputs use natural irreps `(l, (-1)**l)`.
"""

from __future__ import annotations

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import pytest

from mlip.models.gaunt_tensor_product import GauntTensorProduct

ATOL = 1e-5
L_MAX = 3

SOURCE_IRREPS = e3nn.Irreps([(1, (ell, (-1) ** ell)) for ell in range(L_MAX + 1)])
BATCH_DIM = (2, 3)


@pytest.fixture(autouse=True)
def gtp() -> GauntTensorProduct:
    return GauntTensorProduct(
        (str(SOURCE_IRREPS), str(SOURCE_IRREPS)),
        target=None,
        lmax=L_MAX,
    )


def max_abs_rotation_equivariance_error(x: jnp.ndarray, y: jnp.ndarray) -> float:
    return float(jnp.max(jnp.abs(x - y)))


def test_gtp_rotation_equivariance(gtp: GauntTensorProduct) -> None:
    key = jax.random.key(42)
    key_rot, key = jax.random.split(key, 2)
    key_1, key_2 = jax.random.split(key, 2)

    rotation = e3nn.rand_matrix(key_rot)
    x1 = e3nn.normal(irreps=SOURCE_IRREPS, key=key_1, leading_shape=BATCH_DIM)
    x2 = e3nn.normal(irreps=SOURCE_IRREPS, key=key_2, leading_shape=BATCH_DIM)
    out_a = gtp(
        x1.transform_by_matrix(rotation),
        x2.transform_by_matrix(rotation),
    )
    out_b = gtp(x1, x2).transform_by_matrix(rotation)
    max_abs = max_abs_rotation_equivariance_error(out_a.array, out_b.array)
    assert max_abs < ATOL, max_abs
