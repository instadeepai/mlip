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

from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

from mlip.models.predictors.energy_heads import (
    coulomb_energy_computation_head,
    standard_energy_computation_head,
)


def test_standard_energy_computation_head():
    """Test that the standard head sums per-node energies."""
    graph = MagicMock()
    graph.nodes.features = {"energy": jnp.array([1.0, 2.0, 4.0])}
    graph.n_node = jnp.array([2, 1])

    result = standard_energy_computation_head(graph)

    assert result.shape == (2,)
    assert jnp.allclose(result, jnp.array([3.0, 4.0]))


@pytest.mark.parametrize(
    "node_energies,partial_charges,distance,expected",
    [
        ([0.0, 0.0], [1.0, -1.0], 4.0, -3.5936859),
        ([-1.0, -2.0], [1.0, -1.0], 4.0, -6.5936859),
        ([0.0, 0.0], [1.0, -1.0], 6.0, -2.399941),
        ([0.0, 0.0], [2.0, -1.0], 4.0, -7.1873717),
    ],
)
def test_coulomb_energy_computation_head(
    node_energies, partial_charges, distance, expected
):
    graph = MagicMock()
    graph.nodes.features = {
        "energy": jnp.array(node_energies),
        "partial_charges": jnp.array(partial_charges),
    }
    graph.n_node = jnp.array([2])
    graph.senders_long_range = jnp.array([0, 1])
    graph.receivers_long_range = jnp.array([1, 0])
    graph.n_edge_long_range = jnp.array([2])
    graph.edges_long_range.features = {}
    graph.long_range_edge_vectors.return_value = jnp.array([
        [distance, 0.0, 0.0],
        [-distance, 0.0, 0.0],
    ])

    result = coulomb_energy_computation_head(graph)
    assert result.shape == (1,)
    assert jnp.allclose(result, jnp.array([expected]), atol=1e-6)
