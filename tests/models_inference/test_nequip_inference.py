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

import numpy as np
import pytest

from mlip.data.helpers.dynamically_batch import dynamically_batch


def test_nequip_outputs_correct_forces_and_energies_for_example_graph(
    setup_system_and_nequip_model,
):
    _, graph, nequip_apply_fun, nequip_ff = setup_system_and_nequip_model
    params = nequip_ff.params

    num_nodes = graph.nodes.positions.shape[0]
    num_edges = graph.senders.shape[0]
    batched_graph = next(
        dynamically_batch(
            [graph], n_node=num_nodes + 3, n_edge=num_edges + 5, n_graph=2
        )
    )
    result = nequip_apply_fun(params, batched_graph)

    assert list(result.energy) == pytest.approx([0.4042114, 0], abs=1e-1)
    assert result.forces.shape == (num_nodes + 3, 3)
    assert np.all(result.forces[num_nodes:] == 0.0)
    assert result.stress is None
    assert result.stress_virial is None

    expected_forces = np.array(
        [
            [-0.00056573, -0.00188175, 0.00301255],
            [-0.02069944, 0.04880672, 0.01910987],
            [0.00027544, -0.00164071, 0.00320481],
            [0.00869875, -0.00734535, -0.02690949],
            [0.03248089, 0.05118233, 0.01248641],
            [0.01254146, -0.02847607, -0.05386348],
            [-0.01817782, -0.03477869, 0.04697158],
            [0.0287421, -0.02302093, -0.05010285],
            [-0.05753617, 0.0211088, -0.00838852],
            [0.01424053, -0.02395435, 0.05447913],
        ]
    )

    assert np.allclose(np.array(result.forces[:num_nodes]), expected_forces, atol=1e-3)
