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


def test_visnet_outputs_correct_forces_and_energies_for_example_graph(
    setup_system_and_visnet_model,
):
    _, graph, visnet_apply_fun, visnet_ff = setup_system_and_visnet_model
    params = visnet_ff.params

    num_nodes = graph.nodes.positions.shape[0]
    num_edges = graph.senders.shape[0]
    batched_graph = next(
        dynamically_batch(
            [graph], n_node=num_nodes + 3, n_edge=num_edges + 5, n_graph=2
        )
    )
    result = visnet_apply_fun(params, batched_graph)

    assert list(result.energy) == pytest.approx([-0.9996586, 0], abs=1e-1)
    assert result.forces.shape == (num_nodes + 3, 3)
    assert np.all(result.forces[num_nodes:] == 0.0)
    assert result.stress is None
    assert result.stress_forces is None

    expected_forces = np.array(
        [
            [-0.29491305, 0.15716457, -0.05265914],
            [0.05681321, -0.12324873, -0.06801297],
            [0.14221777, 0.3031985, 0.04899645],
            [-0.01085154, -0.01417974, 0.06714264],
            [0.10993972, 0.08165835, 0.03590977],
            [0.08709274, -0.09262571, -0.11566924],
            [0.01208293, -0.09849587, 0.11827305],
            [0.03674538, -0.10947306, -0.12749831],
            [-0.13895479, -0.00171597, -0.02191941],
            [0.00067334, -0.10228296, 0.1155162],
        ]
    )

    assert np.allclose(np.array(result.forces[:num_nodes]), expected_forces, atol=5e-3)
