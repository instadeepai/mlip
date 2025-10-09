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


def test_mace_outputs_correct_forces_and_energies_for_example_graph(
    setup_system_and_mace_model,
):
    _, graph, mace_apply_fun, mace_ff = setup_system_and_mace_model
    params = mace_ff.params

    num_nodes = graph.nodes.positions.shape[0]
    num_edges = graph.senders.shape[0]
    batched_graph = next(
        dynamically_batch(
            [graph], n_node=num_nodes + 3, n_edge=num_edges + 5, n_graph=2
        )
    )
    result = mace_apply_fun(params, batched_graph)

    assert list(result.energy) == pytest.approx([-0.1127705, 0], abs=1e-3)
    assert result.forces.shape == (num_nodes + 3, 3)
    assert np.all(result.forces[num_nodes:] == 0.0)
    assert result.stress is None
    assert result.stress_virial is None

    expected_forces = np.array(
        [
            [0.04929154, -0.02132669, 0.01178984],
            [0.00278281, -0.0041921, -0.00598245],
            [-0.02778297, -0.04706174, -0.00615619],
            [-0.00388734, 0.0063119, 0.00766478],
            [-0.0389901, -0.04678847, -0.01366944],
            [-0.02027022, 0.03045949, 0.05074739],
            [0.00960352, 0.03628635, -0.04992145],
            [-0.02313459, 0.02951152, 0.05012186],
            [0.06018497, -0.01368529, 0.00932314],
            [-0.00779762, 0.03048502, -0.05391748],
        ]
    )

    assert np.allclose(np.array(result.forces[:num_nodes]), expected_forces, atol=1e-3)
