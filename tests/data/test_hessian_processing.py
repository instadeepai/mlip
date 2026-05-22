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

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.helpers.hessian_utils import pad_systems_hessians, process_graph_hessian
from mlip.graph import Graph
from mlip.graph.batching_helpers import batch_graphs

GRAPH_CUTOFF_ANGSTROM = 3.0


@pytest.fixture(scope="module")
def create_systems_list():
    # create two systems with different number of atoms
    systems_list = [
        ChemicalSystem(
            atomic_numbers=np.array([11, 17] * i),
            positions=np.array([[0.0, 0.0, 0.0], [1.5, 1.6, 1.6]] * i),
            hessian=np.random.sample((2 * i, 3, 2 * i, 3)),
            cell=np.array([[3.2, 0.1, 0.0], [0.0, 3.2, 0.0], [0.0, 0.0, 3.1]]),
            pbc=(True, True, True),
        )
        for i in range(1, 3)
    ]
    return systems_list


def test_pad_systems_hessians_logic(create_systems_list):
    systems = create_systems_list  # two systems with different number of atoms
    padded_systems = pad_systems_hessians(systems)
    max_system_size = max(len(system.atomic_numbers) for system in systems)

    assert padded_systems[0].hessian.shape == (
        len(systems[0].atomic_numbers),
        3,
        max_system_size,
        3,
    )
    assert padded_systems[1].hessian.shape == (
        len(systems[1].atomic_numbers),
        3,
        max_system_size,
        3,
    )


def test_process_graph_hessian(create_systems_list):
    num_hessian_rows = 4
    systems = create_systems_list
    padded_systems = pad_systems_hessians(systems)
    graphs = [
        Graph.from_chemical_system(s, GRAPH_CUTOFF_ANGSTROM) for s in padded_systems
    ]
    batched_graph = batch_graphs(graphs)

    processed_graph = process_graph_hessian(batched_graph, num_hessian_rows)

    total_graph_nodes = sum(batched_graph.n_node)
    assert processed_graph.nodes.hessian.shape == (
        total_graph_nodes,
        num_hessian_rows,
        3,
    )

    # Check if global sampling indices were stored in a
    # static shape (batch_size, num_hessian_rows)
    batch_size = batched_graph.num_graphs
    assert processed_graph.globals.sample_hessian_rows is not None
    assert processed_graph.globals.sample_hessian_rows.shape == (
        batch_size,
        num_hessian_rows,
    )
