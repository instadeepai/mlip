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
from mlip.graph.graph import GraphEdges
from mlip.simulation.fep.alchemical_graph import AlchemicalGraph
from mlip.simulation.fep.alchemical_graph.masking import (
    get_alchemical_edge_mask,
    prune_edges_with_mask,
)


@pytest.fixture(scope="module")
def alchemical_graph(setup_system) -> AlchemicalGraph:
    _, graph = setup_system
    graph = graph.replace(edges=GraphEdges(shifts=None, displ_fun=None))
    graph = graph.replace_globals(cell=None)
    return AlchemicalGraph.from_graph(graph)


@pytest.fixture(scope="module")
def graph_a(setup_system) -> AlchemicalGraph:
    """Graph for state A (contains all edges)."""
    _, graph = setup_system
    graph = next(  # Graph must be batched with dummy graph
        dynamically_batch(
            [graph],
            n_node=graph.nodes.positions.shape[0] + 1,
            n_edge=graph.senders.shape[0] + 1,
            n_graph=2,
        )
    )

    graph = AlchemicalGraph.from_graph(graph)
    graph = graph.replace_globals(
        alchemical_lambda=np.asarray([[0.5, 0.5], [0.0, 0.0]]),
        alchemical_atom_indices=np.array([0, 1, 2]),
    )
    return graph


@pytest.fixture(scope="module")
def graph_b(graph_a: AlchemicalGraph) -> AlchemicalGraph:
    """Graph for state B (contains no boundary edges)."""
    sr_mask, _ = get_alchemical_edge_mask(graph_a)
    return prune_edges_with_mask(graph_a, ~sr_mask)
