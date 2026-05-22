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

from copy import deepcopy

import pytest

from mlip.data.graph_dataset_builder import GraphDataset
from mlip.data.helpers.data_prefetching import ParallelGraphDataset
from mlip.graph.batching_helpers import batch_graphs, pad_with_graphs
from mlip.graph.mask_helpers import get_graph_padding_mask, get_node_padding_mask


def test_graph_batching(setup_system):
    atoms, graph = setup_system

    _graph_mask = get_graph_padding_mask(graph)
    assert _graph_mask.shape == (1,)
    assert _graph_mask[0]

    _node_mask = get_node_padding_mask(graph)
    assert _node_mask.shape == (len(atoms),)
    assert _node_mask.all()

    batched_graph = batch_graphs([graph, deepcopy(graph), deepcopy(graph)])

    assert batched_graph.nodes.positions.shape == (len(atoms) * 3, 3)
    assert batched_graph.nodes.atomic_numbers.shape == (len(atoms) * 3,)
    assert batched_graph.globals.cell.shape == (3, 3, 3)
    assert batched_graph.globals.energy.shape == (3,)
    assert batched_graph.senders.shape == (len(graph.senders) * 3,)

    _graph_mask = get_graph_padding_mask(batched_graph)
    assert _graph_mask.shape == (3,)
    assert list(_graph_mask) == [True, True, False]  # last one is considered padding

    _node_mask = get_node_padding_mask(batched_graph)
    assert _node_mask.shape == (len(atoms) * 3,)
    # Nodes of last graph are considered padding
    assert list(_node_mask) == [True] * len(atoms) * 2 + [False] * len(atoms)


def test_graph_padding(setup_system):
    _, graph = setup_system

    # Original graph has 10 nodes and 68 edges.
    padded_graph = pad_with_graphs(graph, n_node=17, n_edge=84, n_graph=4)

    assert padded_graph.nodes.positions.shape == (17, 3)
    assert padded_graph.nodes.atomic_numbers.shape == (17,)
    assert padded_graph.senders.shape == (84,)
    assert padded_graph.receivers.shape == (84,)

    assert list(padded_graph.n_node) == [10, 7, 0, 0]
    assert list(padded_graph.n_edge) == [68, 16, 0, 0]

    assert list(get_graph_padding_mask(padded_graph)) == [True, False, False, False]
    assert list(get_node_padding_mask(padded_graph)) == [True] * 10 + [False] * 7


def test_error_is_raised_if_mask_computed_on_stacked_graph(setup_system):
    _, graph = setup_system

    dataset = GraphDataset([graph] * 5, batch_size=2, max_n_node=25, max_n_edge=145)
    parallel_dataset = ParallelGraphDataset(dataset, 2)

    assert len(parallel_dataset) == 3

    for stacked_batch in parallel_dataset:
        with pytest.raises(ValueError):
            get_graph_padding_mask(stacked_batch)
        with pytest.raises(ValueError):
            get_node_padding_mask(stacked_batch)
