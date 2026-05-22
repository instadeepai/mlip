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

from mlip.data.helpers.dummy_init_graph import get_dummy_graph_for_model_init
from mlip.graph import Graph


class TestGetDummyGraphForModelInit:
    def test_returns_graph(self):
        graph = get_dummy_graph_for_model_init()
        assert isinstance(graph, Graph)

    def test_n_node(self):
        graph = get_dummy_graph_for_model_init()
        np.testing.assert_array_equal(graph.n_node, [1])

    def test_n_edge(self):
        graph = get_dummy_graph_for_model_init()
        np.testing.assert_array_equal(graph.n_edge, [1])

    def test_positions_shape(self):
        graph = get_dummy_graph_for_model_init()
        assert graph.nodes.positions.shape == (1, 3)

    def test_atomic_numbers(self):
        graph = get_dummy_graph_for_model_init()
        np.testing.assert_array_equal(graph.nodes.atomic_numbers, [1])
