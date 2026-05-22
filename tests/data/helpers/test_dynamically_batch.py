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

import pytest

from mlip.data.helpers.dynamically_batch import dynamically_batch


class TestDynamicallyBatch:
    def test_n_graph_less_than_2_raises(self):
        with pytest.raises(ValueError, match="greater or equal to `2`"):
            list(dynamically_batch(iter([]), n_node=10, n_edge=10, n_graph=1))

    def test_single_graph_fits_in_batch(self, make_customizable_graph):
        g = make_customizable_graph(2, 2)
        batches = list(dynamically_batch([g], n_node=10, n_edge=10, n_graph=2))
        assert len(batches) == 1

    def test_multiple_graphs_one_batch(self, make_customizable_graph):
        graphs = [make_customizable_graph(2, 2) for _ in range(3)]
        batches = list(dynamically_batch(graphs, n_node=20, n_edge=20, n_graph=10))
        assert len(batches) == 1

    def test_multiple_graphs_two_batches(self, make_customizable_graph):
        graphs = [make_customizable_graph(3, 3) for _ in range(4)]
        # n_node=7 means valid_batch_size n_node=6, so 2 graphs per batch
        batches = list(dynamically_batch(graphs, n_node=7, n_edge=100, n_graph=10))
        assert len(batches) == 2

    def test_oversized_graph_raises_runtime_error(self, make_customizable_graph):
        small = make_customizable_graph(2, 2)
        big = make_customizable_graph(20, 20)
        with pytest.raises(RuntimeError, match="Found graph bigger than batch size"):
            list(dynamically_batch([small, big], n_node=10, n_edge=10, n_graph=5))

    def test_oversized_graph_yields_accumulated_first(self, make_customizable_graph):
        small = make_customizable_graph(2, 2)
        big = make_customizable_graph(20, 20)
        batches = []
        with pytest.raises(RuntimeError):
            for b in dynamically_batch([small, big], n_node=10, n_edge=10, n_graph=5):
                batches.append(b)
        assert len(batches) == 1  # the small graph was yielded first

    def test_skip_last_batch(self, make_customizable_graph):
        graphs = [make_customizable_graph(3, 3) for _ in range(3)]
        # 2 fit in first batch, 1 left over
        batches_with = list(
            dynamically_batch(
                graphs, n_node=7, n_edge=100, n_graph=10, skip_last_batch=False
            )
        )
        batches_without = list(
            dynamically_batch(
                graphs, n_node=7, n_edge=100, n_graph=10, skip_last_batch=True
            )
        )
        assert len(batches_with) == len(batches_without) + 1

    def test_empty_iterator(self):
        batches = list(dynamically_batch(iter([]), n_node=10, n_edge=10, n_graph=2))
        assert batches == []
