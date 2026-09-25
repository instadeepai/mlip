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

from mlip.data.helpers.data_prefetching import ParallelGraphDataset


class _StubGraphDataset:
    def number_of_graphs(self) -> int:
        return 123

    def number_of_nodes(self) -> int:
        return 456


def test_parallel_graph_dataset_number_of_graphs_and_nodes_delegate():
    parallel_dataset = ParallelGraphDataset(_StubGraphDataset(), num_parallel=4)

    assert parallel_dataset.number_of_graphs() == 123
    assert parallel_dataset.number_of_nodes() == 456
