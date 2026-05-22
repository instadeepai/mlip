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

from typing import Callable, TypeAlias

from mlip.data import ChemicalSystem, ChemicalSystemsReader
from mlip.data.graph_dataset import GraphDataset
from mlip.data.helpers.combined_graph_dataset import CombinedGraphDataset
from mlip.data.helpers.data_prefetching import PrefetchIterator
from mlip.graph import Graph

GraphDatasetLike: TypeAlias = GraphDataset | PrefetchIterator | CombinedGraphDataset
FlatReadersDict: TypeAlias = dict[
    str, ChemicalSystemsReader | list[ChemicalSystemsReader]
]
NestedReadersDict: TypeAlias = dict[
    str, dict[str, ChemicalSystemsReader | list[ChemicalSystemsReader]]
]
SystemsPreprocessingFunction: TypeAlias = Callable[
    [list[ChemicalSystem]], list[ChemicalSystem]
]
GraphPostProcessingFunction: TypeAlias = Callable[[Graph], Graph]
