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

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from mlip.data.helpers.atomic_energies import remove_e0s_from_graphs


@dataclass
class Globals:
    energy: float


@dataclass
class Nodes:
    atomic_numbers: list[int]


class DummyGraph:
    def __init__(self, energy, atomic_numbers):
        self.globals = Globals(energy)
        self.nodes = Nodes(atomic_numbers)

    def replace_globals(self, **kwargs):
        new_energy = kwargs.get("energy", self.globals.energy)
        return DummyGraph(new_energy, self.nodes.atomic_numbers)


def test_remove_e0s_basic():
    graphs = [
        DummyGraph(10.0, [1, 1]),
        DummyGraph(20.0, [2]),
    ]
    atomic_energies_map = {1: 1.0, 2: 2.0}

    result = remove_e0s_from_graphs(graphs, atomic_energies_map)

    assert [g.globals.energy for g in result] == pytest.approx([
        8.0,  # 10 - (1+1)
        18.0,  # 20 - 2
    ])


def test_original_graphs_not_modified():
    graphs = [DummyGraph(10.0, [1])]
    atomic_energies_map = {1: 1.0}

    result = remove_e0s_from_graphs(graphs, atomic_energies_map)

    # original unchanged
    assert graphs[0].globals.energy == 10.0

    # new graph updated
    assert result[0].globals.energy == 9.0

    # ensure new object
    assert result[0] is not graphs[0]


def test_missing_atomic_numbers_in_graphs():
    graphs = [DummyGraph(10.0, [1, 99])]
    atomic_energies_map = {1: 1.0}

    result = remove_e0s_from_graphs(graphs, atomic_energies_map)

    # 99 → 0.0
    assert result[0].globals.energy == pytest.approx(9.0)


def test_empty_graph_list():
    result = remove_e0s_from_graphs([], {1: 1.0})
    assert result == []


def test_graphs_processed_independently():
    graphs = [
        DummyGraph(10.0, [1]),
        DummyGraph(10.0, [2]),
    ]
    atomic_energies_map = {1: 1.0, 2: 5.0}

    result = remove_e0s_from_graphs(graphs, atomic_energies_map)

    assert result[0].globals.energy == pytest.approx(9.0)
    assert result[1].globals.energy == pytest.approx(5.0)


def test_convert_energy_to_formation_energy_helper_called_for_each_graph():
    graphs = [DummyGraph(10.0, [1]), DummyGraph(20.0, [2])]

    with patch(
        "mlip.data.helpers.atomic_energies._convert_energy_to_formation_energy"
    ) as mock_fn:
        mock_fn.side_effect = [1.0, 2.0]

        remove_e0s_from_graphs(graphs, {})

        assert mock_fn.call_count == 2
