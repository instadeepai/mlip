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

import math

import pytest
from pydantic import ValidationError

from mlip.data import DatasetInfo, GraphDatasetBuilderConfig
from mlip.data.dataset_info import (
    check_compatibility_of_ds_info,
    compute_dataset_info_from_graphs,
)


class TestDatasetInfoValidation:
    def test_valid_construction(self):
        info = DatasetInfo(
            atomic_energies_map={1: -1.0, 8: -2.0},
            graph_cutoff_angstrom=5.0,
        )
        assert info.avg_num_neighbors == 1.0
        assert info.scaling_mean == 0.0
        assert info.scaling_stdev == 1.0

    def test_invalid_atomic_number(self):
        with pytest.raises(ValidationError, match="not a valid atomic number"):
            DatasetInfo(
                atomic_energies_map={999: -1.0},
                graph_cutoff_angstrom=5.0,
            )

    def test_allowed_atomic_numbers_sorted(self):
        info = DatasetInfo(
            atomic_energies_map={8: -2.0, 1: -1.0, 6: -1.5},
            graph_cutoff_angstrom=5.0,
        )
        assert info.allowed_atomic_numbers == [1, 6, 8]

    def test_available_total_charges(self):
        info = DatasetInfo(
            atomic_energies_map={1: -1.0, 8: -2.0},
            graph_cutoff_angstrom=5.0,
            total_charge_set={0, 1, -1},
        )
        assert info.available_total_charges == [-1, 0, 1]
        info_no_charges = DatasetInfo(
            atomic_energies_map={1: -1.0, 8: -2.0},
            graph_cutoff_angstrom=5.0,
        )
        assert info_no_charges.available_total_charges == []

    def test_str_contains_all_fields(self):
        info = DatasetInfo(
            atomic_energies_map={1: -1.0, 8: -2.0},
            graph_cutoff_angstrom=5.0,
        )
        s = str(info)
        assert "Atomic Energies" in s
        assert "Graph Cutoff" in s
        assert "5.0" in s
        assert "Avg Num Neighbors" in s
        assert "Avg R Min" in s
        assert "Scaling Mean" in s
        assert "Scaling Stdev" in s

    def test_str_with_none_values(self):
        info = DatasetInfo(
            atomic_energies_map={1: -1.0},
            graph_cutoff_angstrom=5.0,
        )
        assert info.avg_r_min_angstrom is None
        s = str(info)
        assert "Avg R Min" in s
        assert "None" in s

    def test_list_form_atomic_energies(self):
        info = DatasetInfo(
            atomic_energies_map=[{1: -1.0}, {8: -2.0}],
            dataset_name=["ds_a", "ds_b"],
            graph_cutoff_angstrom=5.0,
        )
        assert isinstance(info.atomic_energies_map, list)
        assert len(info.atomic_energies_map) == 2

    def test_mixed_list_and_scalar_raises(self):
        with pytest.raises(ValidationError, match="all lists or all scalars"):
            DatasetInfo(
                atomic_energies_map=[{1: -1.0}, {8: -2.0}],
                dataset_name="single",
                graph_cutoff_angstrom=5.0,
            )

    def test_mismatched_list_lengths_raises(self):
        with pytest.raises(ValidationError, match="same length"):
            DatasetInfo(
                atomic_energies_map=[{1: -1.0}, {8: -2.0}],
                dataset_name=["ds_a", "ds_b", "ds_c"],
                graph_cutoff_angstrom=5.0,
            )

    def test_duplicate_dataset_names_raises(self):
        with pytest.raises(ValidationError, match="duplicates"):
            DatasetInfo(
                atomic_energies_map=[{1: -1.0}, {8: -2.0}],
                dataset_name=["ds_a", "ds_a"],
                graph_cutoff_angstrom=5.0,
            )


class TestComputeDatasetInfoFromGraphs:
    def test_with_precomputed_values(self, salt_graph):
        info = compute_dataset_info_from_graphs(
            [salt_graph],
            graph_cutoff_angstrom=3.0,
            avg_num_neighbors=4.0,
            avg_r_min_angstrom=1.5,
        )
        assert info.graph_cutoff_angstrom == 3.0
        assert info.avg_num_neighbors == 4.0
        assert info.avg_r_min_angstrom == 1.5
        assert isinstance(info.atomic_energies_map, dict)

    def test_computes_from_graphs(self, salt_graph):
        info = compute_dataset_info_from_graphs(
            [salt_graph],
            graph_cutoff_angstrom=3.0,
        )
        assert info.avg_num_neighbors == 8.0  # Two atoms with pbcs True
        assert info.avg_r_min_angstrom == pytest.approx(
            math.sqrt(1.5**2 + 1.6**2 + 1.5**2),
            abs=1e-3,  # Due to MIC
        )
        assert info.graph_cutoff_angstrom == 3.0


class TestCheckCompatibilityOfDsInfo:
    def test_matching_cutoff_no_error(self):
        config = GraphDatasetBuilderConfig(graph_cutoff_angstrom=5.0)
        info = DatasetInfo(
            atomic_energies_map={1: -1.0},
            graph_cutoff_angstrom=5.0,
        )
        check_compatibility_of_ds_info(config, info)

    def test_mismatching_cutoff_raises(self):
        config = GraphDatasetBuilderConfig(graph_cutoff_angstrom=5.0)
        info = DatasetInfo(
            atomic_energies_map={1: -1.0},
            graph_cutoff_angstrom=3.0,
        )
        with pytest.raises(ValueError, match="inconsistent cutoff"):
            check_compatibility_of_ds_info(config, info)
