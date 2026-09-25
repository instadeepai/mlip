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

from mlip.data.graph_dataset import GraphDataset
from mlip.data.helpers.exceptions import GraphsDiscardedError


def _make_dataset(make_graph, n_graphs=10, shuffle=False, **kwargs):
    """Helper to create a GraphDataset with small graphs."""
    graphs = [make_graph(3, 4) for _ in range(n_graphs)]
    return GraphDataset(
        graphs=graphs,
        batch_size=4,
        max_n_node=10,
        max_n_edge=20,
        shuffle=shuffle,
        **kwargs,
    )


class TestGraphDatasetConstruction:
    def test_valid_construction(self, make_customizable_graph):
        ds = _make_dataset(make_customizable_graph)
        assert ds.number_of_graphs() == 10

    def test_oversized_graphs_discarded_warning(self, make_customizable_graph):
        graphs = [make_customizable_graph(3, 4), make_customizable_graph(100, 200)]
        ds = GraphDataset(
            graphs=graphs,
            batch_size=4,
            max_n_node=10,
            max_n_edge=20,
            shuffle=False,
            raise_exc_if_graphs_discarded=False,
        )
        assert ds.number_of_graphs() == 1

    def test_oversized_graphs_raise_exc(self, make_customizable_graph):
        graphs = [make_customizable_graph(3, 4), make_customizable_graph(100, 200)]
        with pytest.raises(GraphsDiscardedError):
            GraphDataset(
                graphs=graphs,
                batch_size=4,
                max_n_node=10,
                max_n_edge=20,
                shuffle=False,
                raise_exc_if_graphs_discarded=True,
            )


class TestGraphDatasetLen:
    def test_len_matches_iteration(self, make_customizable_graph):
        ds = _make_dataset(make_customizable_graph)
        assert len(ds) == sum(1 for _ in ds)


class TestGraphDatasetSubset:
    def test_subset_slice(self, make_customizable_graph):
        ds = _make_dataset(make_customizable_graph, n_graphs=10)
        sub = ds.subset(slice(0, 5))
        assert sub.number_of_graphs() == 5

    def test_subset_int(self, make_customizable_graph):
        ds = _make_dataset(make_customizable_graph, n_graphs=10)
        sub = ds.subset(3)
        assert sub.number_of_graphs() == 3

    def test_subset_list(self, make_customizable_graph):
        ds = _make_dataset(make_customizable_graph, n_graphs=10)
        sub = ds.subset([0, 2, 4])
        assert sub.number_of_graphs() == 3

    def test_subset_float(self, make_customizable_graph):
        ds = _make_dataset(make_customizable_graph, n_graphs=10)
        sub = ds.subset(0.5)
        assert sub.number_of_graphs() == 5

    def test_subset_invalid_type(self, make_customizable_graph):
        ds = _make_dataset(make_customizable_graph)
        with pytest.raises(TypeError, match="incorrect type"):
            ds.subset("invalid")


class TestGraphDatasetCounters:
    def test_number_of_graphs_correct(self, make_customizable_graph):
        ds = _make_dataset(make_customizable_graph, n_graphs=7)
        assert ds.number_of_graphs() == 7

    def test_number_of_nodes_correct(self, make_customizable_graph):
        graphs = [make_customizable_graph(n, 4) for n in (2, 3, 5)]
        ds = GraphDataset(
            graphs=graphs,
            batch_size=4,
            max_n_node=10,
            max_n_edge=20,
            shuffle=False,
        )
        assert ds.number_of_nodes() == 2 + 3 + 5


class TestGraphDatasetHomogenize:
    """Tests for the `homogenize` flag on `GraphDataset.__init__`."""

    def test_homogenize_true_fills_missing_fields_and_batches(
        self, make_customizable_graph
    ):
        """With `homogenize=True`, graphs missing an optional field get filled
        (NaN for globals, zero for edges) and the dataset iterates without
        error."""
        g_with_stress = make_customizable_graph(3, 4)
        g_without_stress = make_customizable_graph(3, 4).replace_globals(stress=None)
        g_without_shifts = make_customizable_graph(3, 4).replace_edges(shifts=None)

        ds = GraphDataset(
            graphs=[g_with_stress, g_without_stress, g_without_shifts],
            batch_size=3,
            max_n_node=10,
            max_n_edge=20,
            shuffle=False,
            homogenize=True,
        )
        batch = next(iter(ds))

        # The graph without stress should have NaN in its stress row; the
        # others keep their original (0.0) stress.
        stress = np.asarray(batch.globals.stress)
        assert np.all(stress[0] == 0.0)
        assert np.all(np.isnan(stress[1]))
        assert np.all(stress[2] == 0.0)

        # Edge fields are zero-filled rather than NaN-filled (shifts=None
        # already means "zero" for a non-periodic graph).
        assert batch.edges.shifts is not None
        assert np.all(np.asarray(batch.edges.shifts) == 0.0)

    @pytest.mark.parametrize("missing", ["stress", "shifts"])
    def test_homogenize_false_raises_on_heterogeneous_graphs(
        self, make_customizable_graph, missing
    ):
        """With `homogenize=False`, heterogeneous optional fields produce a
        clear `ValueError` instead of a cryptic tree-map failure later inside
        `batch_graphs`."""
        base_graph = make_customizable_graph(3, 3)
        if missing == "stress":
            modified_graph = base_graph.replace_globals(stress=None)
        elif missing == "shifts":
            modified_graph = base_graph.replace_edges(shifts=None)
        else:
            raise ValueError

        with pytest.raises(ValueError, match="heterogeneous optional fields") as exc:
            GraphDataset(
                graphs=[base_graph, modified_graph],
                batch_size=2,
                max_n_node=10,
                max_n_edge=20,
                shuffle=False,
                homogenize=False,
            )
        msg = str(exc.value)
        assert "homogenize=True" in msg
        if missing == "stress":
            assert "globals.stress" in msg
        elif missing == "shifts":
            assert "edges.shifts" in msg
        else:
            raise ValueError

    def test_homogenize_false_is_noop_on_homogeneous_graphs(
        self, make_customizable_graph
    ):
        """With `homogenize=False` and uniformly-shaped graphs, the validator
        is a cheap no-op and the dataset iterates as normal."""
        ds = _make_dataset(make_customizable_graph, n_graphs=4, homogenize=False)
        batches = list(ds)
        assert len(batches) >= 1
