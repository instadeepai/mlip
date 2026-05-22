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

import jax.numpy as jnp
import numpy as np
import pytest

from mlip.data.dataset_info import DatasetInfo
from mlip.graph.batching_helpers import batch_graphs
from mlip.models.inference_context import (
    InferenceContext,
    apply_inference_context_to_graph,
)


@pytest.fixture
def multi_ds_info():
    return DatasetInfo(
        dataset_name=["ds_a", "ds_b", "ds_c"],
        atomic_energies_map=[{1: -1.0}, {1: -2.0}, {1: -3.0}],
        graph_cutoff_angstrom=5.0,
    )


@pytest.fixture
def single_ds_info():
    return DatasetInfo(
        atomic_energies_map={1: -1.0},
        graph_cutoff_angstrom=5.0,
    )


class TestInferenceContextResolve:
    def test_name_resolves_to_idx(self, multi_ds_info):
        ctx = InferenceContext(dataset_name="ds_b").resolve(multi_ds_info)
        assert ctx.dataset_name == "ds_b"
        assert ctx.dataset_idx == 1

    def test_idx_resolves_to_name(self, multi_ds_info):
        ctx = InferenceContext(dataset_idx=2).resolve(multi_ds_info)
        assert ctx.dataset_name == "ds_c"
        assert ctx.dataset_idx == 2

    def test_name_and_idx_agree_kept(self, multi_ds_info):
        ctx = InferenceContext(dataset_name="ds_a", dataset_idx=0).resolve(
            multi_ds_info
        )
        assert ctx.dataset_name == "ds_a"
        assert ctx.dataset_idx == 0

    def test_name_and_idx_disagree_raises(self, multi_ds_info):
        with pytest.raises(ValueError, match="disagree"):
            InferenceContext(dataset_name="ds_a", dataset_idx=2).resolve(multi_ds_info)

    def test_unknown_name_raises(self, multi_ds_info):
        with pytest.raises(ValueError, match="Unknown dataset_name"):
            InferenceContext(dataset_name="nope").resolve(multi_ds_info)

    def test_out_of_range_idx_raises(self, multi_ds_info):
        with pytest.raises(ValueError, match="out of range"):
            InferenceContext(dataset_idx=99).resolve(multi_ds_info)

    def test_name_on_single_dataset_model_raises(self, single_ds_info):
        with pytest.raises(ValueError, match="does not have dataset names"):
            InferenceContext(dataset_name="anything").resolve(single_ds_info)

    def test_empty_context_single_dataset_passthrough(self, single_ds_info):
        ctx = InferenceContext().resolve(single_ds_info)
        assert ctx.dataset_name is None
        assert ctx.dataset_idx is None

    def test_empty_context_multi_dataset_passthrough(self, multi_ds_info):
        """Empty context on a multi-dataset model is still valid — the caller
        may intend to set dataset_idx directly on the graph elsewhere."""
        ctx = InferenceContext().resolve(multi_ds_info)
        assert ctx.dataset_name is None
        assert ctx.dataset_idx is None

    def test_conditioning_fields_preserved(self, multi_ds_info):
        ctx = InferenceContext(
            dataset_name="ds_a", charge=1, spin_multiplicity=2
        ).resolve(multi_ds_info)
        assert ctx.charge == 1
        assert ctx.spin_multiplicity == 2


class TestApplyInferenceContextToGraph:
    def test_empty_context_returns_graph_unchanged(self, make_customizable_graph):
        graph = make_customizable_graph(3, 3)
        out = apply_inference_context_to_graph(graph, InferenceContext())
        assert out is graph

    def test_dataset_idx_is_broadcast(self, make_customizable_graph):
        g1 = make_customizable_graph(3, 3)
        g2 = make_customizable_graph(2, 2)
        batched = batch_graphs([g1, g2])  # two graphs
        out = apply_inference_context_to_graph(batched, InferenceContext(dataset_idx=1))
        expected = jnp.full(batched.num_graphs, 1, dtype=jnp.int32)
        assert np.array_equal(np.asarray(out.globals.dataset_idx), np.asarray(expected))

    def test_charge_and_spin_multiplicity_tagged(self, make_customizable_graph):
        graph = make_customizable_graph(3, 3)
        out = apply_inference_context_to_graph(
            graph, InferenceContext(charge=2, spin_multiplicity=3)
        )
        assert np.array_equal(np.asarray(out.globals.charge), np.array([2]))
        assert np.array_equal(np.asarray(out.globals.spin_multiplicity), np.array([3]))

    def test_unset_fields_are_not_tagged(self, make_customizable_graph):
        graph = make_customizable_graph(3, 3).replace_globals(charge=None)
        out = apply_inference_context_to_graph(graph, InferenceContext(dataset_idx=0))
        # `charge` was not set on the context, so the graph's charge stays None.
        assert out.globals.charge is None
