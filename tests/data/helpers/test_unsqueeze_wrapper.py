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

import jax
import jraph
import numpy as np
import pytest

from mlip.data.helpers.data_prefetching import UnsqueezeGraphDatasetWrapper


class _FakeGraphDataset:
    """Minimal stand-in for GraphDataset to avoid heavy real dataset creation."""

    def __init__(self, batches: list[jraph.GraphsTuple]):
        self._batches = batches

    def __iter__(self):
        yield from self._batches

    def __len__(self):
        return len(self._batches)

    def subset(self, i: int):
        return _FakeGraphDataset(self._batches[:i])


@pytest.fixture
def fake_dataset():
    def _make_dummy_graph(n_nodes: int = 3, n_edges: int = 4) -> jraph.GraphsTuple:
        """Create a minimal GraphsTuple with numpy arrays."""
        return jraph.GraphsTuple(
            nodes=np.random.randn(n_nodes, 2).astype(np.float32),
            edges=np.random.randn(n_edges, 1).astype(np.float32),
            senders=np.arange(n_edges, dtype=np.int32) % n_nodes,
            receivers=np.flip(np.arange(n_edges, dtype=np.int32) % n_nodes),
            n_node=np.array([n_nodes], dtype=np.int32),
            n_edge=np.array([n_edges], dtype=np.int32),
            globals=np.array([0.0], dtype=np.float32),
        )

    batches = [_make_dummy_graph() for _ in range(5)]
    return _FakeGraphDataset(batches)


@pytest.fixture
def wrapper(fake_dataset):
    return UnsqueezeGraphDatasetWrapper(fake_dataset)


def test_unsqueeze_adds_leading_dimension(wrapper, fake_dataset):
    """Verify that each leaf in the batch gets a prepended (1, ...) dimension."""
    original_batch = next(iter(fake_dataset))
    wrapped_batch = next(iter(wrapper))

    original_leaves, _ = jax.tree.flatten(original_batch)
    wrapped_leaves, _ = jax.tree.flatten(wrapped_batch)

    for orig, wrapped in zip(original_leaves, wrapped_leaves):
        assert wrapped.shape == (1, *orig.shape)


def test_len_delegates_to_inner_dataset(wrapper, fake_dataset):
    """Verify len(wrapper) == len(dataset)."""
    assert len(wrapper) == len(fake_dataset)
    assert len(wrapper) == 5


def test_subset_returns_wrapped_subset(wrapper):
    """Verify subset() delegation returns a new wrapper with correct length."""
    sub = wrapper.subset(3)
    assert isinstance(sub, UnsqueezeGraphDatasetWrapper)
    assert len(sub) == 3


def test_output_stays_numpy(wrapper):
    """Verify that output arrays remain numpy and are not converted to JAX arrays."""
    batch = next(iter(wrapper))
    leaves, _ = jax.tree.flatten(batch)
    for leaf in leaves:
        assert isinstance(leaf, np.ndarray), f"Expected numpy array, got {type(leaf)}"


def test_iteration_count(wrapper, fake_dataset):
    """Verify the wrapper yields the same number of batches as the dataset."""
    wrapper_count = sum(1 for _ in wrapper)
    dataset_count = sum(1 for _ in fake_dataset)
    assert wrapper_count == dataset_count
    assert wrapper_count == 5
