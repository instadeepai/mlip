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

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mlip.graph.batching_helpers import batch_graphs
from mlip.models.readout import MultiHeadReadoutBlock, ReadoutBlock, select_head


@pytest.mark.parametrize(
    "input_dim,features,activation,use_equiv",
    [
        (32, (16, 8, 1), nn.silu, False),
        (32, (16, 8, 1), nn.silu, True),
        (64, (16, 4), nn.tanh, False),
        (64, (16, 4), nn.tanh, True),
        (16, (8, 4), None, False),
        (16, (8, 4), None, True),
        (16, (1,), nn.swish, False),
        (16, (1,), nn.swish, True),
    ],
)
def test_readout_block_outputs_correct_dimensions(
    setup_system, input_dim, features, activation, use_equiv
):
    _, graph = setup_system

    num_nodes = sum(graph.n_node)
    key = jax.random.PRNGKey(0)

    node_feats = jnp.ones((num_nodes, input_dim))
    if use_equiv:
        input_dim = e3nn.Irreps(f"{input_dim}x0e")
        node_feats = e3nn.IrrepsArray(input_dim, node_feats)

    graph_in = graph.update_node_features(latent=node_feats)

    if use_equiv:
        features_as_int = features
        features = tuple(e3nn.Irreps(f"{f}x0e") for f in features_as_int)

    readout_block = ReadoutBlock(
        features=features,
        activation=activation,
        mlp_kernel_init=nn.initializers.lecun_normal(),
        use_equiv=use_equiv,
    )

    params = readout_block.init(key, graph_in)
    graph_out = readout_block.apply(params, graph_in)

    if use_equiv:
        assert graph_out.nodes.features["outputs"].array.shape == (
            num_nodes,
            features_as_int[-1],
        )
    else:
        assert graph_out.nodes.features["outputs"].shape == (
            num_nodes,
            features[-1],
        )


@pytest.mark.parametrize(
    "num_heads, out_dim, use_equiv",
    [(1, 2, True), (1, 1, False), (2, 3, True), (4, 5, False)],
)
def test_multi_head_readout_works_correctly(
    setup_system, num_heads, out_dim, use_equiv
):
    _, graph = setup_system

    num_nodes = sum(graph.n_node)
    key = jax.random.PRNGKey(0)

    node_feats = jnp.ones((num_nodes, 32))
    if use_equiv:
        input_dim = e3nn.Irreps("32x0e")
        node_feats = e3nn.IrrepsArray(input_dim, node_feats)

    graph_in = graph.update_node_features(latent=node_feats)

    features = (16, 8, out_dim)
    if use_equiv:
        features = (
            e3nn.Irreps("16x0e"),
            e3nn.Irreps("8x0e"),
            e3nn.Irreps(f"{out_dim}x0e"),
        )

    readout_block = MultiHeadReadoutBlock(
        num_heads=num_heads,
        features=features,
        activation=nn.silu,
        mlp_kernel_init=nn.initializers.lecun_normal(),
        use_equiv=use_equiv,
    )

    params = readout_block.init(key, graph_in)
    graph_out = readout_block.apply(params, graph_in)

    expected_shape = (num_nodes, num_heads, out_dim)
    if use_equiv:
        assert graph_out.nodes.features["outputs"].array.shape == expected_shape
    else:
        assert graph_out.nodes.features["outputs"].shape == expected_shape


class TestSelectHead:
    """Tests for the select_head helper function."""

    def test_none_dataset_idx_selects_head_zero(self, make_customizable_graph):
        """When dataset_idx is None, head 0 is selected for all nodes."""
        graph = make_customizable_graph(3, 3)
        node_outputs = jnp.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])
        graph = graph.update_node_features(outputs=node_outputs)

        result = select_head(graph)

        expected = jnp.array([[1.0], [3.0], [5.0]])
        assert jnp.array_equal(result.nodes.features["outputs"], expected)

    def test_single_graph_selects_correct_head(self, make_customizable_graph):
        """A single graph with dataset_idx=1 selects head 1."""
        graph = make_customizable_graph(2, 2)
        graph = graph.replace_globals(dataset_idx=np.array([1]))
        node_outputs = jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        graph = graph.update_node_features(outputs=node_outputs)

        result = select_head(graph)

        expected = jnp.array([[2.0], [4.0]])
        assert jnp.array_equal(result.nodes.features["outputs"], expected)

    def test_batched_graphs_with_mixed_heads(self, make_customizable_graph):
        """Three graphs in a batch each select a different head."""
        g0 = make_customizable_graph(3, 3).replace_globals(dataset_idx=np.array([0]))
        g1 = make_customizable_graph(2, 2).replace_globals(dataset_idx=np.array([1]))
        g2 = make_customizable_graph(1, 1).replace_globals(dataset_idx=np.array([2]))
        graph = batch_graphs([g0, g1, g2])

        node_outputs = jnp.array([
            [[10.0], [20.0], [30.0]],  # graph 0, node 0
            [[40.0], [50.0], [60.0]],  # graph 0, node 1
            [[70.0], [80.0], [90.0]],  # graph 0, node 2
            [[11.0], [22.0], [33.0]],  # graph 1, node 0
            [[44.0], [55.0], [66.0]],  # graph 1, node 1
            [[77.0], [88.0], [99.0]],  # graph 2, node 0
        ])
        graph = graph.update_node_features(outputs=node_outputs)

        result = select_head(graph)

        expected = jnp.array([[10.0], [40.0], [70.0], [22.0], [55.0], [99.0]])
        assert jnp.array_equal(result.nodes.features["outputs"], expected)

    def test_irreps_array_is_preserved(self, make_customizable_graph):
        """IrrepsArray with scalar and vector features preserves irreps metadata."""
        graph = make_customizable_graph(2, 2)
        graph = graph.replace_globals(dataset_idx=np.array([1]))
        # 1x0e + 1x1o = 4 components: [scalar, vx, vy, vz]
        array = jnp.array([
            [[1.0, 0.1, 0.2, 0.3], [2.0, 0.4, 0.5, 0.6]],
            [[3.0, 0.7, 0.8, 0.9], [4.0, 1.0, 1.1, 1.2]],
        ])
        node_outputs = e3nn.IrrepsArray("1x0e + 1x1o", array)
        graph = graph.update_node_features(outputs=node_outputs)

        result = select_head(graph)

        node_out = result.nodes.features["outputs"]
        assert isinstance(node_out, e3nn.IrrepsArray)
        assert node_out.irreps == e3nn.Irreps("1x0e + 1x1o")
        expected = jnp.array([[2.0, 0.4, 0.5, 0.6], [4.0, 1.0, 1.1, 1.2]])
        assert jnp.array_equal(node_out.array, expected)

    def test_oob_dataset_idx_raises(self, make_customizable_graph):
        """dataset_idx >= num_heads raises a ValueError."""
        graph = make_customizable_graph(2, 2)
        graph = graph.replace_globals(dataset_idx=np.array([3]))  # only 2 heads
        node_outputs = jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        graph = graph.update_node_features(outputs=node_outputs)

        with pytest.raises(ValueError, match="dataset_idx"):
            select_head(graph)
