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

from typing import Callable, Sequence

import e3j
import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer

from mlip.graph import Graph
from mlip.models.blocks import MLP
from mlip.models.options import GradientScaledKernelInit


class ReadoutBlock(nn.Module):
    """Readout block that can be used in multiple models.

    The block only has a single head, see
    :class:`~mlip.models.readout.MultiHeadReadoutBlock` for the multi-head version.

    The readout consists of a couple of linear layers (either equivariant or not) and
    activations in between (if requested).

    Attributes:
        features: The feature output dimensions for each linear layer as a sequence of
                  either integers of irreps in the equivariant case.
        activation: The activation function to use. Default is `None` which means
                    no activation is applied.
        mlp_kernel_init: The kernel initialization method to use if using a
                         non-equivariant MLP. Default is `None`.
        use_equiv: Whether to use an equivariant linear layer and assume the features
                   are irreps. Default is false.
    """

    features: Sequence[int | e3nn.Irreps]
    activation: Callable | None = None
    mlp_kernel_init: Initializer | GradientScaledKernelInit | None = None
    use_equiv: bool = False

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        """Call function for this block.

        Args:
            graph: The input graph. Should have its features in
                   `graph.nodes.features["latent"]`.

        Returns:
            graph: The output graph with the resulting node readout outputs in
                   `graph.nodes.features["outputs"]`.
        """
        node_feats = graph.nodes.features["latent"]

        # Without equivariance
        if not self.use_equiv:
            assert self.mlp_kernel_init is not None
            node_feats = MLP(
                layer_sizes=(node_feats.shape[-1],) + tuple(self.features),
                activation=self.activation,
                kernel_init=self.mlp_kernel_init,
                use_bias=True,
            )(node_feats)
            return graph.update_node_features(outputs=node_feats)

        # With equivariance
        layer_irreps = [node_feats.irreps, *self.features]
        for idx, (src_, tgt_) in enumerate(zip(layer_irreps[:-1], layer_irreps[1:])):
            src = e3nn.Irreps(src_)
            tgt = e3nn.Irreps(tgt_)
            linear = e3j.linen.Linear(
                source_irreps=str(src),
                target_irreps=str(tgt.regroup()),
                layout="E3NN",
                kernel_init="FAN_IN",
                rescale_gradients=True,
            )
            node_feats = e3nn.IrrepsArray(tgt, linear(node_feats.array))
            if self.activation is not None and idx != len(self.features) - 1:
                node_feats = e3nn.gate(
                    node_feats, even_act=self.activation, even_gate_act=None
                )

        return graph.update_node_features(outputs=node_feats)


class MultiHeadReadoutBlock(nn.Module):
    """Multi-head readout block that can be used in multiple models.

    This block shares its attributes mostly with the single-head
    :class:`~mlip.models.readout.ReadoutBlock`, however, it has the additional
    `num_heads` attribute.

    Applies the readout block independently across multiple heads.

    Attributes:
        num_heads: The number of readout heads.
        features: The feature output dimensions for each linear layer as a sequence of
                  either integers of irreps in the equivariant case.
        activation: The activation function to use. Default is `None` which means
                    no activation is applied.
        mlp_kernel_init: The kernel initialization method to use if using a
            non-equivariant MLP. Default is `None`.
        use_equiv: Whether to use an equivariant linear layer and assume the features
                   are irreps. Default is false.
    """

    num_heads: int
    features: Sequence[int | e3nn.Irreps]
    activation: Callable | None = None
    mlp_kernel_init: Initializer | GradientScaledKernelInit | None = None
    use_equiv: bool = False

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        """Call function for this block.

        Note that the outputs of this block will have the dimension
        `[num_nodes, num_heads, num_final_readout_layer_output_features]`.

        Args:
            graph: The input graph. Should have its features in
                   `graph.nodes.features["latent"]`.

        Returns:
            graph: The output graph with the resulting node readout outputs in
                   `graph.nodes.features["outputs"]`.
        """

        node_outputs = []
        for _head_idx in range(self.num_heads):
            _out = ReadoutBlock(
                features=self.features,
                activation=self.activation,
                mlp_kernel_init=self.mlp_kernel_init,
                use_equiv=self.use_equiv,
            )(graph).nodes.features["outputs"]
            node_outputs.append(_out)

        if self.use_equiv:
            node_outputs = e3nn.stack(node_outputs, axis=1)
        else:
            node_outputs = jnp.stack(node_outputs, axis=1)

        return graph.update_node_features(outputs=node_outputs)


def select_head(graph: Graph) -> Graph:
    """Select a readout head per node based on `graph.globals.dataset_idx`.

    Expects the graph to have "outputs" in `graph.nodes.features`
    with shape `[num_nodes, num_heads, num_predictions]`. Selects one head per node and
    updates "outputs" to shape `[num_nodes, num_predictions]`.

    In a batch, different graphs may target different heads. The per-graph
    `dataset_idx` is broadcast to per-node using `graph.n_node`.

    When `dataset_idx` is `None`, head 0 is used.

    Args:
        graph: The input graph with "outputs" feature and
            `globals.dataset_idx`.

    Returns:
        The graph with "outputs" of shape `[num_nodes, num_predictions]`.
    """
    node_outputs = graph.nodes.features["outputs"]

    num_heads = node_outputs.shape[1]
    dataset_idx = graph.globals.dataset_idx
    if dataset_idx is None or num_heads == 1:
        node_outputs = node_outputs[:, 0]
    else:

        def _check_bounds(idx, nh):
            if not (idx < nh).all():
                raise ValueError(
                    f"dataset_idx contains values >= num_heads ({nh}): {idx}"
                )

        jax.debug.callback(_check_bounds, dataset_idx, num_heads)
        total_num_nodes = node_outputs.shape[0]
        node_head_idx = jnp.repeat(
            dataset_idx, graph.n_node, total_repeat_length=total_num_nodes
        )
        node_outputs = node_outputs[jnp.arange(total_num_nodes), node_head_idx]

    return graph.update_node_features(outputs=node_outputs)
