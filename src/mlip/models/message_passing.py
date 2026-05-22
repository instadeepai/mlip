# Copyright (c) 2026 InstaDeep Ltd
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


import os
import warnings
from typing import Callable, Literal

import e3j
import e3nn_jax as e3nn
import flax.linen as nn
import jax.nn.initializers as initializers
import jax.numpy as jnp
from e3j.utils.options import Layout
from jax import Array

from mlip.graph import Graph
from mlip.models import options
from mlip.models.blocks import MLP, BetaSwish, SO3Convolution


class O3MessagePassingBlock(nn.Module):
    """Equivariant O(3) convolution used by MACE and NequIP.

    Performs a single message-passing step:

    * Transform RBF radial embeddings with an MLP to form edge scalars,
    * Project sender node features with a Linear layer,
    * Multiply sender features with spherical embeddings of edge vectors
      (Clebsch-Gordan tensor product) to form messages,
    * Reweight messages with the edge scalars,
    * Aggregate messages on receiver nodes.

    Attributes:
        source_irreps: Expected irreps of the input node features, used to
            validate the graph at call time.
        l_max: Maximum degree of for the harmonic embeddings of edge vectors.
        target_irreps: Target irreps for the output node features.
        num_rbf: Number of Bessel radial basis functions, used to validate the
            radial embedding shape.
        radial_mlp_hidden: Dimensions of hidden layers for the radial MLP. The input
            and output layer sizes are dictated by `num_rbf` and the message irreps
            respectively.
        radial_mlp_activation: Activation for hidden layers of the radial MLP.
        avg_num_neighbors: Average number of neighbours per atom, used to
            normalise the aggregated messages.
        layout: internal channel layout for the convolution block.
        radial_mlp_variance_scale: Variance scaling parameter for initialization of the
            radial MLP weights, passed to `jax.nn.initializers.variance_scaling`.
            The default is None, in which case fan-in initializers are used with
            gradient scaling (matching MACE behaviour).
        deterministic_scatter_ops: Whether to use deterministic scatter operations in
            the message passing convolution.
        normalize_activation: Whether to normalize the activation function with
            respect to the L2-norm for a normal Gaussian measure.
            See `e3nn.normalize_function()`. The default is True.
    """

    source_irreps: e3nn.Irreps
    l_max: int
    target_irreps: e3nn.Irreps
    num_rbf: int
    radial_mlp_hidden: list[int]
    radial_mlp_activation: Callable | options.Activation | Literal["beta_swish"]
    avg_num_neighbors: float
    layout: Layout = Layout.TRAILING_CHANNELS
    radial_mlp_variance_scale: float | None = None
    deterministic_scatter_ops: bool = False
    normalize_activation: bool = True

    @property
    def message_irreps(self) -> e3nn.Irreps:
        """Fitlered tensor product representation of node features with harmonics."""
        return e3nn.tensor_product(
            self.source_irreps,
            e3nn.Irreps.spherical_harmonics(self.l_max),
            filter_ir_out=self.target_irreps,
        )

    @property
    def num_channels(self) -> int:
        """GCD of source multiplicities to factorize as a channel axis."""
        return e3nn.Irreps(self.source_irreps).mul_gcd

    @property
    def _src(self) -> e3nn.Irreps:
        """Internal irreps for message-passing.

        The multiplicities may be factorized to a channel axis.
        """
        source_irreps = e3nn.Irreps(self.source_irreps)
        if self.layout == Layout.E3NN:
            # Note: path works but subject to numerical change.
            src = source_irreps
            msg = (
                "E3NN layout not numerically stable for now. The exact ordering of "
                "isomorphic irreps is subject to change to match other layouts, "
                "use TRAILING_CHANNELS instead."
            )
            if os.environ.get("MLIP_TESTING"):
                warnings.warn(msg)
            else:
                raise NotImplementedError(msg)
        else:
            gcd = source_irreps.mul_gcd
            src = e3nn.Irreps([(m // gcd, ir) for m, ir in source_irreps])
        return src

    @staticmethod
    def _fan_in_normal(scale: float) -> initializers.Initializer:
        return initializers.variance_scaling(scale, "fan_in", "normal")

    def setup(self):

        harmonics_irreps = e3nn.Irreps.spherical_harmonics(self.l_max)
        target_irreps = e3nn.Irreps(self.target_irreps)
        num_scalars = self.message_irreps.num_irreps

        # Radial MLP: accomodates for both MACE and Nequip V1 options.
        # - MACE uses FAN_IN_NORMAL + gradient scaling
        # - NequIP uses a fan-in variance scaling on hidden layers
        #   with scale 1.0 on the output layer.
        mlp_init = (
            self._fan_in_normal(self.radial_mlp_variance_scale)
            if self.radial_mlp_variance_scale is not None
            else options.GradientScaledKernelInit.FAN_IN_NORMAL
        )
        mlp_init_out = (
            self._fan_in_normal(1.0)
            if self.radial_mlp_variance_scale is not None
            else options.GradientScaledKernelInit.FAN_IN_NORMAL
        )
        mlp_activation = (
            # NOTE: Single shared BetaSwish across layers for v1 consistency.
            # Should we instead use separate activations per layer?
            BetaSwish()
            if self.radial_mlp_activation == "beta_swish"
            else self.radial_mlp_activation
        )
        self.radial_mlp = MLP(
            layer_sizes=[self.num_rbf, *self.radial_mlp_hidden, num_scalars],
            activation=mlp_activation,
            kernel_init=mlp_init,
            output_kernel_init=mlp_init_out,
            normalize_activation=self.normalize_activation,
            use_bias=False,
        )

        self.linear_in = e3j.linen.Linear(
            source_irreps=self.source_irreps,
            target_irreps=self.source_irreps,
            layout="E3NN",
            kernel_init="FAN_IN",
            rescale_gradients=True,
        )

        self.convolution_block = SO3Convolution(
            source_irreps=(self._src, harmonics_irreps),
            target_irreps=e3nn.Irreps([i for m, i in target_irreps]),
            avg_num_neighbors=self.avg_num_neighbors,
            deterministic_scatter_ops=self.deterministic_scatter_ops,
            layout=self.layout,
        )

        # Note: Nequip requires to rechunk non-grouped gate_irreps.
        self.linear_out = e3j.linen.Linear(
            source_irreps=self.message_irreps,
            target_irreps=self.target_irreps,
            layout="E3NN",
            kernel_init="FAN_IN",
            rescale_gradients=True,
        )

    def __call__(self, graph: Graph) -> Graph:
        node_feats = graph.nodes.features["latent"]
        spherical_embedding = graph.edges.features["spherical_embedding"]
        radial_embedding = graph.edges.features["radial_embedding"].array
        senders = graph.senders
        receivers = graph.receivers

        # Compute scalar radial embeddings
        edge_scalars = self.radial_mlp(radial_embedding)

        # Convert scalars to internal layout
        gcd = self.num_channels
        if self.layout == Layout.TRAILING_CHANNELS:
            edge_scalars = edge_scalars.reshape(edge_scalars.shape[0], -1, gcd)
        if self.layout == Layout.LEADING_CHANNELS:
            edge_scalars = edge_scalars.reshape(edge_scalars.shape[0], -1, gcd)
            edge_scalars = jnp.swapaxes(edge_scalars, -1, -2)

        # Linear in
        node_feats = e3nn.IrrepsArray(
            self.linear_in.target._to_e3nn(),
            self.linear_in(node_feats.array),
        )

        # Apply message passing:
        # Gather + tensor product + scalar mixing + scatter-add
        y_lm = spherical_embedding.array
        x_feats = self._cast_inputs(node_feats)
        y_feats = self.convolution_block(
            x_feats, y_lm, edge_scalars, senders, receivers
        )
        node_feats = self._cast_outputs(y_feats)

        # Linear out
        node_feats = e3nn.IrrepsArray(
            self.target_irreps,
            self.linear_out(node_feats.array),
        )
        return graph.update_node_features(latent=node_feats)

    def _cast_inputs(self, node_feats: e3nn.IrrepsArray) -> Array:
        """Cast E3NN input to internal layout."""
        layout = Layout.parse(self.layout)
        if layout == Layout.E3NN:
            x_feats = node_feats.array
        elif layout == Layout.LEADING_CHANNELS:
            x_feats = node_feats.mul_to_axis().array
        elif layout == Layout.TRAILING_CHANNELS:
            x_feats = node_feats.mul_to_axis().array
            x_feats = jnp.swapaxes(x_feats, -1, -2)
        else:
            raise RuntimeError(f"Unsupported layout {layout}")
        return x_feats

    def _cast_outputs(self, y_feats: Array) -> e3nn.IrrepsArray:
        """Cast internal layout back to E3NN."""
        layout = Layout.parse(self.layout)
        if layout == Layout.E3NN:
            return e3nn.IrrepsArray(self.message_irreps, y_feats)
        # Channels are in their own axis
        gcd = self.num_channels
        tgt = e3nn.Irreps([(mul // gcd, ir) for mul, ir in self.message_irreps])
        if layout == Layout.LEADING_CHANNELS:
            return e3nn.IrrepsArray(tgt, y_feats).axis_to_mul()
        elif layout == Layout.TRAILING_CHANNELS:
            y_feats = jnp.swapaxes(y_feats, -1, -2)
            return e3nn.IrrepsArray(tgt, y_feats).axis_to_mul()
        else:
            raise RuntimeError(f"Unsupported layout {layout}")
