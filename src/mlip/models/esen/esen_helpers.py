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
from typing import List, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from mlip.models.blocks import MLP
from mlip.models.esen.coefficient_mapping import CoefficientMapping
from mlip.models.esen.moe import MoEDense
from mlip.utils.jax_utils import segment_sum

NODE_OFFSET = 0


class GateActivation(nn.Module):
    """
    JAX/Flax version.

    Args:
        l_max: int
        m_max: int
        num_channels: int
        m_prime: if True, uses the m'-ordering expansion; else per-l expansion

    Expects:
        gating_scalars: [N, l_max * num_channels]
        input_tensors:  [N, 1 + num_components, num_channels]
            where num_components = sum_{l=1..l_max} min(2l+1, 2*m_max+1)
            (i.e., first slice is l=0 scalars, remainder are vector components)
    """

    l_max: int
    m_max: int
    num_channels: int
    m_prime: bool = False

    def setup(self) -> None:
        """Initializes the gate activation layers."""
        # total number of non-scalar components (exclude l=0)
        num_components = 0
        for lval in range(1, self.l_max + 1):
            num_components += min(2 * lval + 1, 2 * self.m_max + 1)

        # build expand_index
        idx = []
        if self.m_prime:
            # first the m=0 slice: l=1..l_max → indices 0..(l_max-1)
            idx.extend(list(range(self.l_max)))
            # then for each |m|>=1, repeat [m-1 .. l_max-1] twice (for +m and -m)
            for mval in range(1, self.m_max + 1):
                span = list(range(mval - 1, self.l_max))
                idx.extend(span)  # +m
                idx.extend(span)  # -m
        else:
            # per-l expansion: for each l, repeat (l-1) as many times
            # as its kept m-components
            for lval in range(1, self.l_max + 1):
                length = min(2 * lval + 1, 2 * self.m_max + 1)
                idx.extend([lval - 1] * length)

        assert len(idx) == num_components
        self.expand_index = jnp.array(idx, dtype=jnp.int32)

    @nn.compact
    def __call__(
        self, gating_scalars: jax.Array, input_tensors: jax.Array
    ) -> jax.Array:
        """
        Applies the gate activation to the input tensors.

        Args:
            gating_scalars: [nodes, l_max * num_channels]
            input_tensors:  [nodes, 1 + num_components, num_channels]

        Returns:
            Updated input tensors with gated activation.
            Shape: [nodes, 1 + num_components, num_channels].
        """
        gates = nn.sigmoid(gating_scalars).reshape((
            gating_scalars.shape[0],
            self.l_max,
            self.num_channels,
        ))
        gates = jnp.take(gates, self.expand_index, axis=1)

        # Split scalars (l=0) vs vectors (rest)
        scalars = input_tensors[:, :1, :]
        vectors = input_tensors[:, 1:, :]

        scalars = nn.silu(scalars)
        vectors = vectors * gates

        return jnp.concatenate([scalars, vectors], axis=1)


class SO2mConv(nn.Module):
    """
    Performs an SO(2) convolution on features corresponding to ±m.

    Args:
        m:                  order of the spherical harmonic coefficients
        sphere_channels:    number of spherical input channels
        m_output_channels:  number of output channels used in the SO(2) conv
        l_max:               maximum degree
        m_max:               maximum order
    """

    m: int
    sphere_channels: int
    m_output_channels: int
    l_max: int
    m_max: int
    num_experts: int | None = None

    def setup(self) -> None:
        """Initializes the SO(2) convolution layers."""
        assert self.m_max >= self.m
        num_coefficients = self.l_max - self.m + 1
        num_channels = num_coefficients * self.sphere_channels

        self.out_channels_half = self.m_output_channels * (
            num_channels // self.sphere_channels
        )

        self.fc = MoEDense(
            features=2 * self.out_channels_half,
            num_experts=self.num_experts,
            use_bias=False,
            kernel_init=nn.initializers.variance_scaling(
                scale=1.0 / 2.0,
                mode="fan_in",
                distribution="truncated_normal",
            ),
        )

    @nn.compact
    def __call__(
        self,
        x_m: jax.Array,
        moe_coeffs: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Applies the SO(2) convolution on features corresponding to ±m.

        Implementation follows section 3 of:
            * Saro Passaro, Lawrence Zitnick.
            Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs.
            URL: https://arxiv.org/pdf/2302.03655

        Args:
            x_m: [E, 2, num_channels]  (real/imag parts stacked along axis=1)

        Returns:
            x_m_r, x_m_i: each [E, num_coefficients, m_output_channels]
        """
        e = x_m.shape[0]

        # Apply linear transformation along the channel dimension
        x_m = self.fc(x_m, moe_coeffs)

        # Reshape & split into the four parts: (r0, i0, r1, i1)
        x_m = x_m.reshape(e, -1, self.out_channels_half)
        x_r_0, x_i_0, x_r_1, x_i_1 = jnp.split(x_m, 4, axis=1)

        # Combine into real and imaginary parts
        x_m_r = x_r_0 - x_i_1
        x_m_i = x_r_1 + x_i_0

        # Reshape to match output spec
        x_m_r = x_m_r.reshape(e, -1, self.m_output_channels)
        x_m_i = x_m_i.reshape(e, -1, self.m_output_channels)

        return x_m_r, x_m_i


class SO2Convolution(nn.Module):
    """
    Applies an SO(2) convolution on features corresponding to ±m.

    Using the SO2mConv class to implement the SO(2) convolution.

    Args:
        sphere_channels:    input channels per spherical component (c_in)
        m_output_channels:  output channels per spherical component (c_out)
        l_max, m_max:         degrees/orders
        mapping_reduced:     needs .m_size (list-like of counts per m)
        internal_weights:   if False, use a radial MLP on x_edge and scale inputs
        edge_channels_list: MLP sizes for the radial embedding
                            (if internal_weights=False).
                            Final layer width is inferred automatically.
        extra_m0_output_channels:
                            If set, returns (out, extra_m0) like the PyTorch version.
        SO2mConvCls:        Your Linen class implementing a single-m SO(2) conv:
                            out_list = SO2mConvCls(...)(x_m)
                            where x_m has shape [E, 2, m_size[m]*c_in]
                            and out_list is a list of two tensors (±m) each
                            [E, something_m, c_out]
    """

    sphere_channels: int
    m_output_channels: int
    l_max: int
    m_max: int
    mapping_reduced: object
    internal_weights: bool = True
    edge_channels_list: Sequence[int] | None = None
    extra_m0_output_channels: int | None = None
    num_experts: int | None = None

    def setup(self) -> None:
        """Initializes the SO(2) convolution layers."""
        channels_in = self.sphere_channels

        m_sizes: List[int] = [int(self.mapping_reduced.m_size[0])]
        m_sizes += [int(s) * 2 for s in self.mapping_reduced.m_size[1 : self.m_max + 1]]
        self.m_split_sizes = m_sizes

        edge_splits = [int(self.mapping_reduced.m_size[0]) * channels_in]
        edge_splits += [
            int(self.mapping_reduced.m_size[m]) * channels_in
            for m in range(1, self.m_max + 1)
        ]
        self.edge_split_sizes = edge_splits

        # m=0 linear
        num_channels_m0 = (self.l_max + 1) * channels_in
        m0_out = self.m_output_channels * ((num_channels_m0) // channels_in)
        if self.extra_m0_output_channels is not None:
            m0_out += self.extra_m0_output_channels
        self.fc_m0 = MoEDense(
            features=m0_out,
            num_experts=self.num_experts,
            use_bias=True,
            kernel_init=nn.initializers.lecun_normal(),
        )

        # m>0 SO2 convs
        self.so2_m_conv = [
            SO2mConv(
                m=m,
                sphere_channels=channels_in,
                m_output_channels=self.m_output_channels,
                l_max=self.l_max,
                m_max=self.m_max,
                num_experts=self.num_experts,
            )
            for m in range(1, self.m_max + 1)
        ]

        # optional radial embed
        self.rad_func = None
        if not self.internal_weights:
            if self.edge_channels_list is None:
                raise ValueError(
                    "edge_channels_list must be provided when internal_weights=False."
                )
            total_rad_in = sum(self.edge_split_sizes)
            self.rad_func = MLP(
                layer_sizes=list(self.edge_channels_list) + [int(total_rad_in)],
                activation=nn.silu,
                kernel_init=nn.initializers.lecun_normal(),
                use_bias=True,
                use_layer_norm=True,
            )

        def _sizes_to_indices(sizes):
            acc = 0
            idx = []
            for s in sizes[:-1]:
                acc += int(s)
                idx.append(acc)
            return idx

        self.m_split_indices = _sizes_to_indices(self.m_split_sizes)
        self.edge_split_indices = _sizes_to_indices(self.edge_split_sizes)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        x_edge: jax.Array,
        moe_coeffs: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array | None]:
        """
        Applies the SO(2) convolution on features corresponding to ±m.

        Args:
            x: [E, num_coefficients, sphere_channels]
            x_edge: [E, num_coefficients, sphere_channels]

        Returns:
            out: [E, num_coefficients, m_output_channels]
            x0_extra: [E, extra_m0_output_channels] if extra_m0_output_channels is not
                      None
        """
        e = x.shape[0]

        x_by_m = list(jnp.split(x, self.m_split_indices, axis=1))

        if self.rad_func is not None:
            rad_all = self.rad_func(x_edge)
            x_edge_by_m = list(jnp.split(rad_all, self.edge_split_indices, axis=1))
        else:
            x_edge_by_m = None

        x0 = x_by_m[0].reshape(e, -1)
        if x_edge_by_m is not None:
            x0 = x0 * x_edge_by_m[0]
        x0 = self.fc_m0(x0, moe_coeffs)

        x0_extra = None
        if self.extra_m0_output_channels is not None:
            extra = self.extra_m0_output_channels
            x0_extra = x0[:, :extra]
            x0 = x0[:, extra:]

        x0 = x0.reshape(e, -1, self.m_output_channels)
        out_blocks = (x0,)

        for m in range(1, self.m_max + 1):
            xm = x_by_m[m].reshape(e, 2, -1)
            if x_edge_by_m is not None:
                xm = xm * x_edge_by_m[m][:, None, :]
            out_pm_list = self.so2_m_conv[m - 1](xm, moe_coeffs)
            out_blocks = out_blocks + tuple(out_pm_list)

        out = jnp.concatenate(out_blocks, axis=1)
        return (out, x0_extra) if x0_extra is not None else out


##############################
# SpectralAtomwise and utils #
##############################


class SO3Linear(nn.Module):
    """Applies a SO(3) linear transformation to the input tensors.

    Attributes:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        l_max: The highest harmonic order included in the Spherical Harmonics series.
    """

    in_channels: int
    out_channels: int
    l_max: int

    def setup(self) -> None:
        """Initializes the SO(3) linear transformation layers."""
        idx = []
        for l_index in range(self.l_max + 1):
            idx.extend([l_index] * (2 * l_index + 1))
        self.expand_index = jnp.array(idx, dtype=jnp.int32)

        bound = 1.0 / math.sqrt(self.in_channels + 1e-12)

        def weight_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

        self.weight_init = weight_init

    @nn.compact
    def __call__(self, input_embedding: jax.Array) -> jax.Array:
        """Applies the SO(3) linear transformation to the input tensors.

        Args:
            input_embedding: [N, l_max + 1, in_channels]

        Returns:
            [N, l_max + 1, out_channels]
        """
        b, k, _ = input_embedding.shape
        assert k == (self.l_max + 1) ** 2

        weight = self.param(
            "weight",
            self.weight_init,
            (self.l_max + 1, self.out_channels, self.in_channels),
        )
        bias = self.param("bias", nn.initializers.zeros, (self.out_channels,))

        weight_expanded = jnp.take(weight, self.expand_index, axis=0)

        out = jnp.einsum("bmi,moi->bmo", input_embedding, weight_expanded)

        # bias only on l=0
        out = out.at[:, :1, :].add(bias[None, None, :])

        return out


#######################
# EdgeDegreeEmbedding #
#######################


class EdgeDegreeEmbedding(nn.Module):
    """
    Args:
        sphere_channels:      Number of spherical channels
        l_max, m_max:           Kept for parity with original signature
                              (not used directly here)
        edge_channels_list:   Base MLP layer sizes. We append (m0 * sphere_channels)
                              as final out dim.
        rescale_factor:       Rescale factor for sum aggregation
        mapping_reduced:       Provides m_0 count and total (via l_harmonic length)
        deterministic_scatter_ops: Whether to use deterministic scatter operations.
    """

    sphere_channels: int
    l_max: int
    m_max: int
    edge_channels_list: Sequence[int]
    rescale_factor: float
    mapping_reduced: CoefficientMapping
    deterministic_scatter_ops: bool = False

    def setup(self) -> None:
        """Initializes the EdgeDegreeEmbedding layers."""
        self.m_0_num_coefficients: int = int(self.mapping_reduced.m_size[0])
        self.m_all_num_coefficients: int = int(len(self.mapping_reduced.l_harmonic))

        self.rad_func = MLP(
            layer_sizes=list(self.edge_channels_list)
            + [self.m_0_num_coefficients * self.sphere_channels],
            activation=nn.silu,
            kernel_init=nn.initializers.lecun_normal(),
            use_layer_norm=True,
            use_bias=True,
        )

    def _forward_chunk(
        self,
        x: jax.Array,
        x_edge: jax.Array,
        edge_index: jax.Array,
        wigner_and_m_mapping_inv: jax.Array,
        edge_envelope: jax.Array,
        node_offset: int = 0,
    ) -> jax.Array:

        x_edge_m0 = self.rad_func(x_edge)
        x_edge_m0 = x_edge_m0.reshape(
            -1, self.m_0_num_coefficients, self.sphere_channels
        )  # [E, m0, C]

        pad_m = self.m_all_num_coefficients - self.m_0_num_coefficients
        x_edge_embed = jnp.pad(
            x_edge_m0,
            ((0, 0), (0, pad_m), (0, 0)),
            mode="constant",
        )

        x_edge_embed = jnp.einsum(
            "eij,ejc->eic", wigner_and_m_mapping_inv, x_edge_embed
        )

        x_edge_embed = x_edge_embed * edge_envelope

        dst = (edge_index[1] - node_offset).astype(jnp.int32)
        x_edge_embed = x_edge_embed / (self.rescale_factor + 1e-12)

        summed = segment_sum(
            x_edge_embed,
            dst,
            num_segments=x.shape[0],
            deterministic=self.deterministic_scatter_ops,
        )

        return x + summed

    def __call__(
        self,
        x: jax.Array,
        x_edge: jax.Array,
        edge_index: tuple[jax.Array, jax.Array],
        wigner_and_m_mapping_inv: jax.Array,
        edge_envelope: jax.Array,
        node_offset: int = 0,
    ) -> jax.Array:
        """Applies the EdgeDegreeEmbedding to the input tensors.

        Args:
            x: [E, num_coefficients, sphere_channels]
            x_edge: [E, num_coefficients, sphere_channels]
            edge_index: [E, 2]
            wigner_and_m_mapping_inv: [E, num_coefficients, num_coefficients]
            edge_envelope: [E, 1, 1]
            node_offset: int

        Returns:
            [E, num_coefficients, sphere_channels]
        """
        return self._forward_chunk(
            x, x_edge, edge_index, wigner_and_m_mapping_inv, edge_envelope, node_offset
        )
