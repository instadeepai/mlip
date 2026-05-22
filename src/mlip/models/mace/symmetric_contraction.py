# MIT License
# Copyright (c) 2022 mace-jax
# See https://github.com/ACEsuit/mace-jax/blob/main/MIT.md
#
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

# flake8: noqa: N806
import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
from e3j.core.permutation import Permutation
from e3j.core.power_expansion import PowerExpansion
from e3j.linen.linear_indexwise import LinearIndexwise
from e3j.utils.options import Layout
from jax import Array, vmap


class SymmetricContraction(nn.Module):
    """
    Contracts the power expansion of E3-features with index-wise weights,

        B(A, i) = W[i] @ (A + (A ⊗ A) + ... + A**(⊗ ν))

    where:

    * `ν` is the correlation order (typically lower than 4),
    * `A : (N, C, D)` is an array of `D`-dimensional E3-features,
    * `i : (N,)` is the vector of species indices `0 <= i < S`,
    * `W` stores `(S, C)` batches of `(1, M_lp)` matrices `W_lp`
      acting on momentum-l and parity-p blocks to combine the
      multiplicities linearly. See :class:`LinearIndexwise`.

    The shapes given above depend on

    * `N` the batch size,
    * `C` the number of independent channels,
    * `S` the number of species,
    * `D` the dimension of the input representation.

    References
    ----------
    See MACE, eq.10.

    * `MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and
      Accurate Force Fields <https://arxiv.org/abs/2206.07697>`_.
      Batatia, Kovacs, Simm, Ortner & Csanyi, 2022.
    """

    source_irreps: str
    correlation: int
    keep_irrep_out: str
    num_species: int
    num_channels: int
    layout: Layout | str
    l_max: int | None = None

    @property
    def axis_lm(self) -> int:
        layout = Layout.parse(self.layout)
        return -2 if layout == Layout.TRAILING_CHANNELS else -1

    @property
    def source(self) -> e3nn.Irreps:
        return e3nn.Irreps(self.source_irreps)

    @property
    def target(self) -> e3nn.Irreps:
        return e3nn.Irreps(self.keep_irrep_out)

    def expansion_layer(self) -> PowerExpansion:
        """Direct sum of equivariant powers."""
        return PowerExpansion(
            source=self.source_irreps,
            exponent=self.correlation,
            target_filter=self.keep_irrep_out,
            l_max=self.l_max,
            layout=Layout.parse(self.layout),
        )

    def permutation_layer(self, o3spaces: list) -> Permutation:
        """Groups equivalent irreps before the output linear transform."""
        with jax.ensure_compile_time_eval():
            combined = " + ".join(str(s) for s in o3spaces)
            perm = Permutation.sort(combined, axis=self.axis_lm)
        return perm

    def linear_indexwise_layer(self, source) -> LinearIndexwise:
        """Linear layer aggregating multiplicities with species-dependent weights."""
        layout = Layout.parse(self.layout)
        return LinearIndexwise(
            str(source),
            self.keep_irrep_out,
            num_indices=self.num_species,
            num_channels=self.num_channels,
            kernel_init="FAN_IN",
            rescale_gradients=False,
            layout=layout if layout != Layout.E3NN else Layout.LEADING_CHANNELS,
        )

    @nn.compact
    def __call__(
        self, node_feats: e3nn.IrrepsArray, index: jnp.ndarray
    ) -> e3nn.IrrepsArray:
        """Power expansion of node_feats, contracted with index-wise weights.

        Args:
            node_feats: array of input node features with leading channel axis,
                shape `(-1, num_channels, irreps_in.dim)`
            index: vector of node species-indices, in bounds `[0, num_species)`.

        Returns:
            Array of output node features, shape `(-1, num_channels, target.dim)`
        """
        # Note: can later be removed when fixed on e3j.PowerExpansion side.
        #       Right now would force "OUTER" tp mode on non-trailing channels.
        if self.layout != Layout.TRAILING_CHANNELS:
            raise NotImplementedError("Update e3j to v0.1.0a9 or later")

        expansion_layer = self.expansion_layer()
        permutation = self.permutation_layer(expansion_layer.target)
        linear_out = self.linear_indexwise_layer(permutation.target.regroup())

        # Cast LEADING_CHANNELS input to internal layout.
        x_feats = self._cast_inputs(node_feats.array)

        # Evaluate equivariant powers up to `correlation`.
        powers = expansion_layer(x_feats)
        x_powers = jnp.concat(powers, axis=self.axis_lm)
        # Sort/regroup degrees ahead of the linear mixing
        x_sorted = permutation(x_powers)

        # Apply LinearIndexwise block to average over multiplicities
        node_out = linear_out(x_sorted, index)

        # Cast internal outputs back to LEADING_CHANNELS.
        node_out = self._cast_outputs(node_out)
        return e3nn.IrrepsArray(self.keep_irrep_out, node_out)

    def _cast_inputs(self, x_feats: Array) -> Array:
        # Cast LEADING_CHANNELS input to internal layout.
        # Note: assumes shape (N, C, lm) passed by caller after mul_to_axis()
        layout = Layout.parse(self.layout)
        N, C = x_feats.shape[:2]
        if layout in (Layout.LEADING_CHANNELS, Layout.E3NN):
            pass
        elif layout == Layout.TRAILING_CHANNELS:
            x_feats = jnp.swapaxes(x_feats, -1, -2)
        else:
            raise RuntimeError(f"Unsupported layout {layout}")
        return x_feats

    def _cast_outputs(self, y_feats: Array) -> Array:
        # Cast internal layout back to LEADING_CHANNELS.
        layout = Layout.parse(self.layout)
        if layout == Layout.LEADING_CHANNELS:
            pass
        elif layout == Layout.E3NN:
            C, Dout = self.num_channels, self.target.dim
            y_feats = y_feats.reshape(-1, C, Dout)
        elif layout == Layout.TRAILING_CHANNELS:
            return jnp.swapaxes(y_feats, -1, -2)


# Note: keep until numerical comparison with e3nn v1 models is desired.
#       The e3j alternative cannot currently match 1-1 due a different algorithm
#       and parameter structure, although it should be in theory possible.


class SymmetricContractionE3NN(nn.Module):
    """Former e3nn-based symmetric contraction.

    Uses a (ν+1)-dimensional tensor product basis, where `ν = self.correlation`,
    and species-dependent learnable weights to accumulate linear projections of
    equivariant powers with a for loop of einsum operations.

    The basis is either obtained from `e3nn.reduced_tensor_product_basis` or
    `e3nn.reduced_symmetric_tensor_product_basis` depending on the associated
    `symmetric_tensor_product_basis` flag.

    Hyperparameters follow the v2 API so the module is a drop-in replacement for
    :class:`SymmetricContraction`, which currently relies on `e3j.PowerExpansion`
    and `e3j.LinearIndexwise` for the same task. The 1-1 numerical mapping of both
    implementations is not available.
    """

    source_irreps: str
    correlation: int
    keep_irrep_out: str
    num_species: int
    num_channels: int
    layout: Layout | str = "E3NN"
    l_max: int | None = None
    symmetric_tensor_product_basis: bool = False

    @property
    def _keep_irrep_out(self) -> e3nn.Irreps:
        out = e3nn.Irreps(self.keep_irrep_out)
        if not all(mul == 1 for mul, _ in out):
            raise ValueError("Expecting mul = 1 for `keep_irrep_out` filter")
        return out

    @nn.compact
    def __call__(
        self, node_feats: e3nn.IrrepsArray, index: jnp.ndarray
    ) -> e3nn.IrrepsArray:

        gradient_normalization = 0.0  # "element" normalization, matching v1

        def fn(features: e3nn.IrrepsArray, index: jnp.ndarray):
            assert features.ndim == 2  # [num_features, irreps_x.dim]
            assert index.ndim == 0  # int
            out = {}
            for order in range(self.correlation, 0, -1):  # correlation, ..., 1
                x_ = features.array
                if self.symmetric_tensor_product_basis:
                    U = e3nn.reduced_symmetric_tensor_product_basis(
                        features.irreps, order, keep_ir=self._keep_irrep_out
                    )
                else:
                    U = e3nn.reduced_tensor_product_basis(
                        [features.irreps] * order, keep_ir=self._keep_irrep_out
                    )

                for (mul, ir_out), u_ in zip(U.irreps, U.chunks):
                    u = u_.astype(x_.dtype)
                    # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]
                    w = self.param(
                        f"w{order}_{ir_out}",
                        nn.initializers.normal(
                            stddev=(mul**-0.5) ** (1.0 - gradient_normalization)
                        ),
                        (self.num_species, mul, features.shape[0]),
                        dtype=jnp.float32,
                    )[index]  # [multiplicity, num_features]
                    w = w * (mul**-0.5) ** gradient_normalization
                    if ir_out not in out:
                        out[ir_out] = (
                            "special",
                            jnp.einsum("...jki,kc,cj->c...i", u, w, x_),
                        )  # [num_features, (irreps_x.dim)^(order-1), ir_out.dim]
                    else:
                        out[ir_out] += jnp.einsum(
                            "...ki,kc->c...i", u, w
                        )  # [num_features, (irreps_x.dim)^order, ir_out.dim]
                for ir_out, val in out.items():
                    if isinstance(val, tuple):
                        out[ir_out] = val[1]
                        continue
                    out[ir_out] = jnp.einsum(
                        "c...ji,cj->c...i", val, x_
                    )  # [num_features, (irreps_x.dim)^(order-1), ir_out.dim]
            # out[irrep_out] : [num_features, ir_out.dim]
            irreps_out = e3nn.Irreps(sorted(out.keys()))
            return e3nn.from_chunks(
                irreps_out,
                [out[ir][:, None, :] for (_, ir) in irreps_out],
                (features.shape[0],),
            )

        shape = jnp.broadcast_shapes(node_feats.shape[:-2], index.shape)
        node_feats = node_feats.broadcast_to(shape + node_feats.shape[-2:])
        index = jnp.broadcast_to(index, shape)
        fn_mapped = fn
        for _ in range(node_feats.ndim - 2):
            fn_mapped = vmap(fn_mapped)
        return fn_mapped(node_feats, index)
