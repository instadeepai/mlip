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
from typing import Set, Union

import e3nn_jax as e3nn
import flax.linen as nn
import jax.numpy as jnp
from jax import vmap

A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]


class SymmetricContraction(nn.Module):
    correlation: int
    keep_irrep_out: Set[e3nn.Irrep]
    num_species: int
    gradient_normalization: Union[str, float] = None
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False

    @property
    def _keep_irrep_out(self) -> e3nn.Irreps:
        """Parse `keep_irrep_out` attribute, possibly a string."""
        out = e3nn.Irreps(self.keep_irrep_out)
        if not all(mul == 1 for mul, _ in out):
            raise ValueError("Expecting mul = 1 for `keep_irrep_out` filter")
        return out

    @nn.compact
    def __call__(
        self, node_feats: e3nn.IrrepsArray, index: jnp.ndarray
    ) -> e3nn.IrrepsArray:
        """Power expansion of node_feats, mapped through index-wise weights.

        This module should return the equivalent of

            B = W[index] @ (A + (A ⊗ A) + ... + A**(⊗ ν))

        where `A = node_feats`, and `W` represents learnable weights acting
        specie-index-wise and momentum-wise on the equivariant powers of
        the node features.
        """
        gradient_normalization = self.gradient_normalization
        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
            # possibly a string now
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]

        def fn(features: e3nn.IrrepsArray, index: jnp.ndarray):
            # - This operation is parallel on the feature dimension (but each feature has its own parameters)
            # This operation is an efficient implementation of
            # vmap(lambda w, x: FunctionalLinear(irreps_out)(w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)
            # up to x power self.correlation
            assert features.ndim == 2  # [num_features, irreps_x.dim]
            assert index.ndim == 0  # int
            out = {}
            for order in range(self.correlation, 0, -1):  # correlation, ..., 1
                if self.off_diagonal:
                    x_ = jnp.roll(features.array, A025582[order - 1])
                else:
                    x_ = features.array
                if self.symmetric_tensor_product_basis:
                    U = e3nn.reduced_symmetric_tensor_product_basis(
                        features.irreps, order, keep_ir=self._keep_irrep_out
                    )
                else:
                    U = e3nn.reduced_tensor_product_basis(
                        [features.irreps] * order, keep_ir=self._keep_irrep_out
                    )

                for (mul, ir_out), u in zip(U.irreps, U.chunks):
                    u = u.astype(x_.dtype)
                    # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]
                    w = self.param(
                        f"w{order}_{ir_out}",
                        nn.initializers.normal(
                            stddev=(mul**-0.5) ** (1.0 - gradient_normalization)
                        ),
                        (self.num_species, mul, features.shape[0]),
                        dtype=jnp.float32,
                    )[
                        index
                    ]  # [multiplicity, num_features]
                    w = w * (mul**-0.5) ** gradient_normalization  # normalize weights
                    if ir_out not in out:
                        out[ir_out] = (
                            "special",
                            jnp.einsum("...jki,kc,cj->c...i", u, w, x_),
                        )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                    else:
                        out[ir_out] += jnp.einsum(
                            "...ki,kc->c...i", u, w
                        )  # [num_features, (irreps_x.dim)^order, ir_out.dim]
                # ((w3 x + w2) x + w1) x
                #  \----------------/
                #         out (in the normal case)
                for ir_out in out:
                    if isinstance(out[ir_out], tuple):
                        out[ir_out] = out[ir_out][1]
                        continue  # already done (special case optimization above)
                    out[ir_out] = jnp.einsum(
                        "c...ji,cj->c...i", out[ir_out], x_
                    )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                # ((w3 x + w2) x + w1) x
                #  \-------------------/
                #           out
            # out[irrep_out] : [num_features, ir_out.dim]
            irreps_out = e3nn.Irreps(sorted(out.keys()))
            return e3nn.from_chunks(
                irreps_out,
                [out[ir][:, None, :] for (_, ir) in irreps_out],
                (features.shape[0],),
            )

        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(node_feats.shape[:-2], index.shape)
        node_feats = node_feats.broadcast_to(shape + node_feats.shape[-2:])
        index = jnp.broadcast_to(index, shape)
        fn_mapped = fn
        for _ in range(node_feats.ndim - 2):
            fn_mapped = vmap(fn_mapped)
        return fn_mapped(node_feats, index)
