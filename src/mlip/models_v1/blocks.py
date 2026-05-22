# Copyright (c) 2022 mace-jax
# MIT License. See https://github.com/ACEsuit/mace-jax/blob/main/MIT.md
#
# Copyright (c) 2025 InstaDeep Ltd
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

from typing import Callable, Literal

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import FunctionalLinear, Irreps, IrrepsArray
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct
from e3nn_jax.utils import vmap

from mlip.models_v1.version_compatibility import VERSION


def path_major_to_channel_major_permutation(
    irreps: e3nn.Irreps, num_channels: int
) -> np.ndarray:
    """Permutation from path-major to channel-major ordering within each irrep block.

    E3NN tensor products enumerate coupling paths in path-major order:
    within each irrep block of multiplicity `num_paths * num_channels`,
    paths vary slowly and channels vary fast.

    When converting from a channel-axis layout (TRAILING_CHANNELS or
    LEADING_CHANNELS) back to E3NN via `axis_to_mul()`, the resulting
    ordering is channel-major: channels vary slowly and paths vary fast.

    This permutation reorders the components of an E3NN-layout array to
    match the ordering produced by `axis_to_mul()`.

    Applied in the MACE and NequIP v1 message-passing blocks when
    `version == 2` to match v2 TRAILING_CHANNELS outputs numerically.
    """
    permutation = []
    offset = 0
    for multiplicity, ir in irreps:
        num_paths = multiplicity // num_channels
        dim = ir.dim
        # Path-major layout: (num_paths, num_channels, dim)
        # Channel-major layout: (num_channels, num_paths, dim)
        block = np.arange(multiplicity * dim).reshape(num_paths, num_channels, dim)
        block = block.transpose(1, 0, 2).reshape(-1)
        permutation.append(offset + block)
        offset += multiplicity * dim
    return np.concatenate(permutation)


def path_major_to_channel_major_scalar_permutation(
    irreps: e3nn.Irreps, num_channels: int, inv: bool = True
) -> np.ndarray:
    """Scalar companion of `path_major_to_channel_major_permutation`.

    Same block structure, but one scalar per irrep multiplicity index
    rather than `ir.dim` components.  Used to reorder the radial MLP
    weights (one weight per TP multiplicity) before they are baked into
    the tensor product.
    """
    permutation = []
    offset = 0
    for multiplicity, ir in irreps:
        num_paths = multiplicity // num_channels
        block = np.arange(multiplicity).reshape(num_paths, num_channels)
        block = block.transpose(1, 0).reshape(-1)
        permutation.append(offset + block)
        offset += multiplicity
    p = np.concatenate(permutation)
    if not inv:
        return p
    return np.argsort(p)


class RadialEmbeddingBlock(nn.Module):
    """Radial encoding of interatomic distances."""

    r_max: float
    basis_functions: Callable[[jnp.ndarray], jnp.ndarray]
    envelope_function: Callable[[jnp.ndarray], jnp.ndarray]
    num_bessel: int
    avg_r_min: float | None = None

    @nn.compact
    def __call__(
        self,
        edge_lengths: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:  # [n_edges, num_bessel]
        def func(lengths):
            basis = self.basis_functions(
                lengths,
                self.r_max,
                self.num_bessel,
            )  # [n_edges, num_bessel]
            cutoff = self.envelope_function(lengths, self.r_max)  # [n_edges]
            return basis * cutoff[:, None]  # [n_edges, num_bessel]

        with jax.ensure_compile_time_eval():
            if self.avg_r_min is None:
                factor = 1.0
            else:
                samples = jnp.linspace(
                    self.avg_r_min, self.r_max, 1000, dtype=jnp.float32
                )
                factor = jnp.mean(func(samples) ** 2).item() ** -0.5

        embedding = factor * jnp.where(
            (edge_lengths == 0.0)[:, None], 0.0, func(edge_lengths)
        )  # [n_edges, num_bessel]

        return e3nn.IrrepsArray(f"{embedding.shape[-1]}x0e", embedding)


class FullyConnectedTensorProduct(nn.Module):
    irreps_out: e3nn.Irreps
    irreps_in1: e3nn.Irreps | None = None
    irreps_in2: e3nn.Irreps | None = None
    version: Literal[1, 2] = VERSION

    @nn.compact
    def __call__(
        self, x1: e3nn.IrrepsArray, x2: e3nn.IrrepsArray, **kwargs
    ) -> e3nn.IrrepsArray:
        irreps_out = e3nn.Irreps(self.irreps_out)
        irreps_in1 = (
            e3nn.Irreps(self.irreps_in1) if self.irreps_in1 is not None else None
        )
        irreps_in2 = (
            e3nn.Irreps(self.irreps_in2) if self.irreps_in2 is not None else None
        )
        x1 = e3nn.as_irreps_array(x1)
        x2 = e3nn.as_irreps_array(x2)
        leading_shape = jnp.broadcast_shapes(x1.shape[:-1], x2.shape[:-1])
        x1 = x1.broadcast_to(leading_shape + (-1,))
        x2 = x2.broadcast_to(leading_shape + (-1,))
        if irreps_in1 is not None:
            x1 = x1.rechunk(irreps_in1)
        if irreps_in2 is not None:
            x2 = x2.rechunk(irreps_in2)
        x1 = x1.remove_zero_chunks().simplify()
        x2 = x2.remove_zero_chunks().simplify()
        tp = FunctionalFullyConnectedTensorProduct(
            x1.irreps, x2.irreps, irreps_out.simplify()
        )

        # PATCH: wrapper around Module.param(name, initializer, shape) to
        #        optionally initialize parameters to match v2 numerically.
        ws = [self._get_param(tp, ins, i) for i, ins in enumerate(tp.instructions)]

        def helper(x1, x2):
            return tp.left_right(ws, x1, x2, **kwargs)

        for _ in range(len(leading_shape)):
            helper_vmapped = e3nn.utils.vmap(helper)

        output = helper_vmapped(x1, x2)
        return output.rechunk(self.irreps_out)

    # PATCH: Initialize transposed weights to match V2 numerically.
    def _get_param(self, tp, ins, i: int) -> jnp.ndarray:
        """Initialize/retrieve each degree l's parameter from e3nn instruction.

        The first tp argument is the internal e3nn.legacy.FunctionalTensorProduct
        built in this block. The index `i` should just match `l` with sorted, re-
        grouped irreps. It's used in the param key in V2's LinearIndexwise block.
        """
        initializer = nn.initializers.normal(stddev=ins.weight_std)
        if self.version == 1:
            key = (
                f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                f"{tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},"
                f"{tp.irreps_out[ins.i_out]}"
            )
            shape = ins.path_shape
            return self.param(key, initializer, shape)

        if self.version == 2:
            irrep = tp.irreps_in1[ins.i_in1].ir
            key = f"weight{i}_{irrep.l}_{irrep.p}"

            # LinearIndexwise layout: (num_indices, mul_out, mul_in)
            # from path_shape: (mul_in1, mul_in2, mul_out)
            mul_in1, num_idx, mul_out = ins.path_shape
            shape = (num_idx, mul_out, mul_in1)

            weights = self.param(key, initializer, shape)
            # Note: Can later be removed once e3j.LinearIndexwise rescales
            # parameters as e3nn FullyConnectedTensorProduct.
            # The FCTP einsum applies three runtime factors that
            # LinearIndexwise replaces with 1/sqrt(m_in):
            #   1. path_weight (TP normalization)
            #   2. CG(l, 0, l) = 1/sqrt(2l+1) (Clebsch-Gordan)
            #   3. LinearIndexwise scales by 1/sqrt(m_in)
            # Rescaling here cancels (1,2) and applies (3).
            factor = jnp.sqrt(2 * irrep.l + 1) / (ins.path_weight * jnp.sqrt(mul_in1))
            weights = weights * factor
            # Back to FCTP layout: (mul_in1, mul_in2, mul_out)
            return jnp.transpose(weights, (2, 0, 1))


class LinearNodeEmbeddingBlock(nn.Module):
    num_species: int
    irreps_out: e3nn.Irreps

    @nn.compact
    def __call__(self, node_specie: jnp.ndarray) -> e3nn.IrrepsArray:
        irreps_out = e3nn.Irreps(self.irreps_out).filter("0e").regroup()

        w = (1 / jnp.sqrt(self.num_species)) * self.param(
            "embeddings",
            nn.initializers.normal(stddev=1.0, dtype=jnp.float32),
            (self.num_species, irreps_out.dim),
        )
        return e3nn.IrrepsArray(irreps_out, w[node_specie])


class Linear(nn.Module):
    """Flax module of an equivariant linear layer."""

    irreps_out: Irreps
    irreps_in: Irreps | None = None

    @nn.compact
    def __call__(self, x: IrrepsArray) -> IrrepsArray:
        irreps_out = Irreps(self.irreps_out)
        irreps_in = Irreps(self.irreps_in) if self.irreps_in is not None else None

        if self.irreps_in is None and not isinstance(x, IrrepsArray):
            raise ValueError(
                "the input of Linear must be an IrrepsArray, or "
                "`irreps_in` must be specified"
            )

        if irreps_in is not None:
            x = IrrepsArray(irreps_in, x)

        x = x.remove_zero_chunks().simplify()

        lin = FunctionalLinear(x.irreps, irreps_out, instructions=None, biases=None)

        w = [
            (
                self.param(  # pylint:disable=g-long-ternary
                    f"b[{ins.i_out}] {lin.irreps_out[ins.i_out]}",
                    nn.initializers.normal(stddev=ins.weight_std),
                    ins.path_shape,
                )
                if ins.i_in == -1
                else self.param(
                    f"w[{ins.i_in},{ins.i_out}] {lin.irreps_in[ins.i_in]},"
                    f"{lin.irreps_out[ins.i_out]}",
                    nn.initializers.normal(stddev=ins.weight_std),
                    ins.path_shape,
                )
            )
            for ins in lin.instructions
        ]

        def helper(x):
            return lin(w, x)

        for _ in range(x.ndim - 1):
            helper_vmapped = vmap(helper)

        return helper_vmapped(x)
