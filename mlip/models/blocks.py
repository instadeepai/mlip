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

from typing import Callable, Optional

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
from e3nn_jax import FunctionalLinear, Irreps, IrrepsArray
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct
from e3nn_jax.utils import vmap


class RadialEmbeddingBlock(nn.Module):
    """Radial encoding of interatomic distances."""

    r_max: float
    basis_functions: Callable[[jnp.ndarray], jnp.ndarray]
    envelope_function: Callable[[jnp.ndarray], jnp.ndarray]
    num_bessel: int
    avg_r_min: Optional[float] = None

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
    irreps_in1: Optional[e3nn.Irreps] = None
    irreps_in2: Optional[e3nn.Irreps] = None

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
        ws = [
            self.param(
                (
                    f"w[{ins.i_in1},{ins.i_in2},{ins.i_out}] "
                    f"{tp.irreps_in1[ins.i_in1]},{tp.irreps_in2[ins.i_in2]},"
                    f"{tp.irreps_out[ins.i_out]}"
                ),
                nn.initializers.normal(stddev=ins.weight_std),
                (ins.path_shape),
            )
            for ins in tp.instructions
        ]

        def helper(x1, x2):
            return tp.left_right(ws, x1, x2, **kwargs)

        for _ in range(len(leading_shape)):
            helper_vmapped = e3nn.utils.vmap(helper)

        output = helper_vmapped(x1, x2)
        return output.rechunk(self.irreps_out)


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
    irreps_in: Optional[Irreps] = None

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
