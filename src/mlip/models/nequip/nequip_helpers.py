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

import functools
import operator
from typing import Optional

import e3nn_jax as e3nn
import flax.linen as nn
import jax
from jax.nn import initializers

from mlip.models.options import parse_activation

tree_map = functools.partial(
    jax.tree.map, is_leaf=lambda x: isinstance(x, e3nn.IrrepsArray)
)


class BetaSwish(nn.Module):

    @nn.compact
    def __call__(self, x):
        features = x.shape[-1]
        beta = self.param("Beta", nn.initializers.ones, (features,))
        return x * nn.sigmoid(beta * x)


def normal(var):
    return initializers.variance_scaling(var, "fan_in", "normal")


def prod(xs):
    """From e3nn_jax/util/__init__.py."""
    return functools.reduce(operator.mul, xs, 1)


def tp_path_exists(arg_in1, arg_in2, arg_out):
    """Check if a tensor product path is viable.

    This helper function is similar to the one used in:
    https://github.com/e3nn/e3nn
    """
    arg_in1 = e3nn.Irreps(arg_in1).simplify()
    arg_in2 = e3nn.Irreps(arg_in2).simplify()
    arg_out = e3nn.Irrep(arg_out)

    for _multiplicity_1, irreps_1 in arg_in1:
        for _multiplicity_2, irreps_2 in arg_in2:
            if arg_out in irreps_1 * irreps_2:
                return True
    return False


class MLP(nn.Module):
    """Multilayer Perceptron."""

    features: tuple[int, ...]
    nonlinearity: str

    use_bias: bool = True
    scalar_mlp_std: Optional[float] = None

    @nn.compact
    def __call__(self, x):
        features = self.features

        dense = functools.partial(nn.Dense, use_bias=self.use_bias)

        phi = (
            BetaSwish()
            if self.nonlinearity == "beta_swish"
            else parse_activation(self.nonlinearity)
        )

        kernel_init = normal(self.scalar_mlp_std)

        for h in features[:-1]:
            x = phi(dense(h, kernel_init=kernel_init)(x))

        return dense(features[-1], kernel_init=normal(1.0))(x)
