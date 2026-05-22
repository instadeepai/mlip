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

from enum import Enum
from typing import Callable

import jax
import jax.nn.initializers as initializers
from jax import Array
from numpy import sqrt

from mlip.models.radial_embedding import (
    cosine_cutoff,
    polynomial_envelope_updated,
    soft_envelope,
)


class Activation(Enum):
    """Supported activation functions:

    Options are:
    `TANH = "tanh"`,
    `SILU = "silu"`,
    `RELU = "relu"`,
    `ELU = "elu"`,
    `SWISH = "swish"`,
    `SIGMOID = "sigmoid"`, and
    `NONE = "none"`.
    """

    TANH = "tanh"
    SILU = "silu"
    RELU = "relu"
    ELU = "elu"
    SWISH = "swish"
    SIGMOID = "sigmoid"
    NONE = "none"


class RadialEnvelope(Enum):
    """Radial envelope options.

    Attributes:
        POLYNOMIAL: Polynomial envelope.
        SOFT: Soft envelope.
    """

    POLYNOMIAL = "polynomial_envelope"
    SOFT = "soft_envelope"
    COSINE_CUTOFF = "cosine_cutoff"


class RadialBasis(Enum):
    """Options for the radial basis.

    Attributes:
        GAUSS: Gaussian smearing.
        EXPNORM: Exponential normal smearing.
        BESSEL: Bessel functions.
    """

    GAUSS = "gauss"
    EXPNORM = "expnorm"
    BESSEL = "bessel"


# --- Option parsers ---


def parse_activation(act: Activation | str) -> Callable[[Array], Array]:
    """Parse activation function among available options.

    See :class:`~mlip.models.options.Activation`.
    """
    activations_map = {
        Activation.TANH: jax.nn.tanh,
        Activation.SILU: jax.nn.silu,
        Activation.RELU: jax.nn.relu,
        Activation.ELU: jax.nn.elu,
        Activation.SWISH: jax.nn.swish,
        Activation.SIGMOID: jax.nn.sigmoid,
        Activation.NONE: lambda x: x,
    }
    assert set(Activation) == set(activations_map.keys())
    return activations_map[Activation(act)]


def parse_radial_envelope(envelope: RadialEnvelope | str) -> Callable:
    """Parse `RadialEnvelope` parameter among available options.

    See :class:`~mlip.models.options.RadialEnvelope`.
    """
    radial_envelope_map = {
        RadialEnvelope.POLYNOMIAL: polynomial_envelope_updated,
        RadialEnvelope.SOFT: soft_envelope,
        RadialEnvelope.COSINE_CUTOFF: cosine_cutoff,
    }
    assert set(RadialEnvelope) == set(radial_envelope_map.keys())
    return radial_envelope_map[RadialEnvelope(envelope)]


class GradientScaledKernelInit(Enum):
    """Kernel initializer options for gradient-scaled initialization.

    Gradient-scaled initialization refers to the case that a unit variance
    initializer is used, but runtime scaling is applied to the weights,
    which also scales the gradients.
    """

    FAN_IN_NORMAL = "fan_in_normal"


def get_layer_initializer_and_scale(
    dim_in: int,
    kernel_init: initializers.Initializer | GradientScaledKernelInit | str,
) -> tuple[initializers.Initializer, float]:
    """Return a kernel initialiser and its paired runtime scaling factor.

    If `kernel_init` is a `GradientScaledKernelInit`, returns a unit variance
    initializer with a runtime scaling to adjust the effective variance
    in a way that also applies to the gradients.
    If `kernel_init` is a callable, returns it with a runtime scaling of 1.0.

    Args:
        dim_in: Input width of the Dense layer.
        kernel_init: Initialiser type. Either a `GradientScaledKernelInit`
            or a callable initializer.

    Returns:
        A tuple of `(initializer, runtime_scale)`.
    """
    if callable(kernel_init):
        return kernel_init, 1.0

    if GradientScaledKernelInit(kernel_init) == GradientScaledKernelInit.FAN_IN_NORMAL:
        return initializers.normal(1.0), sqrt(1 / dim_in)

    raise ValueError(f"Unsupported kernel initializer: {kernel_init}")
