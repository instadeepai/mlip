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
from jax import Array

from mlip.models.radial_embedding import (
    bessel_basis,
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


class RadialBasis(Enum):
    """Radial basis option(s). For the moment, only `BESSEL = "bessel"` exists."""

    BESSEL = "bessel"


class RadialEnvelope(Enum):
    """Radial envelope options. For the moment,
    `POLYNOMIAL = "polynomial_envelope"` and `SOFT = "soft_envelope"` exist.
    """

    POLYNOMIAL = "polynomial_envelope"
    SOFT = "soft_envelope"


class VecNormType(Enum):
    """Options for the VecLayerNorm of the ViSNet model."""

    RMS = "rms"
    MAX_MIN = "max_min"
    NONE = "none"


class VisnetRBF(Enum):
    """Options for the radial basis functions used by ViSNet."""

    GAUSS = "gauss"
    EXPNORM = "expnorm"


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


def parse_radial_basis(basis: RadialBasis | str) -> Callable:
    """Parse `RadialBasis` parameter among available options.

    See :class:`~mlip.models.options.RadialBasis`.
    """
    radial_basis_map = {
        RadialBasis.BESSEL: bessel_basis,
    }
    assert set(RadialBasis) == set(radial_basis_map.keys())
    return radial_basis_map[RadialBasis(basis)]


def parse_radial_envelope(envelope: RadialEnvelope | str) -> Callable:
    """Parse `RadialEnvelope` parameter among available options.

    See :class:`~mlip.models.options.RadialEnvelope`."""
    radial_envelope_map = {
        RadialEnvelope.POLYNOMIAL: polynomial_envelope_updated,
        RadialEnvelope.SOFT: soft_envelope,
    }
    assert set(RadialEnvelope) == set(radial_envelope_map.keys())
    return radial_envelope_map[RadialEnvelope(envelope)]
