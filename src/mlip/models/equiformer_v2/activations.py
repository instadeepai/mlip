# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Yi-Lun Liao (https://github.com/atomicarchitects/equiformer_v2) under MIT license.

import jax
import jax.numpy as jnp
import flax.linen as nn

from mlip.models.equiformer_v2.transform import get_s2grid_mats
from mlip.models.equiformer_v2.utils import get_expand_index


class SmoothLeakyReLU(nn.Module):
    """Smooth Leaky ReLU activation."""
    negative_slope: float = 0.2

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x1 = ((1 + self.negative_slope) / 2) * x
        x2 = ((1 - self.negative_slope) / 2) * x * (2 * nn.sigmoid(x) - 1)
        return x1 + x2

# --- Vector activation functions ---


class GateActivation(nn.Module):
    """Apply gate for vector and silu for scalar."""
    lmax: int
    mmax: int
    num_channels: int
    m_prime: bool = False

    @nn.compact
    def __call__(self, gating_scalars: jax.Array, input_tensors: jax.Array) -> jax.Array:
        """
        `gating_scalars`: shape [N, lmax * num_channels]
        `input_tensors`: shape  [N, (lmax + 1) ** 2, num_channels]
        """
        expand_index = get_expand_index(
            self.lmax, self.mmax, vector_only=True, m_prime=self.m_prime
        )

        gating_scalars = nn.sigmoid(gating_scalars)
        gating_scalars = gating_scalars.reshape(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )[:, expand_index]

        input_tensors_scalars = nn.silu(input_tensors[:, 0:1])
        input_tensors_vectors = input_tensors[:, 1:] * gating_scalars
        output_tensors = jnp.concat(
            (input_tensors_scalars, input_tensors_vectors), axis=1
        )

        return output_tensors


class S2Activation(nn.Module):
    """Apply silu on sphere function."""
    lmax: int
    mmax: int
    resolution: int
    m_prime: bool = False

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        so3_grid = get_s2grid_mats(
            self.lmax, self.mmax, resolution=self.resolution, m_prime=self.m_prime
        )

        x_grid = so3_grid.to_grid(inputs)
        x_grid = nn.silu(x_grid)
        outputs = so3_grid.from_grid(x_grid)
        return outputs


class SeparableS2Activation(nn.Module):
    """Apply silu on sphere function for vector and silu directly for scalar."""
    lmax: int
    mmax: int
    resolution: int
    m_prime: bool = False

    @nn.compact
    def __call__(self, input_scalars: jax.Array, input_tensors: jax.Array) -> jax.Array:
        output_scalars = nn.silu(input_scalars)
        output_tensors = S2Activation(
            self.lmax, self.mmax, self.resolution, self.m_prime
        )(input_tensors)
        outputs = jnp.concat(
            (
                output_scalars[:, None],
                output_tensors[:, 1 : output_tensors.shape[1]],
            ),
            axis=1,
        )
        return outputs
