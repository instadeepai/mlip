# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Yi-Lun Liao (https://github.com/atomicarchitects/equiformer_v2) under MIT license.

from collections.abc import Callable
from enum import Enum

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from flax.typing import Dtype

from mlip.models.equiformer_v2.utils import get_expand_index


class LayerNormArray(nn.Module):
    lmax: int
    eps: float = 1e-5
    affine: bool = True
    normalization: str = "component"

    @nn.compact
    def __call__(self, node_input: jax.Array) -> jax.Array:
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        if self.affine:
            affine_weight = self.param(
                'affine_weight', initializers.ones, (self.lmax + 1, node_input.shape[-1])
            )

        out = []

        for lval in range(self.lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1

            feature = node_input[:, start_idx : start_idx+length]

            # For scalars, first compute and subtract the mean
            if lval == 0:
                feature -= jnp.mean(feature, axis=2, keepdims=True)

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = jnp.sum(jnp.pow(feature, 2), axis=1, keepdims=True) # [N, 1, C]
            elif self.normalization == "component":
                feature_norm = jnp.mean(jnp.pow(feature, 2), axis=1, keepdims=True) # [N, 1, C]
            else:
                raise ValueError(f"Unknown normalization option: {self.normalization}")

            feature_norm = jnp.mean(feature_norm, axis=2, keepdims=True)  # [N, 1, 1]
            feature_norm = jnp.pow(feature_norm + self.eps, -0.5)

            if self.affine:
                feature_norm *= affine_weight[None, lval:lval+1] # [N, 1, C]

            feature *= feature_norm

            if self.affine and lval == 0:
                feature += self.param(
                    'affine_bias', initializers.zeros, (node_input.shape[-1],)
                )[None, None]

            out.append(feature)

        out = jnp.concat(out, axis=1)
        return out


def _get_balance_degree_weight(lmax: int, dtype: Dtype, skip_l0: bool = False) -> jax.Array:
    start = 1 if skip_l0 else 0

    balance_degree_weight = jnp.zeros(((lmax + 1) ** 2 - start, 1), dtype=dtype)
    for lval in range(start, lmax + 1):
        start_idx = lval**2 - start
        length = 2 * lval + 1
        balance_degree_weight = balance_degree_weight.at[
            start_idx : (start_idx + length), :
        ].set(1.0 / length)

    return balance_degree_weight / (lmax + 1 - start)


class LayerNormArraySphericalHarmonics(nn.Module):
    """
    1. Normalize over L = 0.
    2. Normalize across all m components from degrees L > 0.
    3. Do not normalize separately for different L (L > 0).
    """

    lmax: int
    eps: float = 1e-5
    affine: bool = True
    normalization: str = "component"
    std_balance_degrees: bool = True

    @nn.compact
    def __call__(self, node_input: jax.Array) -> jax.Array:
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        # for L = 0
        feature = node_input[:, :1]
        feature = nn.LayerNorm(self.eps, use_bias=self.affine, use_scale=self.affine)(feature)

        if self.lmax == 0:
            return feature

        if self.affine:
            affine_weight = self.param(
                'affine_weight', initializers.ones, (self.lmax, node_input.shape[-1])
            )

        out = [feature]

        # for L > 0
        feature = node_input[:, 1:]

        # Then compute the rescaling factor (norm of each feature vector)
        # Rescaling of the norms themselves based on the option "normalization"
        if self.normalization == "norm":
            assert not self.std_balance_degrees
            feature_norm = jnp.sum(
                jnp.pow(feature, 2), axis=1, keepdims=True
            )  # [N, 1, C]
        elif self.normalization == "component":
            if self.std_balance_degrees:
                balance_degree_weight = _get_balance_degree_weight(
                    self.lmax, node_input.dtype, skip_l0=True
                )
                # [N, (L_max + 1)**2 - 1, C], without L = 0
                feature_norm = jnp.einsum(
                    "nic, ia -> nac", jnp.pow(feature, 2), balance_degree_weight
                )  # [N, 1, C]
            else:
                feature_norm = jnp.mean(
                    jnp.pow(feature, 2), axis=1, keepdims=True
                )  # [N, 1, C]
        else:
            raise ValueError(f"Unknown normalization option: {self.normalization}")

        feature_norm = jnp.mean(feature_norm, axis=2, keepdims=True)  # [N, 1, 1]
        feature_norm = jnp.pow(feature_norm + self.eps, -0.5)

        for lval in range(1, self.lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            # [N, (2L + 1), C]
            feature = node_input[:, start_idx : start_idx+length]
            feature_scale = feature_norm
            if self.affine:
                feature_scale *= affine_weight[None, lval-1:lval]  # [N, 1, C]
            out.append(feature * feature_scale)

        out = jnp.concat(out, axis=1)
        return out


class RMSNormArraySphericalHarmonicsV2(nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    """

    lmax: int
    eps: float = 1e-5
    affine: bool = True
    normalization: str = "component"
    centering: bool = True
    std_balance_degrees: bool = True

    @nn.compact
    def __call__(self, node_input: jax.Array) -> jax.Array:
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        feature = node_input

        if self.centering:
            feature_l0 = feature[:, 0:1]
            feature_l0_mean = jnp.mean(feature_l0, axis=2, keepdims=True)  # [N, 1, 1]
            feature = jnp.concat(
                (feature_l0 - feature_l0_mean, feature[:, 1:feature.shape[1]]), axis=1
            )

        # for L >= 0
        if self.normalization == "norm":
            assert not self.std_balance_degrees
            feature_norm = jnp.sum(
                jnp.pow(feature, 2), axis=1, keepdims=True
            )  # [N, 1, C]
        elif self.normalization == "component":
            if self.std_balance_degrees:
                balance_degree_weight = _get_balance_degree_weight(
                    self.lmax, node_input.dtype, skip_l0=False
                )
                feature_norm = jnp.einsum(
                    "nic, ia -> nac", jnp.pow(feature, 2), balance_degree_weight
                )  # [N, 1, C]
            else:
                feature_norm = jnp.mean(
                    jnp.pow(feature, 2), axis=1, keepdims=True
                )  # [N, 1, C]
        else:
            raise ValueError(f"Unknown normalization option: {self.normalization}")

        feature_norm = jnp.mean(feature_norm, axis=2, keepdims=True)  # [N, 1, 1]
        feature_norm = jnp.pow(feature_norm + self.eps, -0.5)

        if self.affine:
            feature_norm *= self.param(
                'affine_weight', initializers.ones, (self.lmax + 1, node_input.shape[-1])
            )[None, get_expand_index(self.lmax)] # [N, (L_max + 1)**2, C]

        out = feature * feature_norm

        if self.affine and self.centering:
            out = out.at[:, 0:1, :].add(self.param(
                'affine_bias', initializers.zeros, (node_input.shape[-1],)
            )[None, None])

        return out


# --- Normalization options ---


class LayerNormType(Enum):
    """Options for the LayerNorm of the EquiformerV2 model."""

    LAYER_NORM = "layer_norm"
    LAYER_NORM_SH = "layer_norm_sh"
    RMS_NORM_SH = "rms_norm_sh"


def parse_layernorm(
    norm_type: LayerNormType | str,
    lmax: int,
    eps: float = 1e-5,
    affine: bool = True,
    normalization: str = "component",
) -> Callable:
    assert normalization in ["norm", "component"]
    norm_type_map = {
        LayerNormType.LAYER_NORM: LayerNormArray,
        LayerNormType.LAYER_NORM_SH: LayerNormArraySphericalHarmonics,
        LayerNormType.RMS_NORM_SH: RMSNormArraySphericalHarmonicsV2,
    }
    norm_class = norm_type_map[LayerNormType(norm_type)]
    return norm_class(lmax, eps, affine, normalization)
