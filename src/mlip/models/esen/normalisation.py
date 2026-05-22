# Code converted from https://github.com/facebookresearch/fairchem
# Some parts of the code may remain identical, distributed under:
#
#     MIT License- Copyright (c) Meta Platforms, Inc. and affiliates.
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


from typing import Literal

import flax.linen as nn
import jax.numpy as jnp


def get_normalization_layer(
    norm_type: Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"],
    l_max: int,
    num_channels: int,
    eps: float = 1e-12,
    affine: bool = True,
    normalization: str = "component",
):
    assert norm_type in ["layer_norm", "layer_norm_sh", "rms_norm_sh"]
    if norm_type == "layer_norm":
        norm_class = EquivariantLayerNormArray
    elif norm_type == "layer_norm_sh":
        norm_class = EquivariantLayerNormArraySphericalHarmonics
    elif norm_type == "rms_norm_sh":
        norm_class = EquivariantRMSNormArraySphericalHarmonicsV2
    else:
        raise ValueError
    return norm_class(l_max, num_channels, eps, affine, normalization)


class EquivariantLayerNormArray(nn.Module):
    l_max: int
    num_channels: int
    eps: float = 1e-5
    affine: bool = True
    normalization: Literal["norm", "component"] = "component"

    @nn.compact
    def __call__(self, node_input: jnp.ndarray) -> jnp.ndarray:
        if self.affine:
            affine_weight = self.param(
                "affine_weight",
                lambda k, shape: jnp.ones(shape, dtype=node_input.dtype),
                (self.l_max + 1, self.num_channels),
            )
            affine_bias = self.param(
                "affine_bias",
                lambda k, shape: jnp.zeros(shape, dtype=node_input.dtype),
                (self.num_channels,),
            )
        else:
            affine_weight = None
            affine_bias = None

        outs = []
        for l_number in range(self.l_max + 1):
            start = l_number * l_number
            length = 2 * l_number + 1

            feat = node_input[:, start : start + length, :]

            if l_number == 0:
                feat_mean = jnp.mean(feat, axis=2, keepdims=True)
                feat = feat - feat_mean

            if self.normalization == "norm":
                feat_norm_m = jnp.sum(feat**2, axis=1, keepdims=True)
            else:
                feat_norm_m = jnp.mean(feat**2, axis=1, keepdims=True)
            feat_scale = jnp.mean(feat_norm_m, axis=2, keepdims=True)
            feat_scale = jnp.power(feat_scale + self.eps, -0.5)

            if self.affine:
                w = affine_weight[l_number][None, None, :]
                feat_scale = feat_scale * w

            feat = feat * feat_scale

            if self.affine and l_number == 0:
                b = affine_bias[None, None, :]
                feat = feat + b

            outs.append(feat)

        return jnp.concatenate(outs, axis=1)


class EquivariantLayerNormArraySphericalHarmonics(nn.Module):
    l_max: int
    num_channels: int
    eps: float = 1e-12
    affine: bool = True
    normalization: Literal["norm", "component"] = "component"
    std_balance_degrees: bool = True

    def setup(self):
        self.norm_l0 = nn.LayerNorm(
            epsilon=self.eps,
            use_bias=self.affine,
            use_scale=self.affine,
        )

        if self.affine and self.l_max > 0:
            self.affine_weight = self.param(
                "affine_weight",
                lambda key, shape: jnp.ones(shape, dtype=jnp.float32),
                (self.l_max, self.num_channels),
            )
        else:
            self.affine_weight = None

        if self.std_balance_degrees and self.l_max > 0:
            m = (self.l_max + 1) ** 2 - 1
            w = jnp.zeros((m, 1), dtype=jnp.float32)
            start = 0
            for l_number in range(1, self.l_max + 1):
                length = 2 * l_number + 1
                w = w.at[start : start + length, 0].set(1.0 / length)
                start += length
            self.balance_degree_weight = w / self.l_max
        else:
            self.balance_degree_weight = None

    @nn.compact
    def __call__(self, node_input: jnp.ndarray) -> jnp.ndarray:

        n, ltot, c = node_input.shape
        assert c == self.num_channels, "num_channels mismatch"

        outs = []

        feat_l0 = node_input[:, 0:1, :]
        feat_l0 = self.norm_l0(feat_l0)
        outs.append(feat_l0)

        if self.l_max > 0:
            num_m_components = (self.l_max + 1) ** 2
            feat_all = node_input[:, 1:num_m_components, :]

            if self.normalization == "norm":
                feat_stat = jnp.sum(feat_all**2, axis=1, keepdims=True)
            else:
                if self.std_balance_degrees and self.balance_degree_weight is not None:
                    feat_stat = jnp.einsum(
                        "nmc,ma->nac", feat_all**2, self.balance_degree_weight
                    )
                else:
                    feat_stat = jnp.mean(feat_all**2, axis=1, keepdims=True)

            scale = jnp.mean(feat_stat, axis=2, keepdims=True)
            scale = jnp.power(scale + self.eps, -0.5)

            for l_number in range(1, self.l_max + 1):
                start = l_number * l_number
                length = 2 * l_number + 1
                feat_l = node_input[:, start : start + length, :]
                if self.affine and self.affine_weight is not None:
                    w = self.affine_weight[l_number - 1][None, None, :]
                    feat_l = feat_l * (scale * w)
                else:
                    feat_l = feat_l * scale
                outs.append(feat_l)

        return jnp.concatenate(outs, axis=1)


class EquivariantRMSNormArraySphericalHarmonicsV2(nn.Module):
    l_max: int
    num_channels: int
    eps: float = 1e-12
    affine: bool = True
    normalization: Literal["norm", "component"] = "component"
    centering: bool = True
    std_balance_degrees: bool = True

    def setup(self):
        self.expand_index = get_l_to_all_m_expand_index(self.l_max)

        if self.affine:
            self.affine_weight = self.param(
                "affine_weight",
                lambda k, shape: jnp.ones(shape, jnp.float32),
                (self.l_max + 1, self.num_channels),
            )
            if self.centering:
                self.affine_bias = self.param(
                    "affine_bias",
                    lambda k, shape: jnp.zeros(shape, jnp.float32),
                    (self.num_channels,),
                )
            else:
                self.affine_bias = None
        else:
            self.affine_weight = None
            self.affine_bias = None

        if self.std_balance_degrees:
            k = (self.l_max + 1) ** 2
            w = jnp.zeros((k, 1), dtype=jnp.float32)
            for l_number in range(self.l_max + 1):
                start = l_number * l_number
                length = 2 * l_number + 1
                w = w.at[start : start + length, 0].set(1.0 / length)
            self.balance_degree_weight = w / (self.l_max + 1)
        else:
            self.balance_degree_weight = None

    @nn.compact
    def __call__(self, node_input: jnp.ndarray) -> jnp.ndarray:

        n, k, c = node_input.shape
        assert c == self.num_channels, "num_channels mismatch with input"

        x = node_input

        if self.centering:
            l0 = x[:, 0:1, :]
            l0_mean = jnp.mean(l0, axis=2, keepdims=True)
            l0_centered = l0 - l0_mean
            x = jnp.concatenate([l0_centered, x[:, 1:, :]], axis=1)

        if self.normalization == "norm":
            assert not self.std_balance_degrees, (
                "std_balance_degrees must be False when normalization='norm'"
            )
            stat_m = jnp.sum(x**2, axis=1, keepdims=True)
        else:
            if self.std_balance_degrees and self.balance_degree_weight is not None:
                stat_m = jnp.einsum("nkc,ka->nac", x**2, self.balance_degree_weight)
            else:
                stat_m = jnp.mean(x**2, axis=1, keepdims=True)

        scale = jnp.mean(stat_m, axis=2, keepdims=True)
        scale = jnp.power(scale + self.eps, -0.5)

        if self.affine and self.affine_weight is not None:
            weight_deg = self.affine_weight[None, :, :]
            weight_full = jnp.take(weight_deg, self.expand_index, axis=1)
            scale = scale * weight_full

        out = x * scale

        if self.affine and self.centering and (self.affine_bias is not None):
            out = out.at[:, 0:1, :].add(self.affine_bias[None, None, :])

        return out


def get_l_to_all_m_expand_index(l_max: int) -> jnp.ndarray:
    size = (l_max + 1) ** 2
    i = jnp.arange(size, dtype=jnp.int32)
    return jnp.floor(jnp.sqrt(i.astype(jnp.float32))).astype(jnp.int32)
