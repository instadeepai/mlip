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
from abc import ABCMeta, abstractmethod
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import activation as act
from flax.linen import initializers

from mlip.models.options import VecNormType, VisnetRBF, parse_activation


class CosineCutoff(nn.Module):
    cutoff: float

    @nn.compact
    def __call__(self, distances: jnp.ndarray) -> jnp.ndarray:
        cutoffs = 0.5 * (jnp.cos(distances * jnp.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).astype(jnp.float32)
        return cutoffs


class ExpNormalSmearing(nn.Module):
    cutoff: float = 5.0
    num_rbf: int = 50
    trainable: bool = True

    def setup(self):
        self.alpha = 5.0 / self.cutoff
        means, betas = self._initial_params()
        if self.trainable:
            self.means = self.param(
                "means", nn.initializers.constant(means), (self.num_rbf,)
            )
            self.betas = self.param(
                "betas", nn.initializers.constant(betas), (self.num_rbf,)
            )
        else:
            self.means = means
            self.betas = betas
        self.cutoff_fn = CosineCutoff(self.cutoff)

    def _initial_params(self):
        start_value = jnp.exp(-self.cutoff)
        means = jnp.linspace(start_value, 1, self.num_rbf)
        betas = jnp.full((self.num_rbf,), (2 / self.num_rbf * (1 - start_value)) ** -2)
        return means, betas

    def __call__(self, dist: jnp.ndarray) -> jnp.ndarray:
        dist = dist[..., jnp.newaxis]
        cutoffs = self.cutoff_fn(dist)
        return cutoffs * jnp.exp(
            (-1 * self.betas) * (jnp.exp(self.alpha * (-dist)) - self.means) ** 2
        )


class GaussianSmearing(nn.Module):
    cutoff: float = 5.0
    num_rbf: int = 50
    trainable: bool = True

    def setup(self):
        offset, coeff = self._initial_params()
        if self.trainable:
            self.offset = self.param(
                "offset", nn.initializers.constant(offset), (self.num_rbf,)
            )
            self.coeff = self.param("coeff", nn.initializers.constant(coeff), ())
        else:
            self.offset = offset
            self.coeff = coeff
        self.cutoff_fn = CosineCutoff(self.cutoff)

    def _initial_params(self):
        offset = jnp.linspace(0, self.cutoff, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def __call__(self, dist: jnp.ndarray) -> jnp.ndarray:
        dist = dist[..., jnp.newaxis] - self.offset
        cutoffs = self.cutoff_fn(dist)
        return cutoffs * jnp.exp(self.coeff * jnp.square(dist))


def parse_rbf_fn(rbf_type: VisnetRBF | str) -> Callable:
    # Mapping of RBF class names to their Flax classes
    rbf_class_mapping = {
        VisnetRBF.GAUSS: GaussianSmearing,
        VisnetRBF.EXPNORM: ExpNormalSmearing,
    }
    return rbf_class_mapping[VisnetRBF(rbf_type)]


class Sphere(nn.Module):
    degree: int = 2

    def __call__(self, edge_vec: jnp.ndarray) -> jnp.ndarray:
        edge_sh = self._spherical_harmonics(
            edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2]
        )
        return edge_sh

    @nn.nowrap
    def _spherical_harmonics(
        self, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        sh_1_0, sh_1_1, sh_1_2 = x, y, z

        if self.degree == 1:
            return jnp.stack([sh_1_0, sh_1_1, sh_1_2], axis=-1)

        sh_2_0 = jnp.sqrt(3.0) * x * z
        sh_2_1 = jnp.sqrt(3.0) * x * y
        y2 = y**2
        x2z2 = x**2 + z**2
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = jnp.sqrt(3.0) * y * z
        sh_2_4 = jnp.sqrt(3.0) / 2.0 * (z**2 - x**2)

        if self.degree == 2:
            return jnp.stack(
                [sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4],
                axis=-1,
            )


class NeighborEmbedding(nn.Module):
    num_channels: int
    cutoff: float
    num_species: int = 100

    def setup(self):
        self.embedding = nn.Embed(self.num_species, self.num_channels)
        self.distance_proj = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.combine = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.cutoff_fn = CosineCutoff(cutoff=self.cutoff)

    def message_fn(self, x_j, w):
        return x_j * w

    def __call__(self, node_z, node_feats, senders, receivers, edge_weight, edge_feats):

        C = self.cutoff_fn(edge_weight)
        W = self.distance_proj(edge_feats) * C[:, jnp.newaxis]

        x_neighbors = self.embedding(node_z)
        x_j = x_neighbors[senders]

        node_msgs = self.message_fn(x_j, W)
        aggregated_msgs = jax.ops.segment_sum(
            node_msgs, receivers, num_segments=node_feats.shape[0]
        )

        # Update between x and aggregated_messages over neighbors
        node_feats = self.combine(
            jnp.concatenate([node_feats, aggregated_msgs], axis=1)
        )
        return node_feats


class EdgeEmbedding(nn.Module):
    num_channels: int

    def setup(self):
        self.edge_proj = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )

    def message_fn(self, x_i, x_j, edge_feats):
        return (x_i + x_j) * self.edge_proj(edge_feats)

    def __call__(self, senders, receivers, edge_attr, x):
        x_j = x[senders]
        x_i = x[receivers]

        edge_messages = self.message_fn(x_i, x_j, edge_attr)
        return edge_messages


class VecLayerNorm(nn.Module):
    num_channels: int
    norm_type: VecNormType | str = "max_min"
    eps: float = 1e-12

    def none_norm(self, vec):
        return vec

    def rms_norm(self, vec):
        dist = jnp.sqrt(jnp.sum(vec**2, axis=1, keepdims=True) + self.eps)
        dist = jnp.clip(dist, a_min=self.eps)
        dist = jnp.sqrt(jnp.mean(dist**2, axis=-1))
        return vec / act.relu(dist).reshape(-1, 1, 1)

    def max_min_norm(self, vec):
        dist = jnp.sqrt(jnp.sum(vec**2, axis=1, keepdims=True) + self.eps)
        direct = vec / jnp.clip(dist, a_min=self.eps)
        max_val = jnp.max(dist, axis=-1, keepdims=True)
        min_val = jnp.min(dist, axis=-1, keepdims=True)
        delta = max_val - min_val
        delta = delta + self.eps
        dist = (dist - min_val) / delta
        return act.relu(dist) * direct

    def __call__(self, vec):
        # validate norm_type option
        norm_type = VecNormType(self.norm_type)

        if vec.shape[1] == 3 or vec.shape[1] == 8:
            if norm_type == VecNormType.RMS:
                norm_fn = self.rms_norm
            elif norm_type == VecNormType.MAX_MIN:
                norm_fn = self.max_min_norm
            elif norm_type == VecNormType.NONE:
                norm_fn = self.none_norm

            if vec.shape[1] == 3:
                vec = norm_fn(vec)
            elif vec.shape[1] == 8:
                vec1, vec2 = jnp.split(vec, indices_or_sections=[3], axis=1)
                vec1 = norm_fn(vec1)
                vec2 = norm_fn(vec2)
                vec = jnp.concatenate([vec1, vec2], axis=1)

            return vec  # We have removed VecNorm trainability
        else:
            raise ValueError("VecLayerNorm only supports 3 or 8 channels")


class GatedEquivariantBlock(nn.Module):
    num_channels: int
    out_channels: int
    intermediate_channels: int = None
    activation: str = "silu"
    scalar_activation: bool = False

    def setup(self):
        if self.intermediate_channels is None:
            intermediate_channels = self.num_channels

        self.vec1_proj = nn.Dense(
            self.num_channels,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
        )
        self.vec2_proj = nn.Dense(
            self.out_channels, use_bias=False, kernel_init=initializers.xavier_uniform()
        )

        self.update_net = nn.Sequential(
            [
                nn.Dense(
                    intermediate_channels,
                    kernel_init=initializers.xavier_uniform(),
                    bias_init=initializers.zeros_init(),
                ),
                parse_activation(self.activation),
                nn.Dense(
                    self.out_channels * 2,
                    kernel_init=initializers.xavier_uniform(),
                    bias_init=initializers.zeros_init(),
                ),
            ]
        )

        if self.scalar_activation:
            self.act = parse_activation(
                self.activation
            )  # Assuming direct call is intended, otherwise needs adjustment

    def __call__(self, x, v):
        vec1 = jnp.linalg.norm(self.vec1_proj(v) + 1e-8, axis=-2)
        vec2 = self.vec2_proj(v)
        x = jnp.concatenate([x, vec1], axis=-1)
        x = self.update_net(x)
        x, v = jnp.split(x, 2, axis=-1)
        v = jnp.expand_dims(v, axis=1) * vec2

        if self.scalar_activation:
            x = self.act(x)
        return x, v


class OutputModel(nn.Module, metaclass=ABCMeta):
    def setup(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        pass  # Must be implemented in subclasses

    def post_reduce(self, x):
        return x


class Scalar(OutputModel):
    num_channels: int
    activation: str = "silu"

    def setup(self):

        self.output_network = nn.Sequential(
            [
                nn.Dense(
                    self.num_channels // 2,
                    kernel_init=initializers.xavier_uniform(),
                    bias_init=initializers.zeros_init(),
                ),
                parse_activation(self.activation),
                nn.Dense(
                    1,
                    kernel_init=initializers.xavier_uniform(),
                    bias_init=initializers.zeros_init(),
                ),
            ]
        )

    def pre_reduce(self, x, v, z=None, pos=None, batch=None):
        return self.output_network(x)


class EquivariantScalar(OutputModel):
    num_channels: int
    activation: str = "silu"
    allow_prior_model: bool = True

    def setup(self):
        self.output_network = [
            GatedEquivariantBlock(
                self.num_channels,
                self.num_channels // 2,
                activation=self.activation,
                scalar_activation=True,
            ),
            GatedEquivariantBlock(
                self.num_channels // 2,
                1,
                activation=self.activation,
                scalar_activation=False,
            ),
        ]

    def pre_reduce(self, x, v, z=None, pos=None, batch=None):
        for layer in self.output_network:
            x, v = layer(x, v)

        # include v in output to make sure all parameters have a gradient
        return x + jnp.sum(v) * 0
