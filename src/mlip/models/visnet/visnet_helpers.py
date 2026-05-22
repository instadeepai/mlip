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

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import activation as act

# Setting eps=1e-05 to reproduce pytorch Layernorm
# See: https://github.com/cgarciae/nanoGPT-jax/blob/24fd60f987a946915e43c0000195bd73ddc34271/model.py#L95  # noqa: E501
LAYER_NORM_EPSILON = 1e-05
VEC_LAYER_NORM_EPSILON = 1e-12


class VecNormType(Enum):
    """Options for the VecLayerNorm of the ViSNet model."""

    RMS = "rms"
    MAX_MIN = "max_min"
    NONE = "none"


class VecLayerNorm(nn.Module):
    """Vector layer normalization used by the ViSNet model."""

    num_channels: int
    norm_type: VecNormType | str
    eps: float

    def _none_norm(self, vec: jax.Array) -> jax.Array:
        return vec

    def _rms_norm(self, vec: jax.Array) -> jax.Array:
        dist = jnp.sqrt(jnp.sum(vec**2, axis=1, keepdims=True) + self.eps)
        dist = jnp.clip(dist, min=self.eps)
        dist = jnp.sqrt(jnp.mean(dist**2, axis=-1))
        return vec / act.relu(dist).reshape(-1, 1, 1)

    def _max_min_norm(self, vec: jax.Array) -> jax.Array:
        dist = jnp.sqrt(jnp.sum(vec**2, axis=1, keepdims=True) + self.eps)
        direct = vec / jnp.clip(dist, min=self.eps)
        max_val = jnp.max(dist, axis=-1, keepdims=True)
        min_val = jnp.min(dist, axis=-1, keepdims=True)
        delta = max_val - min_val
        delta = delta + self.eps
        dist = (dist - min_val) / delta
        return act.relu(dist) * direct

    def __call__(self, vec: jax.Array) -> jax.Array:
        """Applies vector layer normalization to the input vector.

        Options for the norm_type are:
        - VecNormType.RMS: Root mean square normalization.
        - VecNormType.MAX_MIN: Max-min normalization.
        - VecNormType.NONE: No normalization.

        Args:
            vec: The input vector to normalize.

        Returns:
            The normalized vector.
        """

        # validate norm_type option
        norm_type = VecNormType(self.norm_type)

        if vec.shape[1] == 3 or vec.shape[1] == 8:
            if norm_type == VecNormType.RMS:
                norm_fn = self._rms_norm
            elif norm_type == VecNormType.MAX_MIN:
                norm_fn = self._max_min_norm
            elif norm_type == VecNormType.NONE:
                norm_fn = self._none_norm

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
