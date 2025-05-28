# MIT License
# Copyright (c) 2022 mace-jax
# See https://github.com/ACEsuit/mace-jax/blob/main/MIT.md
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

from typing import Callable

import e3nn_jax as e3nn
import flax.linen as nn
import jax.numpy as jnp


class MessagePassingConvolution(nn.Module):
    avg_num_neighbors: float
    target_irreps: e3nn.Irreps
    l_max: int
    activation: Callable

    @nn.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:
        assert node_feats.ndim == 2

        target_irreps = e3nn.Irreps(self.target_irreps)

        messages = node_feats[senders]
        messages = e3nn.concatenate(
            [
                messages.filter(target_irreps),
                e3nn.tensor_product(
                    messages,
                    e3nn.spherical_harmonics(range(1, self.l_max + 1), -vectors, True),
                    filter_ir_out=target_irreps,
                ),
            ]
        ).regroup()  # [n_edges, irreps]
        mix = e3nn.flax.MultiLayerPerceptron(
            3 * [64] + [messages.irreps.num_irreps],
            self.activation,
            gradient_normalization=1.0,
            output_activation=False,
        )(
            radial_embedding
        )  # [n_edges, num_irreps]
        messages = messages * mix  # [n_edges, irreps]

        zeros = e3nn.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats / self.avg_num_neighbors
