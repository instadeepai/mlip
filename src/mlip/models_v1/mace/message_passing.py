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

from typing import Callable, Literal

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from mlip.models_v1.blocks import path_major_to_channel_major_permutation
from mlip.models_v1.version_compatibility import VERSION
from mlip.utils.jax_utils import segment_sum


def _v2_to_v1_permutation(irreps: e3nn.Irreps, l_max: int) -> np.ndarray:
    """Permutation mapping channel indices from v2 to v1 orderings.

    Within each irrep block, coupling paths for a given output irrep
    are enumerated in a different order by v1 (e3nn concatenate + regroup)
    and v2 (e3j TensorProduct). This function reorders v2 multiplicity
    indices in E3NN layout to match v1 path ordering.
    """
    M = irreps.mul_gcd
    perm: list[int] = []
    offset = 0

    for mul, ir in irreps:
        n = mul
        indices = list(range(offset, offset + n))

        if str(ir) == "1o":
            indices = indices[M : 2 * M] + indices[:M] + indices[2 * M :]
        elif str(ir) == "2e":
            if l_max == 2:
                indices = indices[2 * M : 3 * M] + indices[: 2 * M] + indices[3 * M :]
            elif l_max == 3:
                indices = indices[3 * M : 4 * M] + indices[: 3 * M] + indices[4 * M :]
        elif str(ir) == "3o":
            indices = indices[4 * M : 5 * M] + indices[: 4 * M] + indices[5 * M :]

        perm.extend(indices)
        offset += n

    return np.array(perm)


class MessagePassingConvolution(nn.Module):
    avg_num_neighbors: float
    target_irreps: e3nn.Irreps
    l_max: int
    activation: Callable
    species_embedding_dim: int | None = None
    version: Literal[1, 2] = VERSION
    deterministic_scatter_ops: bool = False

    @nn.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
        edge_species_feat: jnp.ndarray
        | None = None,  # [n_edges, species_embedding_dim * 3]
    ) -> e3nn.IrrepsArray:
        assert node_feats.ndim == 2
        if self.species_embedding_dim is not None:
            assert edge_species_feat is not None

        target_irreps = e3nn.Irreps(self.target_irreps)

        # PATCH: don't apply sign on vectors to match v2
        if self.version == 1:
            vectors = -vectors

        messages = node_feats[senders]
        messages = e3nn.concatenate([
            messages.filter(target_irreps),
            e3nn.tensor_product(
                messages,
                e3nn.spherical_harmonics(range(1, self.l_max + 1), vectors, True),
                filter_ir_out=target_irreps,
            ),
        ]).regroup()  # [n_edges, irreps]

        if self.version == 2:
            # PATCH: Reorder messages from v1 multiplicity ordering to v2 ordering.
            v2_to_v1 = _v2_to_v1_permutation(messages.irreps, self.l_max)
            v1_to_v2 = np.argsort(v2_to_v1)
            # Expand per-irrep permutation to per-component permutation
            dims = np.array([ir.dim for mul, ir in messages.irreps for _ in range(mul)])
            comp_offsets = np.concatenate([[0], np.cumsum(dims)])
            comp_perm = np.concatenate([
                np.arange(comp_offsets[i], comp_offsets[i + 1]) for i in v1_to_v2
            ])
            messages = e3nn.IrrepsArray(messages.irreps, messages.array[:, comp_perm])

        mix = e3nn.flax.MultiLayerPerceptron(
            3 * [64] + [messages.irreps.num_irreps],
            self.activation,
            gradient_normalization=1.0,
            output_activation=False,
        )(radial_embedding)  # [n_edges, num_irreps]

        if self.species_embedding_dim is not None:
            mix_species = e3nn.flax.MultiLayerPerceptron(
                3 * [64] + [messages.irreps.num_irreps],
                self.activation,
                gradient_normalization=1.0,
                output_activation=False,
                with_bias=True,
            )(edge_species_feat)  # [n_edges, num_irreps]
            mix = jax.vmap(jnp.multiply)(mix.array, mix_species)

        messages = messages * mix  # [n_edges, irreps]

        node_feats = segment_sum(
            messages,
            receivers,
            num_segments=node_feats.shape[0],
            deterministic=self.deterministic_scatter_ops,
        )  # [n_nodes, irreps]

        if self.version == 2:
            # PATCH: Convert path-major to channel-major sorting of isomorphic irreps
            #        to match TRAILING_CHANNELS and LEADING_CHANNELS outputs of v2.
            num_channels = node_feats.irreps.mul_gcd
            perm = path_major_to_channel_major_permutation(
                node_feats.irreps, num_channels
            )
            node_feats = e3nn.IrrepsArray(node_feats.irreps, node_feats.array[:, perm])

        return node_feats / self.avg_num_neighbors
