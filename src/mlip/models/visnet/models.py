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

from typing import Tuple

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import initializers

from mlip.data.dataset_info import DatasetInfo
from mlip.models.atomic_energies import get_atomic_energies
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.options import parse_activation
from mlip.models.visnet.blocks import (
    CosineCutoff,
    EdgeEmbedding,
    EquivariantScalar,
    NeighborEmbedding,
    Sphere,
    VecLayerNorm,
    parse_rbf_fn,
)
from mlip.models.visnet.config import VisnetConfig
from mlip.utils.safe_norm import safe_norm


class Visnet(MLIPNetwork):
    """The ViSNet model flax module. It is derived from the
    :class:`~mlip.models.mlip_network.MLIPNetwork` class.

    References:
        * Yusong Wang, Tong Wang, Shaoning Li, Xinheng He, Mingyu Li, Zun Wang,
          Nanning Zheng, Bin Shao, and Tie-Yan Liu. Enhancing geometric
          representations for molecules with equivariant vector-scalar interactive
          message passing. Nature Communications, 15(1), January 2024.
          ISSN: 2041-1723. URL: https://dx.doi.org/10.1038/s41467-023-43720-2.


    Attributes:
        config: Hyperparameters / configuration for the ViSNet model, see
                :class:`~mlip.models.visnet.config.VisnetConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = VisnetConfig

    config: VisnetConfig
    dataset_info: DatasetInfo

    @nn.compact
    def __call__(
        self,
        edge_vectors: jnp.ndarray,
        node_species: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> jnp.ndarray:

        r_max = self.dataset_info.cutoff_distance_angstrom

        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.atomic_energies_map)

        visnet_kwargs = dict(
            lmax=self.config.l_max,
            vecnorm_type=self.config.vecnorm_type,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            num_channels=self.config.num_channels,
            num_rbf=self.config.num_rbf,
            rbf_type="expnorm",
            trainable_rbf=self.config.trainable_rbf,
            activation=self.config.activation,
            attn_activation=self.config.attn_activation,
            cutoff=r_max,
            num_species=num_species,
        )

        representation_model = VisnetBlock(**visnet_kwargs)
        output_model = EquivariantScalar(
            self.config.num_channels, activation=self.config.activation
        )

        node_feats, vector_feats = representation_model(
            edge_vectors, node_species, senders, receivers
        )
        node_feats = output_model.pre_reduce(node_feats, vector_feats, node_species)
        node_feats = node_feats.squeeze(axis=-1)

        node_feats *= self.dataset_info.scaling_stdev
        node_feats = output_model.post_reduce(node_feats)

        node_feats += self.dataset_info.scaling_mean

        atomic_energies_ = get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species
        )
        atomic_energies_ = jnp.asarray(atomic_energies_)
        node_feats += atomic_energies_[node_species]  # [n_nodes, ]

        return node_feats


class VisnetBlock(nn.Module):
    lmax: int = 2
    vecnorm_type: str = "none"
    num_heads: int = 8
    num_layers: int = 9
    num_channels: int = 256
    num_rbf: int = 32
    rbf_type: str = "expnorm"
    trainable_rbf: bool = False
    activation: str = "silu"
    attn_activation: str = "silu"
    cutoff: float = 5.0
    num_species: int = 5

    def setup(self) -> None:
        self.node_embedding = nn.Embed(self.num_species, self.num_channels)
        self.radial_embedding = parse_rbf_fn(self.rbf_type)(
            self.cutoff, self.num_rbf, self.trainable_rbf
        )
        self.spherical_embedding = Sphere(self.lmax)

        self.neighbor_embedding = NeighborEmbedding(
            self.num_channels, self.cutoff, self.num_species
        )

        self.edge_embedding = EdgeEmbedding(self.num_channels)

        self.visnet_layers = [
            VisnetLayer(
                num_heads=self.num_heads,
                num_channels=self.num_channels,
                activation=self.activation,
                attn_activation=self.attn_activation,
                cutoff=self.cutoff,
                vecnorm_type=self.vecnorm_type,
                last_layer=i == self.num_layers - 1,
            )
            for i in range(self.num_layers)
        ]

        self.out_norm = nn.LayerNorm(epsilon=1e-05)
        self.vec_out_norm = VecLayerNorm(
            num_channels=self.num_channels,
            norm_type=self.vecnorm_type,
        )

    def __call__(
        self,
        edge_vectors: jnp.ndarray,  # [n_edges, 3]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        assert edge_vectors.ndim == 2 and edge_vectors.shape[1] == 3
        assert node_species.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert edge_vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        # Calculate distances
        distances = safe_norm(edge_vectors, axis=-1)

        # Embedding Layers
        node_feats = self.node_embedding(node_species)  # Is that necessary?

        # Seems like doubled from within the neighbor embedding module
        edge_feats = self.radial_embedding(distances)

        spherical_feats = self.spherical_embedding(
            edge_vectors / (distances[:, None] + 1e-8)
        )
        node_feats = self.neighbor_embedding(
            node_species, node_feats, senders, receivers, distances, edge_feats
        )  # h in paper

        edge_feats = self.edge_embedding(
            senders, receivers, edge_feats, node_feats
        )  # f in paper

        vec_shape = (
            node_feats.shape[0],
            ((self.lmax + 1) ** 2) - 1,
            node_feats.shape[1],
        )
        vector_feats = jnp.zeros(vec_shape, dtype=node_feats.dtype)

        assert self.num_channels % self.num_heads == 0, (
            f"The number of hidden channels ({self.num_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({self.num_heads})"
        )

        for i in range(self.num_layers):
            diff_node_feats, diff_edge_feats, diff_vector_feats = self.visnet_layers[i](
                node_feats,
                edge_feats,
                vector_feats,
                distances,
                senders,
                receivers,
                spherical_feats,
            )

            node_feats += diff_node_feats
            edge_feats += diff_edge_feats
            vector_feats += diff_vector_feats

        node_feats = self.out_norm(node_feats)
        vector_feats = self.vec_out_norm(vector_feats)
        return node_feats, vector_feats


class VisnetLayer(nn.Module):
    num_heads: int
    num_channels: int
    activation: str
    attn_activation: str
    cutoff: float
    vecnorm_type: str
    last_layer: bool = False

    def setup(self):
        assert self.num_channels % self.num_heads == 0, (
            f"The number of hidden channels ({self.num_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({self.num_heads})"
        )
        self.head_dim = self.num_channels // self.num_heads

        # Setting eps=1e-05 to reproduce pytorch Layernorm
        # See: https://github.com/cgarciae/nanoGPT-jax/blob/24fd60f987a946915e43c0000195bd73ddc34271/model.py#L95  # noqa: E501
        self.layernorm = nn.LayerNorm(epsilon=1e-05)
        self.vec_layernorm = VecLayerNorm(
            num_channels=self.num_channels,
            norm_type=self.vecnorm_type,
        )
        self.act = parse_activation(self.activation)
        self.attn_act = parse_activation(self.attn_activation)
        self.cutoff_fn = CosineCutoff(self.cutoff)

        self.vec_proj = nn.Dense(
            features=self.num_channels * 3,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
        )
        self.q_proj = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.k_proj = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.v_proj = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.dk_proj = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.dv_proj = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.s_proj = nn.Dense(
            features=self.num_channels * 2,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )
        self.o_proj = nn.Dense(
            features=self.num_channels * 3,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros_init(),
        )

        if not self.last_layer:
            self.f_proj = nn.Dense(
                features=self.num_channels,
                kernel_init=initializers.xavier_uniform(),
                bias_init=initializers.zeros_init(),
            )
            self.w_src_proj = nn.Dense(
                features=self.num_channels,
                use_bias=False,
                kernel_init=initializers.xavier_uniform(),
            )
            self.w_trg_proj = nn.Dense(
                features=self.num_channels,
                use_bias=False,
                kernel_init=initializers.xavier_uniform(),
            )

    def message_fn(
        self,
        q_i: jnp.ndarray,
        k_j: jnp.ndarray,
        v_j: jnp.ndarray,
        vec_j: jnp.ndarray,
        dk: jnp.ndarray,
        dv: jnp.ndarray,
        r_ij: jnp.ndarray,
        d_ij: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        attn = (q_i * k_j * dk).sum(axis=-1)
        attn = self.attn_act(attn) * jnp.expand_dims(self.cutoff_fn(r_ij), 1)

        v_j = v_j * dv
        v_j = (v_j * jnp.expand_dims(attn, 2)).reshape(-1, self.num_channels)

        s1, s2 = jnp.split(self.act(self.s_proj(v_j)), [self.num_channels], axis=1)
        vec_j = vec_j * jnp.expand_dims(s1, 1) + jnp.expand_dims(
            s2, 1
        ) * jnp.expand_dims(d_ij, 2)

        return v_j, vec_j

    def edge_update(
        self,
        vec_i: jnp.ndarray,
        vec_j: jnp.ndarray,
        d_ij: jnp.ndarray,
        f_ij: jnp.ndarray,
    ) -> jnp.ndarray:
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(axis=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def __call__(
        self,
        node_feats: jnp.ndarray,
        edge_feats: jnp.ndarray,
        vector_feats: jnp.ndarray,
        distances: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        d_ij: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        node_feats = self.layernorm(node_feats)
        vector_feats = self.vec_layernorm(vector_feats)
        q_feats = self.q_proj(node_feats)  # Correspond to Wq weights in the paper
        k_feats = self.k_proj(node_feats)  # Correspond to Wk weights in the paper
        v_feats = self.v_proj(node_feats)  # Correspond to Wv weights in the paper
        dk_feats = self.dk_proj(edge_feats)  # Correspond to Dk weights in the paper
        dv_feats = self.dv_proj(edge_feats)  # Correspond to Dv weights in the paper
        # Reshape the outputs to include the num_heads dimension
        q_feats = jnp.reshape(q_feats, (-1, self.num_heads, self.head_dim))
        k_feats = jnp.reshape(k_feats, (-1, self.num_heads, self.head_dim))
        v_feats = jnp.reshape(v_feats, (-1, self.num_heads, self.head_dim))
        dk_feats = jnp.reshape(self.act(dk_feats), (-1, self.num_heads, self.head_dim))
        dv_feats = jnp.reshape(self.act(dv_feats), (-1, self.num_heads, self.head_dim))

        projected_vec = self.vec_proj(vector_feats)
        split_sizes = [self.num_channels] * 3
        # we use numpy (instead of jax) here to ensure split_sizes is static
        # and not a tracer
        split_indices = np.cumsum(np.array(split_sizes[:-1]))
        vec1, vec2, vec3 = jnp.split(projected_vec, split_indices, axis=-1)
        vec_dot = jnp.sum(vec1 * vec2, axis=1)

        # Apply message function for each edge
        q_i = q_feats[receivers, :, :]
        k_j = k_feats[senders, :, :]
        v_j = v_feats[senders, :, :]
        vec_j = vector_feats[senders, :, :]

        node_msgs, vec_msgs = self.message_fn(
            q_i, k_j, v_j, vec_j, dk_feats, dv_feats, distances, d_ij
        )
        # Aggregate the messages
        node_feats = jax.ops.segment_sum(
            node_msgs, receivers, num_segments=node_feats.shape[0]
        )
        vec_out = jax.ops.segment_sum(
            vec_msgs, receivers, num_segments=node_feats.shape[0]
        )

        o1, o2, o3 = jnp.split(self.o_proj(node_feats), split_indices, axis=1)

        dx = vec_dot * o2 + o3
        dvec = vec3 * jnp.expand_dims(o1, 1) + vec_out

        if not self.last_layer:
            df_ij = self.edge_update(
                vector_feats[receivers, :], vec_j, d_ij, edge_feats
            )
            return dx, df_ij, dvec
        else:
            return dx, jnp.zeros_like(edge_feats), dvec

    @nn.nowrap
    def vector_rejection(self, vec, d_ij):
        # Implement vector rejection logic using JAX
        vec_proj = (vec * jnp.expand_dims(d_ij, 2)).sum(axis=1, keepdims=True)
        return vec - vec_proj * jnp.expand_dims(d_ij, 2)
