# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Su Qun (https://github.com/lingcon01/LiTEN) under MIT license.

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers

from mlip.data.dataset_info import DatasetInfo
from mlip.models.cutoff import CosineCutoff
from mlip.models.radial_basis import parse_radial_basis
from mlip.models.atomic_energies import get_atomic_energies
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.options import parse_activation
from mlip.models.visnet.blocks import VecLayerNorm
from mlip.models.liten.config import LitenConfig
from mlip.utils.safe_norm import safe_norm


class Liten(MLIPNetwork):
    """The LiTEN model flax module. It is derived from the
    :class:`~mlip.models.mlip_network.MLIPNetwork` class.

    References:
        * Qun Su, Kai Zhu, Qiaolin Gou, Jintu Zhang, Renling Hu, Yurong Li,
          Yongze Wang, Hui Zhang, Ziyi You, Linlong Jiang, Yu Kang, Jike Wang,
          Chang-Yu Hsieh and Tingjun Hou. A Scalable and Quantum-Accurate
          Foundation Model for Biomolecular Force Field via Linearly Tensorized
          Quadrangle Attention. arXiv, Jul 2025.
          URL: https://arxiv.org/abs/2507.00884.

    Attributes:
        config: Hyperparameters / configuration for the LiTEN model, see
                :class:`~mlip.models.liten.config.LitenConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = LitenConfig

    config: LitenConfig
    dataset_info: DatasetInfo

    @nn.compact
    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        **_kwargs, # ignore any additional kwargs
    ) -> jax.Array:

        r_max = self.dataset_info.cutoff_distance_angstrom

        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.atomic_energies_map)

        liten_kwargs = dict(
            vecnorm_type=self.config.vecnorm_type,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            num_channels=self.config.num_channels,
            num_rbf=self.config.num_rbf,
            rbf_type="expnorm",
            trainable_rbf=self.config.trainable_rbf,
            activation=self.config.activation,
            cutoff=r_max,
            num_species=num_species,
        )

        representation_model = LitenBlock(**liten_kwargs)
        node_energies = representation_model(
            edge_vectors, node_species, senders, receivers
        )
        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies

        atomic_energies_ = get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species
        )
        atomic_energies_ = jnp.asarray(atomic_energies_)
        node_energies += atomic_energies_[node_species]  # [n_nodes, ]

        return node_energies


class LitenBlock(nn.Module):
    vecnorm_type: str = "none"
    num_heads: int = 8
    num_layers: int = 9
    num_channels: int = 256
    num_rbf: int = 32
    rbf_type: str = "expnorm"
    trainable_rbf: bool = False
    activation: str = "silu"
    cutoff: float = 5.0
    num_species: int = 5

    def setup(self) -> None:
        self.node_embedding = nn.Embed(self.num_species, self.num_channels)
        self.radial_embedding = parse_radial_basis(self.rbf_type)(
            self.cutoff, self.num_rbf, self.trainable_rbf
        )

        self.edge_embedding = nn.Dense(self.num_channels)

        self.liten_layers = [
            LitenLayer(
                num_heads=self.num_heads,
                num_channels=self.num_channels,
                activation=self.activation,
                cutoff=self.cutoff,
                vecnorm_type=self.vecnorm_type,
                last_layer=i == self.num_layers - 1,
                first_layer=i == 0,
            )
            for i in range(self.num_layers)
        ]

        self.out_norm = nn.LayerNorm(epsilon=1e-05)
        self.readout_energy = nn.Sequential(
            [
                nn.Dense(self.num_channels // 2),
                parse_activation(self.activation),
                nn.Dense(1),
            ]
        )

    def __call__(
        self,
        edge_vectors: jax.Array,  # [n_edges, 3]
        node_species: jax.Array,  # [n_nodes] int between 0 and num_species-1
        senders: jax.Array,  # [n_edges]
        receivers: jax.Array,  # [n_edges]
    ) -> jax.Array:
        assert edge_vectors.ndim == 2 and edge_vectors.shape[1] == 3
        assert node_species.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert edge_vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        # Calculate distances
        distances = safe_norm(edge_vectors, axis=-1)

        # Normalize edge vectors
        edge_vectors = edge_vectors / (distances[:, None] + 1e-8)

        # Embedding Layers
        node_feats = self.node_embedding(node_species)

        edge_feats = self.radial_embedding(distances)
        # Cosine cutoff is seperated from radial basis function.
        edge_feats = edge_feats * CosineCutoff(self.cutoff)(distances)
        edge_feats = self.edge_embedding(edge_feats)

        # It will be [n_nodes, 3, num_channels]
        vector_feats = None

        assert self.num_channels % self.num_heads == 0, (
            f"The number of hidden channels ({self.num_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({self.num_heads})"
        )

        for layer in self.liten_layers:
            node_feats, edge_feats, vector_feats = layer(
                node_feats,
                edge_feats,
                vector_feats,
                distances,
                senders,
                receivers,
                edge_vectors,
            )

        node_feats = self.out_norm(node_feats)
        node_energies = self.readout_energy(node_feats).squeeze(-1)

        return node_energies


class LitenLayer(nn.Module):
    num_heads: int
    num_channels: int
    activation: str
    cutoff: float
    vecnorm_type: str
    last_layer: bool = False
    first_layer: bool = False

    def setup(self):
        assert self.num_channels % self.num_heads == 0, (
            f"The number of hidden channels ({self.num_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({self.num_heads})"
        )
        self.head_dim = self.num_channels // self.num_heads

        # Setting eps=1e-05 to reproduce pytorch Layernorm
        self.layernorm = nn.LayerNorm(epsilon=1e-05)
        self.vec_layernorm = VecLayerNorm(
            num_channels=self.num_channels,
            norm_type=self.vecnorm_type,
        )
        self.act = parse_activation(self.activation)
        self.cutoff_fn = CosineCutoff(self.cutoff)

        self.alpha = self.param(
            "alpha",
            initializers.xavier_uniform(),
            (
                1,
                self.num_heads,
                self.head_dim,
            ),
        )

        self.vec_linear = nn.Dense(
            features=self.num_channels * 2,
            use_bias=False,
            kernel_init=initializers.xavier_uniform(),
        )
        self.node_linear = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
        )
        self.edge_linear = nn.Dense(
            features=self.num_channels,
            kernel_init=initializers.xavier_uniform(),
        )
        self.part_linear1 = nn.Dense(
            features=self.num_channels if self.first_layer else self.num_channels * 2,
            kernel_init=initializers.xavier_uniform(),
        )
        self.part_linear2 = nn.Dense(
            features=self.num_channels * 2 if self.last_layer else self.num_channels * 3,
            kernel_init=initializers.xavier_uniform(),
        )

        if not (self.last_layer or self.first_layer):
            self.cross_linear = nn.Dense(
                features=self.num_channels,
                use_bias=False,
                kernel_init=initializers.xavier_uniform(),
            )
            self.f_linear = nn.Dense(
                features=self.num_channels,
                kernel_init=initializers.xavier_uniform(),
            )

    def message_fn(
        self,
        node_feats: jax.Array,
        edge_feats: jax.Array,
        vector_feats: jax.Array | None,
        distances: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        edge_vectors: jax.Array,
    ):
        edge_feats = self.act(self.edge_linear(edge_feats)).reshape(
            -1, self.num_heads, self.head_dim
        )
        node_feats = self.node_linear(node_feats).reshape(-1, self.num_heads, self.head_dim)
        attn = node_feats[receivers] + node_feats[senders] + edge_feats
        attn = self.act(attn) * self.alpha
        attn = attn.sum(axis=-1) * self.cutoff_fn(distances)[:, None]
        attn = attn[:, :, None]

        n_nodes = len(node_feats)
        node_feats = node_feats[senders] * edge_feats
        node_feats = (node_feats * attn).reshape(-1, self.num_channels)

        node_sca = self.act(self.part_linear1(node_feats))[:, None] # [n_edges, 1, 2*num_channels]
        if self.first_layer:
            vector_feats = node_sca * edge_vectors[:, :, None]
        else:
            node_sca1, node_sca2 = jnp.split(node_sca, 2, axis=2)
            vector_feats = (
                vector_feats[senders] * node_sca1 + node_sca2 * edge_vectors[:, :, None]
            )

        node_feats = jax.ops.segment_sum(node_feats, receivers, num_segments=n_nodes)
        vector_feats = jax.ops.segment_sum(vector_feats, receivers, num_segments=n_nodes)

        return node_feats, vector_feats

    def edge_update(
        self,
        vector_feats: jax.Array, # [n_nodes, 3, num_channels]
        edge_feats: jax.Array, # [n_edges, num_channels]
        senders: jax.Array, # [n_edges]
        receivers: jax.Array, # [n_edges]
        edge_vectors: jax.Array, # [n_edges, 3]
    ):
        vector_feats = self.cross_linear(vector_feats)

        vec_cross_i = jnp.cross(vector_feats[senders], edge_vectors[:, :, None], axis=1)
        vec_cross_j = jnp.cross(vector_feats[receivers], edge_vectors[:, :, None], axis=1)
        sum_phi = jnp.sum(vec_cross_i * vec_cross_j, axis=1)

        diff_edge_feats = self.act(self.f_linear(edge_feats)) * sum_phi

        return diff_edge_feats

    def node_update(
        self,
        node_feats: jax.Array,
        vector_feats: jax.Array,
    ):
        vec1, vec2 = jnp.split(self.vec_linear(vector_feats), 2, axis=-1)
        vec_tri = jnp.sum(vec1 * vec2, axis=1)

        norm_vec = jnp.sqrt(jnp.sum(vec2 ** 2, axis=-2) + 1e-16)
        vec_qua = norm_vec ** 3

        node_feats = self.part_linear2(node_feats)

        if self.last_layer:
            sca1, sca2 = jnp.split(node_feats, 2, axis=1)
        else:
            sca1, sca2, sca3 = jnp.split(node_feats, 3, axis=1)

        diff_scalar = (vec_qua + vec_tri) * sca1 + sca2

        if self.last_layer:
            return diff_scalar

        diff_vector = vec1 * sca3[:, None]
        return diff_scalar, diff_vector

    def __call__(
        self,
        node_feats: jax.Array,
        edge_feats: jax.Array,
        vector_feats: jax.Array,
        distances: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        edge_vectors: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        scalar_out = self.layernorm(node_feats)

        if not self.first_layer:
            vector_feats = self.vec_layernorm(vector_feats)

        scalar_out, vector_out = self.message_fn(
            scalar_out, edge_feats, vector_feats, distances, senders, receivers, edge_vectors
        )

        if not (self.last_layer or self.first_layer):
            diff_edge_feats = self.edge_update(
                vector_feats, edge_feats, senders, receivers, edge_vectors
            )
            edge_feats = edge_feats + diff_edge_feats

        node_feats = node_feats + scalar_out

        if self.first_layer:
            vector_feats = vector_out
        else:
            vector_feats = vector_feats + vector_out

        diff_scalar = self.node_update(node_feats, vector_feats)

        if not self.last_layer:
            diff_scalar, diff_vector = diff_scalar
            vector_feats = vector_feats + diff_vector

        node_feats = node_feats + diff_scalar

        return node_feats, edge_feats, vector_feats
