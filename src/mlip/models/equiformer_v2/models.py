# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Yi-Lun Liao (https://github.com/atomicarchitects/equiformer_v2) under MIT license.

import flax.linen as nn
import jax
import jax.numpy as jnp

from mlip.data.dataset_info import DatasetInfo
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.atomic_energies import get_atomic_energies
from mlip.models.radial_basis import GaussianBasis
from mlip.models.equiformer_v2.config import EquiformerV2Config
from mlip.models.equiformer_v2.blocks import (
    SO3Linear, EdgeDegreeEmbedding, GraphDropPath
)
from mlip.models.equiformer_v2.layernorm import LayerNormType, parse_layernorm
from mlip.models.equiformer_v2.transform import (
    WignerMats, get_wigner_mats
)
from mlip.models.equiformer_v2.transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
)
from mlip.models.equiformer_v2.utils import (
    AttnActType,
    FeedForwardType,
    get_mapping_coeffs,
)
from mlip.utils.safe_norm import safe_norm


class EquiformerV2(MLIPNetwork):
    """The EquiformerV2 model flax module. It is derived from the
    :class:`~mlip.models.mlip_network.MLIPNetwork` class.

    References:
        * Yi-Lun Liao, Brandon Wood, Abhishek Das and Tess Smidt. EquiformerV2:
          Improved Equivariant Transformer for Scaling to Higher-Degree 
          Representations. International Conference on Learning Representations (ICLR),
          January 2024. URL: https://openreview.net/forum?id=mCOBKZmrzD.

    Attributes:
        config: Hyperparameters / configuration for the EquiformerV2 model, see
                :class:`~mlip.models.equiformer_v2.config.EquiformerV2Config`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """


    Config = EquiformerV2Config

    config: EquiformerV2Config
    dataset_info: DatasetInfo

    @nn.compact
    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        n_node: jax.Array | None = None, # For dropout, can be None for eval
        training: bool = False,
    ) -> jax.Array:
        r_max = self.dataset_info.cutoff_distance_angstrom

        avg_num_neighbors = self.config.avg_num_neighbors
        if avg_num_neighbors is None:
            avg_num_neighbors = self.dataset_info.avg_num_neighbors

        avg_num_nodes = self.config.avg_num_nodes
        if avg_num_nodes is None:
            avg_num_nodes = self.dataset_info.avg_num_nodes

        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.atomic_energies_map)

        equiformer_kargs = dict(
            avg_num_neighbors=avg_num_neighbors,
            num_layers=self.config.num_layers,
            lmax=self.config.lmax,
            mmax=self.config.mmax,
            sphere_channels=self.config.sphere_channels,
            num_edge_channels=self.config.num_edge_channels,
            atom_edge_embedding=self.config.atom_edge_embedding,
            num_rbf=self.config.num_rbf,
            attn_hidden_channels=self.config.attn_hidden_channels,
            num_heads=self.config.num_heads,
            attn_alpha_channels=self.config.attn_alpha_channels,
            attn_value_channels=self.config.attn_value_channels,
            ffn_hidden_channels=self.config.ffn_hidden_channels,
            norm_type=self.config.norm_type,
            grid_resolution=self.config.grid_resolution,
            use_m_share_rad=self.config.use_m_share_rad,
            use_attn_renorm=self.config.use_attn_renorm,
            attn_act_type=self.config.attn_act_type,
            ff_type=self.config.ff_type,
            alpha_drop=self.config.alpha_drop,
            drop_path_rate=self.config.drop_path_rate,
            avg_num_nodes=avg_num_nodes,
            cutoff=r_max,
            num_species=num_species,
            direct_force=self.config.direct_force,
            deterministic=not training,
        )

        equiformer_model = EquiformerV2Block(**equiformer_kargs)
        backbone_outputs = equiformer_model(
            edge_vectors, node_species, senders, receivers, n_node
        )

        if self.config.direct_force:
            node_energies, forces = backbone_outputs
        else:
            node_energies = backbone_outputs

        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies

        atomic_energies_ = get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species
        )
        atomic_energies_ = jnp.asarray(atomic_energies_)
        node_energies += atomic_energies_[node_species]  # [n_nodes, ]

        if self.config.direct_force:
            return jnp.concat([node_energies[:, None], std * forces], axis=-1)
        return node_energies


class EquiformerV2Block(nn.Module):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon
    S2 activation.
    """

    avg_num_neighbors: float
    num_layers: int
    lmax: int
    mmax: int
    sphere_channels: int
    num_species: int
    num_edge_channels: int
    atom_edge_embedding: str
    attn_hidden_channels: int
    num_heads: int
    attn_alpha_channels: int
    attn_value_channels: int
    ffn_hidden_channels: int
    norm_type: LayerNormType
    grid_resolution: int
    use_m_share_rad: bool
    use_attn_renorm: bool
    attn_act_type: AttnActType
    ff_type: FeedForwardType
    alpha_drop: float
    drop_path_rate: float
    avg_num_nodes: float
    num_rbf: int = 600
    cutoff: float = 5.0
    direct_force: bool = False
    deterministic: bool = True

    def setup(self):
        # Weights for message initialization
        self.sphere_embedding = nn.Embed(self.num_species, self.sphere_channels)

        # Function used to measure the distances between atoms
        self.distance_expansion = GaussianBasis(
            self.cutoff,
            self.num_rbf,
            trainable=False,
            rbf_width=2.0,
        )

        # Sizes of radial functions (2 hidden channels, input and output are ignored)
        edge_channels_list = [self.num_edge_channels] * 2

        # Atom edge embedding
        self.edge_embedding = None
        if self.atom_edge_embedding == 'shared':
            self.edge_embedding = nn.Embed(self.num_species, 2 * self.num_edge_channels)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.lmax,
            self.mmax,
            self.sphere_channels,
            edge_channels_list,
            self.atom_edge_embedding == 'isolated',
            num_species=self.num_species,
            rescale_factor=self.avg_num_neighbors,
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.layers = [
            EquiformerV2Layer(
                self.lmax,
                self.mmax,
                self.grid_resolution,
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.num_species,
                edge_channels_list,
                self.atom_edge_embedding == 'isolated',
                self.use_m_share_rad,
                self.use_attn_renorm,
                self.attn_act_type,
                self.ff_type,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.deterministic,
            )
            for _ in range(self.num_layers)
        ]

        # Output blocks for energy and forces
        self.norm = parse_layernorm(self.norm_type, self.lmax)
        self.energy_block = FeedForwardNetwork(
            self.lmax,
            self.ffn_hidden_channels,
            1,
            self.grid_resolution,
            self.ff_type,
        )
        if self.direct_force:
            self.force_block = SO2EquivariantGraphAttention(
                self.lmax,
                self.mmax,
                self.grid_resolution,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.num_species,
                edge_channels_list,
                self.atom_edge_embedding == 'isolated',
                self.use_m_share_rad,
                self.use_attn_renorm,
                self.attn_act_type,
                alpha_drop=0.0,
                deterministic=self.deterministic,
            )

    def __call__(
        self,
        edge_vectors: jax.Array,  # [n_edges, 3]
        node_species: jax.Array,  # [n_nodes] int between 0 and num_species-1
        senders: jax.Array,  # [n_edges]
        receivers: jax.Array,  # [n_edges]
        n_node: jax.Array,  # [batch_size]
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        num_atoms = len(node_species)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        if not self.deterministic:
            rng = self.make_rng('rotation')
            rot_gamma = jax.random.uniform(
                rng, shape=len(edge_vectors), maxval=2 * jnp.pi,
                dtype=edge_vectors.dtype
            )
        else:
            rot_gamma = jnp.zeros(len(edge_vectors), dtype=edge_vectors.dtype)

        mapping_coeffs = get_mapping_coeffs(self.lmax, self.mmax)
        wigner_mats = get_wigner_mats(
            self.lmax, self.mmax, edge_vectors, rot_gamma, mapping_coeffs.perm
        )

        # Initialize the l = 0, m = 0 coefficients
        node_feats_0 = self.sphere_embedding(node_species)[:, None]
        node_feats_m_pad = jnp.zeros(
            [num_atoms, (self.lmax + 1) ** 2 - 1, node_feats_0.shape[-1]],
            dtype=edge_vectors.dtype,
        )
        node_feats = jnp.concat((node_feats_0, node_feats_m_pad), axis=1)

        # Edge encoding (distance and atom edge)
        edge_distances = safe_norm(edge_vectors, axis=-1)
        edge_embeds = self.distance_expansion(edge_distances)
        if self.edge_embedding is not None:
            node_embeds = self.edge_embedding(node_species)
            senders_embeds, receivers_embeds = jnp.split(node_embeds, 2, axis=-1)
            edge_embeds = jnp.concat(
                (edge_embeds, senders_embeds[senders], receivers_embeds[receivers]), axis=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            node_species, edge_embeds, senders, receivers, wigner_mats
        )
        node_feats = node_feats + edge_degree

        for layer in self.layers:
            node_feats = layer(
                node_feats,
                node_species,
                edge_embeds,
                senders,
                receivers,
                wigner_mats,
                n_node=n_node,  # for GraphDropPath
            )

        # Final layer norm
        node_feats = self.norm(node_feats)

        node_energies = self.energy_block(node_feats)
        node_energies = node_energies[:, 0, 0] / self.avg_num_nodes

        if self.direct_force:
            forces = self.force_block(
                node_feats,
                node_species,
                edge_embeds,
                senders,
                receivers,
                wigner_mats,
            )
            return node_energies, forces[:, 1:4, 0]

        return node_energies


class EquiformerV2Layer(nn.Module):
    lmax: int
    mmax: int
    resolution: int
    sphere_channels: int
    attn_hidden_channels: int
    num_heads: int
    attn_alpha_channels: int
    attn_value_channels: int
    ffn_hidden_channels: int
    output_channels: int
    num_species: int
    edge_channels_list: tuple[int, ...]
    use_atom_edge_embedding: bool = True
    use_m_share_rad: bool = False
    use_attn_renorm: bool = True
    attn_act_type: AttnActType = AttnActType.S2_SEP
    ff_type: FeedForwardType = FeedForwardType.GRID_SEP
    norm_type: LayerNormType = LayerNormType.RMS_NORM_SH
    alpha_drop: float = 0.0
    drop_path_rate: float = 0.0
    deterministic: bool = False # Randomness of rotation matrix

    def setup(self):
        self.norm_1 = parse_layernorm(self.norm_type, self.lmax)

        self.graph_attn = SO2EquivariantGraphAttention(
            self.lmax,
            self.mmax,
            self.resolution,
            hidden_channels=self.attn_hidden_channels,
            num_heads=self.num_heads,
            attn_alpha_channels=self.attn_alpha_channels,
            attn_value_channels=self.attn_value_channels,
            output_channels=self.sphere_channels,
            num_species=self.num_species,
            edge_channels_list=self.edge_channels_list,
            use_atom_edge_embedding=self.use_atom_edge_embedding,
            use_m_share_rad=self.use_m_share_rad,
            use_attn_renorm=self.use_attn_renorm,
            attn_act_type=self.attn_act_type,
            alpha_drop=self.alpha_drop,
            deterministic=self.deterministic,
        )

        self.drop_path = GraphDropPath(
            self.drop_path_rate, self.deterministic
        ) if self.drop_path_rate > 0.0 else None

        self.norm_2 = parse_layernorm(self.norm_type, self.lmax)

        self.ffn = FeedForwardNetwork(
            self.lmax,
            hidden_channels=self.ffn_hidden_channels,
            output_channels=self.output_channels,
            resolution=self.resolution,
            ff_type=self.ff_type,
        )

        self.ffn_shortcut = None
        if self.sphere_channels != self.output_channels:
            self.ffn_shortcut = SO3Linear(self.lmax, self.output_channels)

    def __call__(
        self,
        node_feats: jax.Array,
        node_species: jax.Array,
        edge_embeds: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        wigner_mats: WignerMats,
        n_node: jax.Array | None = None,
    ) -> jax.Array:
        # Attention block
        node_feats_res = node_feats
        node_feats = self.norm_1(node_feats)
        node_feats = self.graph_attn(
            node_feats, node_species, edge_embeds, senders, receivers, wigner_mats
        )

        if self.drop_path is not None:
            node_feats = self.drop_path(node_feats, n_node)

        node_feats = node_feats + node_feats_res

        # FFN block
        node_feats_res = node_feats
        node_feats = self.norm_2(node_feats)
        node_feats = self.ffn(node_feats)

        if self.drop_path is not None:
            node_feats = self.drop_path(node_feats, n_node)

        if self.ffn_shortcut is not None:
            node_feats_res = self.ffn_shortcut(node_feats_res)

        node_feats = node_feats + node_feats_res

        return node_feats
