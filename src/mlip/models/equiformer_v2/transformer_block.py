# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Yi-Lun Liao (https://github.com/atomicarchitects/equiformer_v2) under MIT license.

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers

from mlip.models.equiformer_v2.transform import (
    WignerMats,
    get_s2grid_mats,
)
from mlip.models.equiformer_v2.activations import (
    SmoothLeakyReLU,
    GateActivation,
    S2Activation,
    SeparableS2Activation,
)
from mlip.models.equiformer_v2.blocks import SO3Linear, MLP, SO2Convolution
from mlip.models.equiformer_v2.utils import (
    AttnActType,
    FeedForwardType,
    get_expand_index,
    pyg_softmax,
)


class SO2EquivariantGraphAttention(nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention
        weights and non-linear messages attention weights * non-linear messages -> Linear

    Args:
        lmax (int): Maximum degree of spherical harmonics
        mmax (int): Maximum degree of spherical harmonics
        resolution (int): Resolution of the spherical grid
        hidden_channels (int): Number of hidden channels used during the SO(2) conv
        num_heads (int): Number of attention heads
        attn_alpha_channels (int): Number of channels for alpha vector in each attention head
        attn_value_channels (int): Number of channels for value vector in each attention head
        output_channels (int): Number of output channels
        num_species (int): Maximum number of atomic numbers
        edge_channels_list (list:int): List of sizes of invariant edge embedding. For example, 
            [input_channels, hidden_channels, hidden_channels]. The last one will be used as hidden
            size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative
            distance for edge scalar features
        use_m_share_rad (bool): Whether all m components within a type-L vector of one channel
            share radial function weights
        use_attn_renorm (bool): Whether to re-normalize attention weights
        attn_act_type (AttnActType): Type of attention activation function
        alpha_drop (float): Dropout rate for attention weights
    """

    lmax: int
    mmax: int
    resolution: int
    hidden_channels: int
    num_heads: int
    attn_alpha_channels: int
    attn_value_channels: int
    output_channels: int
    num_species: int
    edge_channels_list: tuple[int, ...]
    use_atom_edge_embedding: bool = True
    use_m_share_rad: bool = False
    use_attn_renorm: bool = True
    attn_act_type: AttnActType = AttnActType.S2_SEP
    alpha_drop: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(
        self,
        node_feats: jax.Array,
        node_species: jax.Array,
        edge_embeds: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        wigner_mats: WignerMats,
    ) -> jax.Array:
        num_nodes = node_feats.shape[0]
        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance expansion as inputs
        if self.use_atom_edge_embedding:
            assert self.num_species is not None, "num_species must be provided"
            node_embeds = nn.Embed(
                self.num_species, 2 * self.edge_channels_list[-1],
                embedding_init=initializers.normal(stddev=0.001), # Why?
            )(node_species)
            senders_embeds, receivers_embeds = jnp.split(node_embeds, 2, axis=-1)
            edge_embeds = jnp.concat(
                (edge_embeds, senders_embeds[senders], receivers_embeds[receivers]), axis=-1
            )

        messages = jnp.concat((node_feats[senders], node_feats[receivers]), axis=2)

        # radial function (scale all m components within a type-L vector of one channel
        # with the same weight)
        if self.use_m_share_rad:
            edge_embeds_weight = MLP(
                self.edge_channels_list + (messages.shape[-1] * (self.lmax + 1),)
            )(edge_embeds).reshape(-1, (self.lmax + 1), messages.shape[-1])
            # [E, (L_max + 1) ** 2, C]
            messages *= edge_embeds_weight[:, get_expand_index(self.lmax)]

        # Rotate the irreps to align with the edge, get m primary
        messages = wigner_mats.rotate(messages)

        # First SO(2)-convolution
        alpha_channels = self.num_heads * self.attn_alpha_channels
        if self.attn_act_type == AttnActType.GATE:
            extra_scalar_channels = alpha_channels + self.lmax * self.hidden_channels
        elif self.attn_act_type == AttnActType.S2_SEP:
            extra_scalar_channels = alpha_channels + self.hidden_channels
        else:
            extra_scalar_channels = alpha_channels

        messages, scalar_extra = SO2Convolution(
            self.lmax,
            self.mmax,
            self.hidden_channels,
            internal_weights=self.use_m_share_rad,
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad else None
            ),
            # for attention weights and/or gate activation
            extra_scalar_channels=extra_scalar_channels
        )(messages, edge_embeds)

        # Activation
        if self.attn_act_type == AttnActType.GATE:
            # Gate activation
            scalar_alpha, scalar_gating = jnp.split(scalar_extra, [alpha_channels], axis=-1)
            messages = GateActivation(
                self.lmax, self.mmax, self.hidden_channels, m_prime=True
            )(scalar_gating, messages)

        elif self.attn_act_type == AttnActType.S2_SEP:
            scalar_alpha, scalar_gating = jnp.split(scalar_extra, [alpha_channels], axis=-1)
            messages = SeparableS2Activation(
                self.lmax, self.mmax, self.resolution, m_prime=True
            )(scalar_gating, messages)

        else:
            scalar_alpha = scalar_extra
            messages = S2Activation(
                self.lmax, self.mmax, self.resolution, m_prime=True
            )(messages)
            # x_message._grid_act(self.so3_grid, self.value_act, self.mappingReduced)

        # Second SO(2)-convolution
        messages = SO2Convolution(
            self.lmax, self.mmax, self.num_heads * self.attn_value_channels
        )(messages, edge_embeds)

        # Attention weights
        scalar_alpha = scalar_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
        if self.use_attn_renorm:
            scalar_alpha = nn.LayerNorm()(scalar_alpha)
        scalar_alpha = SmoothLeakyReLU()(scalar_alpha)

        # torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        alpha = jnp.einsum("bik, ki -> bi", scalar_alpha, self.param(
            'alpha_dot',
            initializers.lecun_normal(),
            (self.attn_alpha_channels, self.num_heads),
        ))
        alpha = pyg_softmax(alpha, receivers, num_nodes)
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)

        if self.alpha_drop != 0.0:
            alpha = nn.Dropout(self.alpha_drop, deterministic=self.deterministic)(alpha)

        # Attention weights * non-linear messages
        attn = messages.reshape(
            messages.shape[0],
            messages.shape[1],
            self.num_heads,
            self.attn_value_channels,
        )
        attn = attn * alpha
        messages = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads * self.attn_value_channels,
        )

        # Rotate back the irreps
        messages = wigner_mats.rotate_inv(messages)

        # Compute the sum of the incoming neighboring messages for each target node
        node_feats = jax.ops.segment_sum(messages, receivers, num_nodes)

        node_feats = SO3Linear(self.lmax, self.output_channels)(node_feats)
        return node_feats


class FeedForwardNetwork(nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        lmax (int): Degree (l)
        hidden_channels (int): Number of hidden channels used during feedforward network
        output_channels (int): Number of output channels
        resolution (int): Resolution of the S2 grid
        ff_type (FeedForwardType): Type of feedforward network
    """

    lmax: int
    hidden_channels: int
    output_channels: int
    resolution: int
    ff_type: FeedForwardType

    @nn.compact
    def __call__(self, node_feats: jax.Array) -> jax.Array:
        node_feats_orig = node_feats
        node_feats = SO3Linear(self.lmax, self.hidden_channels)(node_feats)

        if self.ff_type in [FeedForwardType.GRID, FeedForwardType.GRID_SEP]:
            so3_grid = get_s2grid_mats(self.lmax, self.lmax)

            node_feats_grid = so3_grid.to_grid(node_feats)
            node_feats_grid = MLP(
                [self.hidden_channels] * 3, use_bias=False, use_layer_norm=False,
            )(node_feats_grid)
            node_feats = so3_grid.from_grid(node_feats_grid)

            if self.ff_type == FeedForwardType.GRID_SEP:
                gating_scalars = nn.silu(
                    nn.Dense(self.hidden_channels)(node_feats_orig[:, 0:1])
                )
                node_feats = jnp.concat(
                    (gating_scalars, node_feats[:, 1:]), axis=1
                )

        elif self.ff_type == FeedForwardType.GATE:
            gating_scalars = nn.Dense(self.lmax * self.hidden_channels)(node_feats_orig[:, 0:1])
            node_feats = GateActivation(
                self.lmax, self.lmax, self.hidden_channels
            )(gating_scalars, node_feats)

        elif self.ff_type == FeedForwardType.S2_SEP:
            gating_scalars = nn.Dense(self.hidden_channels)(node_feats_orig[:, 0:1])
            node_feats = SeparableS2Activation(
                self.lmax, self.lmax, self.resolution
            )(gating_scalars, node_feats)

        else:
            node_feats = S2Activation(self.lmax, self.lmax, self.resolution)(node_feats)

        node_feats = SO3Linear(self.lmax, self.output_channels)(node_feats)
        return node_feats
