# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Yi-Lun Liao (https://github.com/atomicarchitects/equiformer_v2) under MIT license.

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from flax.typing import PRNGKey

from mlip.models.options import parse_activation
from mlip.models.equiformer_v2.utils import get_expand_index, get_mapping_coeffs
from mlip.models.equiformer_v2.transform import WignerMats


class MLP(nn.Module):
    """MLP with layer norm."""

    features: tuple[int, ...]
    activation: str = 'silu'
    use_bias: bool = True
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, use_bias=self.use_bias)(x)
            if i != len(self.features) - 1:
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = parse_activation(self.activation)(x)
        return x


class SO3Linear(nn.Module):
    """EquiformerV2 linear layer."""

    lmax: int
    features: int

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        expand_index = get_expand_index(self.lmax)

        kernel = self.param(
            'kernel',
            initializers.lecun_normal(),
            ((self.lmax + 1), inputs.shape[-1], self.features),
        )
        bias = self.param('bias', initializers.zeros_init(), (self.features,))

        kernel_expanded = kernel[expand_index] # [(L_max + 1) ** 2, C_in, C_out]
        out = jnp.einsum(
            "bmi, mio -> bmo", inputs, kernel_expanded
        )  # [N, (L_max + 1) ** 2, C_out]
        out = out.at[:, 0:1, :].add(bias.reshape(1, 1, self.features))

        return out


class SO2mConvolution(nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int): Order of the spherical harmonic coefficients
        channels (int): Number of output channels used during the SO(2) conv
        lmax (int): Maximum degree of the spherical harmonics
    """

    m: int
    channels: int
    lmax: int

    @nn.compact
    def __call__(self, feats_m: jax.Array) -> tuple[jax.Array, jax.Array]:
        num_edges = len(feats_m)

        out_channels = 2 * (self.lmax - self.m + 1) * self.channels

        feats_m = nn.Dense(out_channels, use_bias=False)(feats_m)
        feats_r, feats_i = jnp.split(feats_m, 2, axis=2)
        feats_m_r = feats_r[:, 0] - feats_i[:, 1]
        feats_m_i = feats_r[:, 1] + feats_i[:, 0]

        return (
            feats_m_r.reshape(num_edges, -1, self.channels),
            feats_m_i.reshape(num_edges, -1, self.channels),
        )


class SO2Convolution(nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        lmax (int): Maximum degree of the spherical harmonics
        mmax (int): Maximum order of the spherical harmonics
        output_channels (int): Number of output channels used during the SO(2) conv
        internal_weights (bool): If True, not using radial function to multiply inputs features
        edge_channels_list (list:int): List of sizes of invariant edge embedding. For example,
            [hidden_channels, hidden_channels].
        extra_scalar_channels (int): If not None, return `out` and `extra_scalar_features`.
    """
    lmax: int
    mmax: int
    output_channels: int
    internal_weights: bool = True
    edge_channels_list: tuple[int, ...] | None = None
    extra_scalar_channels: int | None = None

    @nn.compact
    def __call__(
        self,
        edge_feats: jax.Array, # in m primary order
        edge_embeds: jax.Array | None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        num_edges = len(edge_embeds)

        m_size = get_mapping_coeffs(self.lmax, self.mmax).m_size

        # radial function
        if not self.internal_weights:
            assert self.edge_channels_list is not None, "`edge_channels_list` must be provided."
            assert edge_embeds is not None, "`edge_embeds` must be provided."
            edge_embeds = MLP(
                self.edge_channels_list + (edge_feats.shape[-1] * sum(m_size),)
            )(edge_embeds)

        m0_out_channels = (self.lmax + 1) * self.output_channels
        if self.extra_scalar_channels is not None:
            m0_out_channels += self.extra_scalar_channels

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        # `feats` means `egde_feats` in the following code for simplicity
        feats_0 = edge_feats[:, :m_size[0]].reshape(num_edges, -1)
        if not self.internal_weights:
            feats_0 *= edge_embeds[:, :feats_0.shape[-1]]
            offset_rad = feats_0.shape[-1]

        feats_0 = nn.Dense(m0_out_channels)(feats_0)

        if self.extra_scalar_channels is not None:
            feats_extra, feats_0 = jnp.split(
                feats_0, [self.extra_scalar_channels], axis=-1
            )

        # x[:, 0 : self.mappingReduced.m_size[0]] = feats_0
        feats_out = [feats_0.reshape(num_edges, -1, self.output_channels)]

        # Compute the values for the m > 0 coefficients
        offset = m_size[0]
        for m in range(1, self.mmax + 1):
            # Get the m order coefficients, shape: [N, 2, m_size[m] * sphere_channels]
            feats_m = edge_feats[:, offset : 2*m_size[m]+offset].reshape(num_edges, 2, -1)
            offset += 2 * m_size[m]

            if not self.internal_weights:
                feats_m *= edge_embeds[:, None, offset_rad : feats_m.shape[-1] + offset_rad]
                offset_rad += feats_m.shape[-1]

            # x[:, offset : offset + 2 * self.mappingReduced.m_size[m]] = feats_m
            feats_out.extend(SO2mConvolution(m, self.output_channels, self.lmax)(feats_m))

        edge_feats = jnp.concat(feats_out, axis=1)

        if self.extra_scalar_channels is not None:
            return edge_feats, feats_extra
        return edge_feats


class EdgeDegreeEmbedding(nn.Module):
    """

    Args:
        lmax (int): Maximum degree of the spherical harmonics
        mmax (int): Maximum order of the spherical harmonics
        sphere_channels (int): Number of spherical channels
        edge_channels_list (list:int): List of sizes of invariant edge embedding. For example, 
            [hidden_channels, hidden_channels]. The last one will be used as hidden size when
            `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance
            for edge scalar features
        num_species (int): Maximum number of atomic numbers
        rescale_factor (float): Rescale the sum aggregation
    """
    lmax: int
    mmax: int
    sphere_channels: int
    edge_channels_list: tuple[int, ...]
    use_atom_edge_embedding: bool = False
    num_species: int | None = None
    rescale_factor: float = 5.0

    @nn.compact
    def __call__(
        self,
        node_species: jax.Array,
        edge_embeds: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        wigner_mats: WignerMats,
    ) -> jax.Array:
        num_nodes = node_species.shape[0]

        mapping_coeffs = get_mapping_coeffs(self.lmax, self.mmax)
        m_size_0 = mapping_coeffs.m_size[0]
        m_size_pad = mapping_coeffs.num_coefficients - m_size_0

        if self.use_atom_edge_embedding:
            assert self.num_species is not None, "num_species must be provided"
            # I have changed the layer order and merged the two embedding layers into one.
            node_embeds = nn.Embed(
                self.num_species, 2 * self.edge_channels_list[-1],
                embedding_init=initializers.normal(stddev=0.001), # Why?
            )(node_species)
            senders_embeds, receivers_embeds = jnp.split(node_embeds, 2, axis=-1)
            edge_embeds = jnp.concat(
                (edge_embeds, senders_embeds[senders], receivers_embeds[receivers]), axis=-1
            )

        feats_m_0 = MLP(
            self.edge_channels_list + (m_size_0 * self.sphere_channels,)
        )(edge_embeds).reshape(-1, m_size_0, self.sphere_channels)

        feats_m_pad = jnp.zeros(
            (edge_embeds.shape[0], m_size_pad, self.sphere_channels), dtype=feats_m_0.dtype
        )
        # edge_feats: [n_edges, (lmax + 1) ^ 2, num_channels], m primary
        edge_feats = jnp.concat((feats_m_0, feats_m_pad), axis=1)

        edge_feats = wigner_mats.rotate_inv(edge_feats)
        # NOTE: In eSEN, there is a edge_envelope, however, it seems EquiFormerV2 does not use a
        # edge_envelope at all.

        # Compute the sum of the incoming neighboring messages for each target node
        node_feats = jax.ops.segment_sum(edge_feats, receivers, num_nodes) / self.rescale_factor

        return node_feats


def _drop_path(inputs: jax.Array, rate: float, rng: PRNGKey) -> jax.Array:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    keep_prob = 1 - rate
    # work with diff dim tensors, not just 2D ConvNets
    shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
    random_tensor = keep_prob + jax.random.uniform(rng, shape, dtype=inputs.dtype)
    random_tensor = jnp.floor(random_tensor)  # binarize
    output = (inputs / keep_prob) * random_tensor
    return output


class GraphDropPath(nn.Module):
    """Consider batch for graph inputs when dropping paths."""

    rate: float
    deterministic: bool = True

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        n_node: jax.Array,
    ) -> jax.Array:
        if (self.rate == 0.0) or self.deterministic:
            return inputs

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.rate == 1.0:
            return jnp.zeros_like(inputs)

        rng = self.make_rng('dropout')

        batch_size = len(n_node)
        # work with diff dim tensors, not just 2D ConvNets
        shape = (batch_size,) + (1,) * (inputs.ndim - 1)
        ones = jnp.ones(shape, dtype=inputs.dtype)
        drop = _drop_path(ones, self.rate, rng)

        # create pyg batch from n_node
        output_size = n_node.shape[0]
        num_elements = inputs.shape[0]
        batch = jnp.repeat(jnp.arange(output_size), n_node, total_repeat_length=num_elements)

        out = inputs * drop[batch]
        return out
