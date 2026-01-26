# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Thorben Frank (https://github.com/thorben-frank/mlff) under MIT license.

import flax.linen as nn
from flax.linen.initializers import constant
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn

from mlip.models.options import parse_activation


class MLP(nn.Module):
    features: tuple[int, ...]
    activation: str
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, use_bias=self.use_bias)(x)
            if i != len(self.features) - 1:
                x = parse_activation(self.activation)(x)
        return x


class ResidualMLP(nn.Module):
    num_blocks: int = 3
    activation: str = 'silu'
    # In original So3krates, this is set to False. But using bias is better.
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jax.Array):
        x = inputs
        feat = x.shape[-1]
        for _ in range(self.num_blocks):
            x = parse_activation(self.activation)(x)
            x = nn.Dense(feat, use_bias=self.use_bias)(x)
        x = x + inputs
        # In original So3krates, there exists a non-residual Linear. But it would
        # be slightly better to include it in the residual.
        # x = parse_activation(self.activation)(x)
        # x = nn.Dense(feat, use_bias=self.use_bias)(x)
        return x


class InteractionBlock(nn.Module):
    @nn.compact
    def __call__(
        self,
        node_feats: jax.Array,
        chi: e3nn.IrrepsArray,
    ) -> tuple[jax.Array, e3nn.IrrepsArray]:
        num_features = node_feats.shape[-1]

        # Tensor product using CG coefficents has been removed for simplicity.
        chi_scalar = e3nn.norm(chi, squared=True, per_irrep=True).array

        feats = jnp.concatenate([node_feats, chi_scalar], axis=-1)
        feats = nn.Dense(num_features + chi.irreps.num_irreps)(feats)

        # node_feats: [n_nodes, num_features], chi_coeffs: [n_nodes, n_heads]
        node_feats, chi_coeffs = jnp.split(feats, [num_features], axis=-1)

        return node_feats, chi_coeffs * chi


class FeatureBlock(nn.Module):
    num_heads: int
    rad_features: tuple[int, ...]
    sph_features: tuple[int, ...]
    activation: str
    avg_num_neighbors: float

    @nn.compact
    def __call__(
        self,
        node_feats: jax.Array,
        edge_feats: jax.Array,
        chi_scalar: jax.Array,
        cutoffs: jax.Array,
        senders: jax.Array,
        receivers: jax.Array
    ) -> jax.Array:
        alpha = FilterScaledAttentionMap(
            num_heads=self.num_heads,
            rad_features=self.rad_features,
            sph_features=self.sph_features,
            activation=self.activation
        )(node_feats, edge_feats, chi_scalar, senders, receivers)

        alpha = alpha * cutoffs[:, None] # [n_edges, n_heads]

        head_dim = node_feats.shape[-1] // self.num_heads
        v_j = nn.Dense(node_feats.shape[-1], use_bias=False)(node_feats)[senders]
        v_j = v_j.reshape(-1, self.num_heads, head_dim)

        node_feats = jax.ops.segment_sum(
            alpha[..., None] * v_j, receivers, num_segments=node_feats.shape[0]
        ) / self.avg_num_neighbors
        node_feats = node_feats.reshape(-1, head_dim * self.num_heads)
        return node_feats


class GeometricBlock(nn.Module):
    rad_features: tuple[int, ...]
    sph_features: tuple[int, ...]
    activation: str
    avg_num_neighbors: float

    @nn.compact
    def __call__(
        self,
        edge_sh: e3nn.IrrepsArray,
        node_feats: jax.Array,
        edge_feats: jax.Array,
        chi_scalar: jax.Array,
        cutoffs: jax.Array,
        senders: jax.Array,
        receivers: jax.Array
    ) -> e3nn.IrrepsArray:
        alpha = FilterScaledAttentionMap(
            num_heads=edge_sh.irreps.num_irreps,
            rad_features=self.rad_features,
            sph_features=self.sph_features,
            activation=self.activation
        )(node_feats, edge_feats, chi_scalar, senders, receivers)

        alpha = alpha * cutoffs[:, None]

        # e3nn supports directly multiply IrrepsArray with scalars.
        chi = e3nn.scatter_sum(alpha * edge_sh, dst=receivers, output_size=node_feats.shape[0])
        return chi / self.avg_num_neighbors


class FilterScaledAttentionMap(nn.Module):
    num_heads: int
    rad_features: tuple[int, ...]
    sph_features: tuple[int, ...]
    activation: str

    @nn.compact
    def __call__(
        self,
        node_feats: jax.Array,
        edge_feats: jax.Array,
        chi_scalar: jax.Array,
        senders: jax.Array,
        receivers: jax.Array
    ) -> jax.Array:
        head_dim = node_feats.shape[-1] // self.num_heads

        # Radial spherical filter
        w_ij = MLP(self.rad_features, self.activation)(edge_feats)
        w_ij += MLP(self.sph_features, self.activation)(chi_scalar)
        w_ij = w_ij.reshape(-1, self.num_heads, head_dim)

        # Geometric attention coefficients
        q_i = nn.Dense(node_feats.shape[-1], use_bias=False)(node_feats)
        q_i = q_i.reshape(-1, self.num_heads, head_dim)[receivers]

        k_j = nn.Dense(node_feats.shape[-1], use_bias=False)(node_feats)
        k_j = k_j.reshape(-1, self.num_heads, head_dim)[senders]

        return (q_i * w_ij * k_j).sum(axis=-1) / jnp.sqrt(head_dim) # [n_edges, n_heads]


class ZBLRepulsion(nn.Module):
    """Ziegler-Biersack-Littmark repulsion."""
    index_to_z: tuple[int, ...]
    a0: float = 0.5291772105638411
    ke: float = 14.399645351950548

    @nn.compact
    def __call__(
        self,
        node_species: jax.Array,
        distances: jax.Array,
        cutoffs: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> jax.Array:
        def softplus_inverse(x):
            return x + jnp.log(-jnp.expm1(-x))

        # We vectorize a/c for simplicity.
        a_init = softplus_inverse(jnp.array([3.20000, 0.94230, 0.40280, 0.20160]))
        c_init = softplus_inverse(jnp.array([0.18180, 0.50990, 0.28020, 0.02817]))

        a = nn.softplus(self.param('a', constant(a_init), (4,)))
        c = nn.softplus(self.param('c', constant(c_init), (4,)))
        c = c / jnp.sum(c)

        p = nn.softplus(self.param('p', constant(softplus_inverse(0.23)), (1,)))
        d = nn.softplus(self.param(
            'd', constant(softplus_inverse(1 / (0.8854 * self.a0))), (1,)
        ))

        z = jnp.array(self.index_to_z)[node_species]
        z_i = z[receivers]
        z_j = z[senders]

        x = self.ke * cutoffs * z_i * z_j / (distances + 1e-8)

        rzd = distances * (jnp.power(z_i, p) + jnp.power(z_j, p)) * d

        # ZBL screening function, shape: [n_edges]
        y = jnp.sum(c * jnp.exp(-a * rzd[:, None]), axis=-1)

        scaled_d = distances / 1.5
        sigma_d = jnp.exp(-1. / (jnp.where(scaled_d > 1e-8, scaled_d, 1e-8)))
        sigma_1_d = jnp.exp(-1. / (jnp.where(1 - scaled_d > 1e-8, 1 - scaled_d, 1e-8)))
        w = sigma_1_d / (sigma_1_d + sigma_d)

        energy_rep = w * x * y / 2
        return jax.ops.segment_sum(energy_rep, receivers, num_segments=node_species.shape[0])
