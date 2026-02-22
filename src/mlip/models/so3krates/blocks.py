# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Thorben Frank (https://github.com/thorben-frank/mlff) under MIT license.

import flax.linen as nn
from flax.linen.initializers import constant
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import numpy as np

from mlip.models.options import parse_activation


def _check_irreps_aligned(irreps: e3nn.Irreps) -> tuple[int, e3nn.Irreps]:
    mul_in = set(m for m, _ in irreps.regroup())
    assert len(mul_in) == 1, "Input irreps must have the same multiplicity."

    mul_in, = mul_in
    irreps_out = irreps.regroup() // mul_in
    assert e3nn.Irreps([i for _ in range(mul_in) for i in irreps_out]) == irreps, (
        "Input irreps must be in order `1e+2e+...+1e+2e+...`"
    )

    return mul_in, irreps_out


class AlignedLinear(nn.Module):
    r"""Aligned equivariant Linear. A fast version of e3nn.flax.Linear.

    - input irreps = $mul \times (l, p)$
    - output irreps = $mul_out \times (l, p)$
    """
    mul_out: int
    split: int = 1

    @nn.compact
    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        mul_in, irreps = _check_irreps_aligned(x.irreps)

        gradient_normalization = e3nn.config("gradient_normalization")
        if gradient_normalization == "element":
            alpha = 1.0
            initializer = nn.initializers.lecun_normal()
        elif gradient_normalization == "path":
            alpha = np.sqrt(1.0 / mul_in)
            initializer = nn.initializers.normal(stddev=1.0)
        else:
            raise ValueError(f"Unknown gradient_normalization: {gradient_normalization}")

        kernel = self.param(
            "kernel", initializer, (len(irreps), mul_in, self.mul_out)
        )
        kernel = jnp.repeat(kernel, np.array([ir.dim for _, ir in irreps]), axis=0)

        x_arr = x.array.reshape(*x.shape[:-1], mul_in, -1)

        y = alpha * jnp.einsum(
            "...im, mio -> ...om", x_arr, kernel
        ).reshape(*x.shape[:-1], -1)

        if self.split == 1:
            return e3nn.IrrepsArray([i for _ in range(self.mul_out) for i in irreps], y)

        assert self.split > 1, "split must be positive."
        assert self.mul_out % self.split == 0, "mul_out must be divisible by split."

        return (
            e3nn.IrrepsArray([i for _ in range(self.mul_out // self.split) for i in irreps], y_i)
            for y_i in jnp.split(y, self.split, axis=-1)
        )


def aligned_norm(x: e3nn.IrrepsArray) -> jax.Array:
    r"""e3nn.norm for aligned irreps."""
    mul_in, irreps = _check_irreps_aligned(x.irreps)

    x_norm = jnp.square(x.array).reshape(*x.shape[:-1], mul_in, -1)

    offset = 0
    outs = []
    for _, ir in irreps:
        outs.append(
            x_norm[..., offset:offset+ir.dim].sum(axis=-1)
        )
        offset += ir.dim
    x_scalar = jnp.stack(outs, axis=-1)

    return x_scalar.reshape(*x.shape[:-1], -1)


def aligned_dot(x: e3nn.IrrepsArray, y: e3nn.IrrepsArray) -> jax.Array:
    r"""e3nn.dot for aligned irreps."""
    mul_in, irreps = _check_irreps_aligned(x.irreps)
    mul_in_y, irreps_y = _check_irreps_aligned(y.irreps)

    assert mul_in_y == mul_in, "Input irreps must have the same multiplicity."
    assert irreps == irreps_y, "Input irreps must be the same."

    x_norm = (x.array * y.array).reshape(*x.shape[:-1], mul_in, -1)

    offset = 0
    outs = []
    for _, ir in irreps:
        outs.append(
            x_norm[..., offset:offset+ir.dim].sum(axis=-1)
        )
        offset += ir.dim
    x_scalar = jnp.stack(outs, axis=-1)

    return x_scalar.reshape(*x.shape[:-1], -1)


def aligned_mul(x: e3nn.IrrepsArray, y: jax.Array) -> e3nn.IrrepsArray:
    r"""e3nn.IrrepsArray.__mul__ for aligned irreps."""
    mul_in, irreps = _check_irreps_aligned(x.irreps)
    assert x.irreps.num_irreps == y.shape[-1], "Input irreps and array must have the same shape."

    y_arr = y.reshape(*y.shape[:-1], mul_in, -1)
    y_arr = jnp.repeat(y_arr, np.array([ir.dim for _, ir in irreps]), axis=-1)

    x_arr = x.array.reshape(*x.shape[:-1], mul_in, -1)
    x_arr = (x_arr * y_arr).reshape(*x.shape[:-1], -1)
    return e3nn.IrrepsArray(x.irreps, x_arr)


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
    scalar_num_scale: int | None = None
    num_linear: int | None = None

    @nn.compact
    def __call__(
        self,
        node_feats: jax.Array,
        chi: e3nn.IrrepsArray,
    ) -> tuple[jax.Array, e3nn.IrrepsArray]:
        num_features = node_feats.shape[-1]

        # Tensor product using CG coefficents has been removed for simplicity.
        if self.scalar_num_scale is not None:
            chi_left, chi_right = AlignedLinear(
                2 * chi.irreps.regroup().mul_gcd * self.scalar_num_scale, split=2
            )(chi)
            chi_scalar = aligned_dot(chi_left, chi_right)
        else:
            chi_scalar = aligned_norm(chi)

        feats = jnp.concatenate([node_feats, chi_scalar], axis=-1)
        num_chi_coeffs = chi.irreps.num_irreps
        if self.num_linear is not None:
            num_chi_coeffs *= self.num_linear + 1
        feats = nn.Dense(num_features + num_chi_coeffs)(feats)

        # node_feats: [n_nodes, num_features], chi_coeffs: [n_nodes, n_heads]
        node_feats, chi_coeffs = jnp.split(feats, [num_features], axis=-1)

        if self.num_linear is not None:
            chi_coeffs = jnp.split(chi_coeffs, self.num_linear + 1, axis=-1)
            for coeff in chi_coeffs[:-1]:
                chi = AlignedLinear(chi.irreps.regroup().mul_gcd)(aligned_mul(chi, coeff))
            return node_feats, aligned_mul(chi, chi_coeffs[-1])

        return node_feats, aligned_mul(chi, chi_coeffs)


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
        chi_scalar: jax.Array | None, # None for first layer
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
        chi_scalar: jax.Array | None, # None for first layer
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

        chi = e3nn.scatter_sum(
            aligned_mul(edge_sh, alpha), dst=receivers, output_size=node_feats.shape[0]
        )
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
        chi_scalar: jax.Array | None, # None for first layer
        senders: jax.Array,
        receivers: jax.Array
    ) -> jax.Array:
        head_dim = node_feats.shape[-1] // self.num_heads

        # Radial spherical filter
        w_ij = MLP(self.rad_features, self.activation)(edge_feats)
        if chi_scalar is not None:
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
