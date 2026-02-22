# Copyright 2025 Zhongguancun Academy

"""
This module contains all radial basis functions seen in all models.
It can be used to refactor Mace, Visnet and Nequip.
"""

from enum import Enum

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import e3nn_jax as e3nn


class ExpNormalBasis(nn.Module):
    """Original ExpNormalSmearing from Visnet without cutoff function."""
    cutoff: float
    num_rbf: int
    trainable: bool = True

    def setup(self):
        self.alpha = 5.0 / self.cutoff
        means, betas = self._initial_params()
        if self.trainable:
            self.means = self.param(
                "means", nn.initializers.constant(means), (self.num_rbf,)
            )
            self.betas = self.param(
                "betas", nn.initializers.constant(betas), (self.num_rbf,)
            )
        else:
            self.means = means
            self.betas = betas

    def _initial_params(self):
        start_value = jnp.exp(-self.cutoff)
        means = jnp.linspace(start_value, 1, self.num_rbf)
        betas = jnp.full((self.num_rbf,), (2 / self.num_rbf * (1 - start_value)) ** -2)
        return means, betas

    def __call__(self, dist: jax.Array) -> jax.Array:
        dist = dist[..., None]
        return jnp.exp(
            (-1 * self.betas) * (jnp.exp(self.alpha * (-dist)) - self.means) ** 2
        )


class GaussianBasis(nn.Module):
    """
    Original GaussianSmearing from Visnet without cutoff function.
    It's also used in So3krates named RBF.
    """
    cutoff: float
    num_rbf: int
    trainable: bool = True
    rbf_width: float = 1.0

    def setup(self):
        offset, coeff = self._initial_params()
        if self.trainable:
            self.offset = self.param(
                "offset", nn.initializers.constant(offset), (self.num_rbf,)
            )
            self.coeff = self.param("coeff", nn.initializers.constant(coeff), ())
        else:
            self.offset = offset
            self.coeff = coeff

    def _initial_params(self):
        offset = jnp.linspace(0, self.cutoff, self.num_rbf)
        coeff = -0.5 / (self.rbf_width * (offset[1] - offset[0])) ** 2
        return offset, coeff

    def __call__(self, dist: jax.Array) -> jax.Array:
        dist = dist[..., None] - self.offset
        return jnp.exp(self.coeff * jnp.square(dist))


class BesselBasis(nn.Module):
    """Bessel basis used in Mace and Nequip. This is not the same named function in So3krates."""
    cutoff: float
    num_rbf: int
    trainable: bool = False # ignored

    @nn.compact
    def __call__(self, dist: jax.Array) -> jax.Array:
        return e3nn.bessel(dist, self.num_rbf, self.cutoff)


def log_binomial(n: int) -> jax.Array:
    """
    Returns: jax.Array of shape (n+1,)
    [log C(n, 0), ..., log C(n, n)]
    """
    out = []
    for k in range(n + 1):
        n_factorial = np.sum(np.log(np.arange(1, n + 1)))
        k_factorial = np.sum(np.log(np.arange(1, k + 1)))
        n_k_factorial = np.sum(np.log(np.arange(1, n - k + 1)))
        out.append(n_factorial - k_factorial - n_k_factorial)
    return jnp.stack(out)

class BernsteinBasis(nn.Module):
    """Bernstein polynomial basis from So3krates."""
    cutoff: float
    num_rbf: int
    gamma: float = 0.9448630629184640
    trainable: bool = False # ignored

    @nn.compact
    def __call__(self, dist: jax.Array) -> jax.Array:
        b = log_binomial(self.num_rbf - 1)
        k = jnp.arange(self.num_rbf)
        k_rev = k[::-1]

        scaled_dist = -self.gamma * dist[..., None]
        k_x = k * scaled_dist
        kk_x = k_rev * jnp.log(1e-8 - jnp.expm1(scaled_dist))
        return jnp.exp(b + k_x + kk_x)


class PhysNetBasis(nn.Module):
    """Expand distances in the basis used in PhysNet (see https://arxiv.org/abs/1902.08408)"""
    cutoff: float
    num_rbf: int
    trainable: bool = False # ignored

    @nn.compact
    def __call__(self, dist: jax.Array) -> jax.Array:
        exp_dist = jnp.exp(-dist)[..., None]
        exp_cutoff = jnp.exp(-self.cutoff)

        offset = jnp.linspace(exp_cutoff, 1, self.num_rbf)
        coeff = self.num_rbf / 2 / (1 - exp_cutoff)
        return jnp.exp(-(coeff * (exp_dist - offset)) ** 2)


class FourierBasis(nn.Module):
    """
    Expand distances in the Bessel basis (see https://arxiv.org/pdf/2003.03123.pdf).
    It's also called Bessel basis in So3krates. Since we already have the BesselBasis, we have to
    use another name.
    """
    cutoff: float
    num_rbf: int
    trainable: bool = False # ignored

    def setup(self):
        self.offset = jnp.arange(0, self.num_rbf, 1)

    def __call__(self, dist: jax.Array) -> jax.Array:
        dist = dist[..., None]
        # In So3krates, safe_mask is used to avoid divide by zero. Here, we use a small epsilon.
        return jnp.sin(jnp.pi / self.cutoff * self.offset * dist) / (dist + 1e-8)

# --- Options ---


class RadialBasis(Enum):
    GAUSS = "gauss"
    EXPNORM = "expnorm"
    BESSEL = "bessel"
    BERNSTEIN = "bernstein"
    PHYS = "phys"
    FOURIER = "fourier"


def parse_radial_basis(basis: RadialBasis | str) -> type[nn.Module]:
    """Parse `RadialBasis` parameter among available options.

    See :class:`~dipm.models.options.RadialBasis`.
    """
    radial_basis_map = {
        RadialBasis.GAUSS: GaussianBasis,
        RadialBasis.EXPNORM: ExpNormalBasis,
        RadialBasis.BESSEL: BesselBasis,
        RadialBasis.BERNSTEIN: BernsteinBasis,
        RadialBasis.PHYS: PhysNetBasis,
        RadialBasis.FOURIER: FourierBasis,
    }
    assert set(RadialBasis) == set(radial_basis_map.keys())
    return radial_basis_map[RadialBasis(basis)]
