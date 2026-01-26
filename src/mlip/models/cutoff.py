# Copyright 2025 Zhongguancun Academy

"""
This module contains all cutoff / radial envelope functions seen in all models.
It can be used to refactor Mace, Visnet and Nequip.
"""

from enum import Enum

import jax
import jax.numpy as jnp
import flax.linen as nn
import e3nn_jax as e3nn


class SoftCutoff(nn.Module):
    """Soft envelope radial envelope function."""
    cutoff: float
    arg_multiplicator: float = 2.0
    value_at_origin: float = 1.2

    @nn.compact
    def __call__(self, length):
        return e3nn.soft_envelope(
            length,
            self.cutoff,
            arg_multiplicator=self.arg_multiplicator,
            value_at_origin=self.value_at_origin,
        )


class PolynomialCutoff(nn.Module):
    """Polynomial radial envelope function from the MACE torch version."""
    cutoff: float
    p: int = 5

    @nn.compact
    def __call__(self, length: jax.Array):
        a = - (self.p + 1.0) * (self.p + 2.0) / 2.0
        b = self.p * (self.p + 2.0)
        c = - self.p * (self.p + 1.0) / 2

        x_norm = length / self.cutoff
        envelope = 1.0 + jnp.pow(x_norm, self.p) * (
            a + x_norm * (b + x_norm * c)
        )
        return envelope * (length < self.cutoff)


class CosineCutoff(nn.Module):
    """Behler-style cosine cutoff function."""
    cutoff: float

    @nn.compact
    def __call__(self, length: jax.Array) -> jax.Array:
        cutoffs = 0.5 * (jnp.cos(length * jnp.pi / self.cutoff) + 1.0)
        return cutoffs * (length < self.cutoff)


class PhysCutoff(nn.Module):
    """Cutoff function used in PhysNet."""
    cutoff: float

    @nn.compact
    def __call__(self, length: jax.Array) -> jax.Array:
        x_norm = length / self.cutoff
        cutoffs = 1 - 6 * x_norm ** 5 + 15 * x_norm ** 4 - 10 * x_norm ** 3
        return cutoffs * (length < self.cutoff)


class ExponentialCutoff(nn.Module):
    """Exponential cutoff function used in SpookyNet."""
    cutoff: float

    @nn.compact
    def __call__(self, length: jax.Array) -> jax.Array:
        # TODO(bhcao): Check if this is numerically stable.
        cutoffs = jnp.exp(-length ** 2 / ((self.cutoff - length) * (self.cutoff + length)))
        return cutoffs * (length < self.cutoff)

# --- Options ---


class CutoffFunction(Enum):
    POLYNOMIAL = "polynomial"
    SOFT = "soft"
    COSINE = "cosine"
    PHYS = "phys"
    EXPONENTIAL = "exponential"


def parse_cutoff(cutoff: CutoffFunction | str) -> type[nn.Module]:
    cutoff_function_map = {
        CutoffFunction.POLYNOMIAL: PolynomialCutoff,
        CutoffFunction.SOFT: SoftCutoff,
        CutoffFunction.COSINE: CosineCutoff,
        CutoffFunction.PHYS: PhysCutoff,
        CutoffFunction.EXPONENTIAL: ExponentialCutoff,
    }
    assert set(CutoffFunction) == set(cutoff_function_map.keys())
    return cutoff_function_map[CutoffFunction(cutoff)]
