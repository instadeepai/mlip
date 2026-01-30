# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Yi-Lun Liao (https://github.com/atomicarchitects/equiformer_v2) under MIT license.

from enum import Enum
from functools import cache

import jax
import jax.numpy as jnp
from flax.struct import dataclass


class AttnActType(Enum):
    """The options are as follows. Parameters not mentioned are False.

    Attributes:
        GATE: use_gate_act=True
        S2_SEP: use_sep_s2_act=True
        S2: else
    """
    GATE = 'gate'
    S2_SEP ='s2_sep'
    S2 ='s2'


class FeedForwardType(Enum):
    """The options are as follows. Parameters not mentioned are False.

    Attributes:
        GATE: Spectral atomwise, use_gate_act=True
        GRID: Grid atomwise, use_grid_mlp=True
        GRID_SEP: Grid atomwise, use_grid_mlp=True, use_sep_s2_act=True
        S2: S2 activation
        S2_SEP: S2 activation, use_sep_s2_act=True
    """
    GATE = 'gate'
    GRID = 'grid'
    GRID_SEP = 'grid_sep'
    S2 = 's2'
    S2_SEP = 's2_sep'


def pyg_softmax(src: jax.Array, index: jax.Array, num_segments: int) -> jax.Array:
    r"""Computes a sparsely evaluated softmax referenced from torch_geometric.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.

    Returns:
        The softmax-ed tensor.
    """

    src_max = jax.ops.segment_max(src, index, num_segments)
    out = src - src_max[index]
    out = jnp.exp(out)
    out_sum = jax.ops.segment_sum(out, index, num_segments) + 1e-16
    out_sum = out_sum[index]

    return out / out_sum


@cache
def get_expand_index(
    lmax: int, mmax: int = None, vector_only: bool = False, m_prime: bool = False
) -> jax.Array:
    """Expand coefficients from l or l-1 values on different irreps to (l+1)**2 or (1+1)**2-1
    values on all elements.
    
    Args:
        lmax: Maximum degree (l).
        mmax (optional): Maximum order (m), defaults to `lmax`.
        vector_only (optional): If True, only return coefficients on vector part, otherwise, return
            coefficients on both scalar and vector parts.
        m_prime (optional): If True, indices are in order (l0m0, l1m0, l2m0, l1m1, l2m1, l2m2,
            l1m-1, l2m-1, l2m-2, ...), otherwise, indices are in order (l0m0, l1m-1, l1m0, l1m1,
            l2m-2, l2m-1, l2m0, l2m1, l2m2, ...).
    """
    if mmax is None:
        mmax = lmax
    lmin = 1 if vector_only else 0

    expand_index_list = []

    if m_prime:
        expand_index_list.append(jnp.arange(0, lmax + 1 - lmin))
        for mval in range(1, mmax + 1):
            expand_index_list.extend([
                jnp.arange(mval - lmin, lmax + 1 - lmin),
                jnp.arange(mval - lmin, lmax + 1 - lmin),
            ])
    else:
        for lval in range(lmin, lmax + 1):
            length = min((2 * lval + 1), (2 * mmax + 1))
            expand_index_list.append(
                jnp.ones([length], dtype=jnp.int32) * (lval - lmin)
            )

    return jnp.concat(expand_index_list)


@cache
def get_order_mask(lmax: int, mmax: int, lmax_emb: int = None) -> jax.Array:
    """Compute the mask of orders less than or equal to `mmax` on IrrepsArray of
    `(1, ..., lmax)`."""

    if lmax_emb is None:
        lmax_emb = lmax

    # Compute the degree (lval) and order (m) for each entry of the embedding
    m_harmonic_list = []
    l_harmonic_list = []
    for lval in range(lmax_emb + 1):
        m = jnp.arange(-lval, lval + 1)
        m_harmonic_list.append(jnp.abs(m))
        l_harmonic_list.append(jnp.ones_like(m) * lval)

    m_harmonic = jnp.concat(m_harmonic_list)
    l_harmonic = jnp.concat(l_harmonic_list)

    # Compute the indices of the entries to keep
    # We only use a subset of m components for SO(2) convolution
    return jnp.logical_and(l_harmonic <= lmax, m_harmonic <= mmax)


@cache
def get_rescale_mat(lmax: int, mmax: int, dim: int = 1) -> jax.Array:
    """Rescale matrix for masked entries based on `mmax`."""

    size = (lmax + 1) ** 2
    matrix = jnp.ones([size] * dim)

    if lmax != mmax:
        for lval in range(lmax + 1):
            if lval <= mmax:
                continue
            start = lval ** 2
            length = 2 * lval + 1
            rescale_factor = jnp.sqrt(length / (2 * mmax + 1))
            slices = [slice(start, start + length)] * dim
            matrix = matrix.at[*slices].set(rescale_factor)

    return matrix


@dataclass
class MappingCoeffs:
    """Holds the mapping coefficients to reduce parameters."""
    lmax: int
    mmax: int
    perm: jax.Array
    m_size: tuple[int, ...]
    num_coefficients: int


@cache
def get_mapping_coeffs(lmax: int, mmax: int) -> MappingCoeffs:
    """Return the mapping matrix from lval <--> m and size of each degree."""

    # Compute the degree (lval) and order (m) for each entry of the embedding
    m_complex_list = []

    num_coefficients = 0
    for lval in range(lmax + 1):
        mmax_ = min(mmax, lval)
        m = jnp.arange(-mmax_, mmax_ + 1)
        m_complex_list.append(m)
        num_coefficients += len(m)

    m_complex = jnp.concat(m_complex_list, axis=0)

    # `perm` moves m components from different L to contiguous index (m_prime)
    perm_list = []
    m_size = []

    for m in range(mmax + 1):
        indices = jnp.arange(len(m_complex))

        # Real part
        idx_r = indices[m_complex == m]
        perm_list.append(idx_r)

        m_size.append(len(idx_r))

        # Imaginary part
        if m != 0:
            idx_i = indices[m_complex == -m]
            perm_list.append(idx_i)

    perm = jnp.concat(perm_list)

    return MappingCoeffs(lmax, mmax, perm, m_size, num_coefficients)
