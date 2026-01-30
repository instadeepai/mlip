# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Yi-Lun Liao (https://github.com/atomicarchitects/equiformer_v2) under MIT license.

"""
In e3nn @ 0.4.0, the Wigner-D matrix is computed using Jd, while in e3nn @ 0.5.0,
it is computed using generators and matrix_exp causing a significant slowdown.
However, in e3nn_jax, `_wigner_D_from_angles` uses Jd for l <= 11 and matrix_exp
for l > 11, so it is well-optimized and there is no need for reimplement.
"""

from functools import cache

from e3nn_jax._src.J import Jd
from e3nn_jax._src.s2grid import (
    _spherical_harmonics_s2grid, _normalization, _expand_matrix, _rollout_sh
)
from flax.typing import Dtype
from flax.struct import dataclass
import jax
import jax.numpy as jnp

from mlip.models.equiformer_v2.utils import get_order_mask, get_rescale_mat, get_mapping_coeffs


def _chebyshev(cos_x: jax.Array, sin_x: jax.Array, lmax: int) -> tuple[jax.Array, jax.Array]:
    """Calculate cos(nx) and sin(nx) using Chebyshev polynomials.
    
    Args:
        cos_x (jax.Array): Cosine of the angle x.
        sin_x (jax.Array): Sine of the angle x.
        lmax (int): Maximum degree of the representation.
    
    Returns:
        Tuple of arrays (cos_nx, sin_nx) with shape of (..., lmax) and (..., lmax).
    """

    if lmax == 1:
        return cos_x[..., None], sin_x[..., None]

    cos_2x = 2 * cos_x * cos_x - 1
    sin_2x = 2 * cos_x * sin_x

    if lmax == 2:
        return jnp.stack([cos_x, cos_2x], axis=-1), jnp.stack([sin_x, sin_2x], axis=-1)

    init_carry = (jnp.stack([cos_2x, sin_2x]), jnp.stack([cos_x, sin_x]))

    def body(carry, _):
        prev, prev2 = carry
        out = 2 * cos_x * prev - prev2
        carry = (out, prev)
        return carry, out

    _, results = jax.lax.scan(body, init_carry, length=lmax - 2)
    results = results.transpose(*range(1, len(results.shape)), 0)
    cos_all = jnp.concat([cos_x[..., None], cos_2x[..., None], results[0]], axis=-1)
    sin_all = jnp.concat([sin_x[..., None], sin_2x[..., None], results[1]], axis=-1)

    return cos_all, sin_all


def _rot_y(cos_x: jax.Array, sin_x: jax.Array, lmax: int) -> list[jax.Array]:
    """Rotational matrix around y-axis by angle phi.
    
    Args:
        cos_x (jax.Array): Cosine of the angle.
        sin_x (jax.Array, optional): Sine of the angle.
        lmax (int): Maximum degree of representation to return.
    """
    cos_all, sin_all = _chebyshev(cos_x, sin_x, lmax)
    cos_all = jnp.concat([cos_all[..., ::-1], jnp.ones_like(cos_x)[..., None], cos_all], axis=-1)
    sin_all = jnp.concat([sin_all[..., ::-1], jnp.zeros_like(sin_x)[..., None], -sin_all], axis=-1)

    rot_mat_list = []
    for l in range(lmax + 1):
        rot_mat = jnp.zeros(cos_x.shape + (2 * l + 1, 2 * l + 1), dtype=cos_x.dtype)
        inds = jnp.arange(0, 2 * l + 1, 1)
        rev_inds = jnp.arange(2 * l, -1, -1)
        rot_mat = rot_mat.at[..., inds, rev_inds].set(sin_all[..., lmax-l:lmax+l+1])
        rot_mat = rot_mat.at[..., inds, inds].set(cos_all[..., lmax-l:lmax+l+1])
        rot_mat_list.append(rot_mat)

    return rot_mat_list


def _wigner_d_from_angles(
    alpha: tuple[jax.Array, jax.Array],
    beta: tuple[jax.Array, jax.Array],
    gamma: tuple[jax.Array, jax.Array],
    lmax: int,
) -> list[jax.Array]:
    r"""The Wigner-D matrix of the real irreducible representations of :math:`SO(3)`.

    Args:
        
        alpha (jax.Array): Cosine and sine of the first Euler angle.
        beta (jax.Array): Cosine and sine of the second Euler angle.
        gamma (jax.Array): Cosine and sine of the third Euler angle.
        lmax (int): The representation order of the irrep.

    Returns:
        List of Wigner-D matrices from 0 to lmax.
    """

    alpha_mats = _rot_y(alpha[0], alpha[1], lmax)
    beta_mats = _rot_y(beta[0], beta[1], lmax)
    gamma_mats = _rot_y(gamma[0], gamma[1], lmax)

    mats = []
    for l, (a, b, c) in enumerate(zip(alpha_mats, beta_mats, gamma_mats)):
        if l < len(Jd):
            j = Jd[l].astype(b.dtype)
            b = j @ b @ j
        else:
            # TODO(bhcao): implement Wigner-D for l > 11
            # x = generators(l)
            # b = jax.scipy.linalg.expm(b.astype(x.dtype) * x[0]).astype(b.dtype)
            raise NotImplementedError("Wigner-D not implemented for l > 11")
        mats.append(a @ b @ c)

    return mats


def _xyz_to_angles(xyz: jax.Array):
    r"""The rotation :math:`R(\alpha, \beta, 0)` such that :math:`\vec r = R \vec e_y`.

    .. math::
        \vec r = R(\alpha, \beta, 0) \vec e_y
        \alpha = \arctan(x/z)
        \beta = \arccos(y)

    Args:
        xyz (`jax.Array`): array of shape :math:`(..., 3)`

    Returns:
        (tuple): tuple of `(\cos(\alpha), \sin(\alpha))`, `(\cos(\beta), \sin(\beta))`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    xyz2 = x**2 + y**2 + z**2
    len_xyz = jnp.sqrt(jnp.where(xyz2 > 1e-16, xyz2, 1e-16))
    xz2 = x**2 + z**2
    len_xz = jnp.sqrt(jnp.where(xz2 > 1e-16, xz2, 1e-16))

    sin_alpha = jnp.clip(x / len_xz, -1, 1)
    cos_alpha = jnp.clip(z / len_xz, -1, 1)

    sin_beta = jnp.clip(len_xz / len_xyz, 0, 1)
    cos_beta = jnp.clip(y / len_xyz, -1, 1)

    return (cos_alpha, sin_alpha), (cos_beta, sin_beta)


def _get_s2grid_mat(
    lmax: int,
    res_beta: int,
    res_alpha: int,
    *,
    dtype: Dtype = jnp.float32,
    normalization: str = "integral",
) -> tuple[jax.Array, jax.Array]:
    r"""Modified `e3nn_jax._src.s2grid.to_s2grid` and `e3nn_jax._src.s2grid.from_s2grid` to act
    like `e3nn.o3.ToS2Grid` and `e3nn.o3.FromS2Grid`.

    Args:
        lmax (int): Maximum degree of the spherical harmonics
        res_beta (int): Number of points on the sphere in the :math:`\theta` direction
        res_alpha (int): Number of points on the sphere in the :math:`\phi` direction
        normalization ({'norm', 'component', 'integral'}): Normalization of the basis

    Returns:
        (to_grid_mat, from_grid_mat):
            Transform matrix from irreps to spherical grid and its inverse.
    """
    _, _, sh_y, sha, qw = _spherical_harmonics_s2grid(
        lmax, res_beta, res_alpha, quadrature="soft", dtype=dtype
    )
    # sh_y: (res_beta, l, |m|)
    sh_y = _rollout_sh(sh_y, lmax)

    m = jnp.asarray(_expand_matrix(range(lmax + 1)), dtype)  # [l, m, i]

    # construct to_grid_mat
    n_to = _normalization(lmax, normalization, dtype, "to_s2")
    sh_y_to = jnp.einsum("lmj,bj,lmi,l->mbi", m, sh_y, m, n_to)  # [m, b, i]
    to_grid_mat = jnp.einsum("mbi,am->bai", sh_y_to, sha)  # [beta, alpha, i]

    # construct from_grid_mat
    n_from = _normalization(lmax, normalization, dtype, "from_s2", lmax)
    sh_y_from = jnp.einsum("lmj,bj,lmi,l,b->mbi", m, sh_y, m, n_from, qw)  # [m, b, i]
    from_grid_mat = jnp.einsum("mbi,am->bai", sh_y_from, sha / res_alpha) # [beta, alpha, i]
    return to_grid_mat, from_grid_mat


# There is no need to promote the dtype because it is determined by the input.
@dataclass
class WignerMats:
    """Wigner-D matrix"""

    wigner: jax.Array
    wigner_inv: jax.Array

    def rotate(self, embedding):
        """Rotate the embedding, l primary -> m primary."""
        return jnp.matmul(self.wigner, embedding)

    def rotate_inv(self, embedding):
        """Rotate the embedding by the inverse of rotation matrix, m primary -> to l primary."""
        return jnp.matmul(self.wigner_inv, embedding)


def get_wigner_mats(
    lmax: int,
    mmax: int,
    xyz: jax.Array,
    gamma: jax.Array,
    perm: jax.Array,
    scale: bool = True,
) -> WignerMats:
    """
    Init the Wigner-D matrix for given euler angles. For continuity of derivatives, `alpha`
    and `beta` are implicitly calculated through given `xyz`. Mathematically, it is
    equivalent to calculate `alpha, beta = xyz_to_angles(xyz)`.
    """
    mask = get_order_mask(lmax, mmax)
    # Compute the re-scaling for rotating back to original frame
    if scale:
        rotate_inv_rescale = jnp.asarray(get_rescale_mat(lmax, mmax, dim=2), dtype=xyz.dtype)
        rotate_inv_rescale = rotate_inv_rescale[None, :, mask]

    alpha, beta = _xyz_to_angles(xyz)
    gamma = jnp.cos(gamma), jnp.sin(gamma)
    blocks = _wigner_d_from_angles(alpha, beta, gamma, lmax)

    # Cache the Wigner-D matrices
    size = (lmax + 1) ** 2
    wigner_inv = jnp.zeros([len(xyz), size, size], dtype=xyz.dtype)
    start = 0
    for i, block in enumerate(blocks):
        end = start + block.shape[1]
        wigner_inv = wigner_inv.at[:, start:end, start:end].set((-1) ** i * block)
        start = end

    # Mask the output to include only modes with m < mmax
    wigner_inv = wigner_inv[:, :, mask]
    wigner = wigner_inv.transpose((0, 2, 1))

    if scale:
        wigner_inv *= rotate_inv_rescale

    wigner = wigner[:, perm, :]
    wigner_inv = wigner_inv[:, :, perm]

    return WignerMats(wigner, wigner_inv)


@dataclass
class S2GridMats:
    """Scaled S2 grid matrix"""

    to_grid_mat: jax.Array
    from_grid_mat: jax.Array

    def to_grid(self, embedding: jax.Array) -> jax.Array:
        """Compute grid from irreps representation"""
        to_grid_mat = jnp.asarray(self.to_grid_mat, dtype=embedding.dtype)
        grid = jnp.einsum("bai, zic -> zbac", to_grid_mat, embedding)
        return grid

    def from_grid(self, grid: jax.Array) -> jax.Array:
        """Compute irreps from grid representation"""
        from_grid_mat = jnp.asarray(self.from_grid_mat, dtype=grid.dtype)
        embedding = jnp.einsum("bai, zbac -> zic", from_grid_mat, grid)
        return embedding


@cache
def get_s2grid_mats(
    lmax: int,
    mmax: int,
    normalization: str = "component",
    resolution: int | None = None,
    m_prime: bool = False,
) -> S2GridMats:
    """Create the S2Grid matrix for given lmax and mmax."""
    mask = get_order_mask(lmax, mmax)

    if resolution is not None:
        lat_resolution = resolution
        long_resolution = resolution
    else:
        lat_resolution = 2 * (lmax + 1)
        long_resolution = 2 * (mmax + 1 if lmax == mmax else mmax) + 1

    # rescale last dimension based on mmax
    rescale_matrix = get_rescale_mat(lmax, mmax)

    to_grid_mat, from_grid_mat = _get_s2grid_mat(
        lmax,
        lat_resolution,
        long_resolution,
        normalization=normalization,
    )
    to_grid_mat = (to_grid_mat * rescale_matrix)[:, :, mask]
    from_grid_mat = (from_grid_mat * rescale_matrix)[:, :, mask]

    if m_prime:
        # This will be reused by lru_cache.
        perm = get_mapping_coeffs(lmax, mmax).perm
        to_grid_mat = to_grid_mat[:, :, perm]
        from_grid_mat = from_grid_mat[:, :, perm]

    return S2GridMats(to_grid_mat, from_grid_mat)
