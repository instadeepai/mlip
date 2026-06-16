# Code converted from https://github.com/facebookresearch/fairchem
# Some parts of the code may remain identical, distributed under:
#
#     MIT License- Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Copyright 2026 InstaDeep Ltd and Zhongguancun Academy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General Wigner D matrix computation for arbitrary l via Ra/Rb decomposition.

Only used if esen config has `l_max >= 5`. Used by `wigner_hybrid.py` for l>=5,
where the custom kernels in `wigner_custom.py` are not defined.

The approach decomposes a unit quaternion q = (w, x, y, z) into two complex
numbers Ra = w + iz and Rb = y + ix, then expresses each Wigner D matrix
element as a polynomial in Ra, Rb and their conjugates. Two cases
handle numerical stability depending on whether |Ra| >= |Rb| (Case 1) or
|Ra| < |Rb| (Case 2), with Horner evaluation to reduce roundoff.

The computation requires float64: the polynomials reach degree 2*lmax,
and the exp/log of magnitudes can cause overflow and nans in float32.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from flax.typing import Dtype


@dataclass
class CaseCoeffs:
    """Polynomial coefficients for one case (|Ra|>=|Rb| or |Ra|<|Rb|)."""

    coeff: jax.Array
    horner: jax.Array
    poly_len: jax.Array
    ra_exp: jax.Array
    rb_exp: jax.Array
    sign: jax.Array

    @classmethod
    def from_tuples(cls, coeff_tuples: list[tuple[float, ...]]) -> "CaseCoeffs":
        """Construct CaseCoeffs from list of tuples of coefficients.

        Args:
            coeff_tuples: List of tuples of coefficients, one per term.

        Returns:
            CaseCoeffs object.
        """
        dtype = coeff_tuples[0][1].dtype

        coeff = jnp.array([t[0] for t in coeff_tuples], dtype=dtype)
        horner = jnp.stack([t[1] for t in coeff_tuples])
        poly_len = jnp.array([t[2] for t in coeff_tuples], dtype=jnp.int64)
        ra_exp = jnp.array([t[3] for t in coeff_tuples], dtype=dtype)
        rb_exp = jnp.array([t[4] for t in coeff_tuples], dtype=dtype)
        sign = jnp.array([t[5] for t in coeff_tuples], dtype=dtype)

        return cls(coeff, horner, poly_len, ra_exp, rb_exp, sign)


@dataclass
class WignerCoefficients:
    """Precomputed coefficients for Wigner D matrix computation."""

    lmin: int
    lmax: int
    size: int
    n_primary: int
    n_derived: int

    primary_row: jax.Array
    primary_col: jax.Array

    case1: CaseCoeffs
    case2: CaseCoeffs

    mp_plus_m: jax.Array
    m_minus_mp: jax.Array

    diagonal_mask: jax.Array
    anti_diagonal_mask: jax.Array
    special_2m: jax.Array
    anti_diag_sign: jax.Array

    derived_row: jax.Array
    derived_col: jax.Array
    derived_primary_idx: jax.Array
    derived_sign: jax.Array


def _factorial_table(n: int) -> np.ndarray:
    """Compute factorial table [0!, 1!, 2!, ..., n!]."""
    table = np.zeros(n + 1, dtype=np.float64)
    table[0] = 1.0
    for i in range(1, n + 1):
        table[i] = table[i - 1] * i
    return table


def _binomial(n: int, k: int, factorial: np.ndarray) -> float:
    """Compute binomial coefficient C(n, k) using precomputed factorials."""
    if k < 0 or k > n:
        return 0.0
    return float(factorial[n] / (factorial[k] * factorial[n - k]))


def _compute_case_coefficients(
    ell: int,
    mp: int,
    m: int,
    sqrt_factor: float,
    factorial: np.ndarray,
    max_poly_len: int,
    is_case1: bool,
    dtype: Dtype,
) -> tuple[float, np.ndarray, int, float, float, float]:
    """Compute polynomial coefficients for Case1 or Case2.

    Case1 (|Ra| >= |Rb|): rho ranges [max(0, mp-m), min(l+mp, l-m)]
    Case2 (|Ra| < |Rb|): rho ranges [max(0, -(mp+m)), min(l-m, l-mp)]

    Args:
        ell: Angular momentum quantum number.
        mp, m: Magnetic quantum numbers.
        sqrt_factor: Precomputed sqrt(factorial ratios).
        factorial: Factorial lookup table.
        is_case1: True for Case1, False for Case2.
        max_poly_len: Maximum polynomial length.
        dtype: Data type for horner coefficients.

    Returns:
        Tuple of (coeff, horner, poly_len, ra_exp, rb_exp, sign).
    """
    if is_case1:
        rho_min = max(0, mp - m)
        rho_max = min(ell + mp, ell - m)
    else:
        rho_min = max(0, -(mp + m))
        rho_max = min(ell - m, ell - mp)

    if rho_min > rho_max:
        return 0.0, np.zeros(max_poly_len, dtype=dtype), 0, 0.0, 0.0, 0.0

    if is_case1:
        binom1 = _binomial(ell + mp, rho_min, factorial)
        binom2 = _binomial(ell - mp, ell - m - rho_min, factorial)
    else:
        binom1 = _binomial(ell + mp, ell - m - rho_min, factorial)
        binom2 = _binomial(ell - mp, rho_min, factorial)
    coeff = sqrt_factor * binom1 * binom2

    poly_len = rho_max - rho_min + 1

    horner = np.zeros(max_poly_len, dtype=dtype)
    for i, rho in enumerate(range(rho_max, rho_min, -1)):
        if is_case1:
            n1 = ell + mp - rho + 1
            n2 = ell - m - rho + 1
            d1 = rho
            d2 = m - mp + rho
        else:
            n1 = ell - m - rho + 1
            n2 = ell - mp - rho + 1
            d1 = rho
            d2 = mp + m + rho
        if d1 != 0 and d2 != 0:
            horner[i] = (n1 * n2) / (d1 * d2)

    if is_case1:
        ra_exp = 2 * ell + mp - m - 2 * rho_min
        rb_exp = m - mp + 2 * rho_min
        sign = (-1) ** rho_min
    else:
        ra_exp = mp + m + 2 * rho_min
        rb_exp = 2 * ell - mp - m - 2 * rho_min
        sign = ((-1) ** (ell - m)) * ((-1) ** rho_min)

    return coeff, horner, poly_len, ra_exp, rb_exp, sign


def _precompute_wigner_coefficients(
    lmax: int, lmin: int, dtype: Dtype = jnp.float64
) -> WignerCoefficients:
    """Precompute Wigner D coefficients for l in [lmin, lmax].

    Uses the symmetry D^l_{-m',-m} = (-1)^{m'-m} x conj(D^l_{m',m}) to compute
    only ~half the elements ("primary") and derive the rest ("derived").

    Primary elements: m' + m > 0, OR (m' + m = 0 AND m' >= 0)

    This version supports an optional lmin parameter for memory-efficient
    computation when lower l values are computed via other methods.

    Args:
        lmax: Maximum angular momentum.
        lmin: Minimum angular momentum.
        dtype: Data type for coefficients (float64 used by calling method).

    Returns:
        WignerCoefficients with symmetric coefficient tables.
    """
    factorial = _factorial_table(2 * lmax + 1)

    n_total = sum((2 * ell + 1) ** 2 for ell in range(lmin, lmax + 1))
    n_primary = sum(
        1
        for ell in range(lmin, lmax + 1)
        for mp in range(-ell, ell + 1)
        for m in range(-ell, ell + 1)
        if mp + m > 0 or (mp + m == 0 and mp >= 0)
    )
    n_derived = n_total - n_primary
    max_poly_len = lmax + 1
    size = (lmax + 1) ** 2 - lmin**2

    primary_row = []
    primary_col = []
    mp_plus_m = []
    m_minus_mp = []

    diagonal_mask = []
    anti_diagonal_mask = []
    special_2m = []
    anti_diag_sign = []

    case1 = []
    case2 = []

    primary_map = {}
    primary_idx = 0
    block_start = 0

    for ell in range(lmin, lmax + 1):
        block_size = 2 * ell + 1

        for mp_local in range(block_size):
            mp = mp_local - ell
            for m_local in range(block_size):
                m = m_local - ell
                row = block_start + mp_local
                col = block_start + m_local

                is_primary = (mp + m > 0) or (mp + m == 0 and mp >= 0)
                if not is_primary:
                    continue

                primary_map[(row, col)] = primary_idx
                primary_row.append(row)
                primary_col.append(col)
                mp_plus_m.append(mp + m)
                m_minus_mp.append(m - mp)

                diagonal_mask.append(mp == m)
                anti_diagonal_mask.append(mp == -m)
                special_2m.append(2 * m)
                anti_diag_sign.append((-1) ** (ell - m))

                sqrt_factor = math.sqrt(
                    float(factorial[ell + m] * factorial[ell - m])
                    / float(factorial[ell + mp] * factorial[ell - mp])
                )

                case1.append(
                    _compute_case_coefficients(
                        ell,
                        mp,
                        m,
                        sqrt_factor,
                        factorial,
                        max_poly_len,
                        is_case1=True,
                        dtype=dtype,
                    )
                )
                case2.append(
                    _compute_case_coefficients(
                        ell,
                        mp,
                        m,
                        sqrt_factor,
                        factorial,
                        max_poly_len,
                        is_case1=False,
                        dtype=dtype,
                    )
                )

                primary_idx += 1

        block_start += block_size

    primary_row = jnp.array(primary_row, dtype=jnp.int64)
    primary_col = jnp.array(primary_col, dtype=jnp.int64)
    mp_plus_m = jnp.array(mp_plus_m, dtype=dtype)
    m_minus_mp = jnp.array(m_minus_mp, dtype=dtype)
    diagonal_mask = jnp.array(diagonal_mask, dtype=jnp.bool_)
    anti_diagonal_mask = jnp.array(anti_diagonal_mask, dtype=jnp.bool_)
    special_2m = jnp.array(special_2m, dtype=dtype)
    anti_diag_sign = jnp.array(anti_diag_sign, dtype=dtype)
    case1 = CaseCoeffs.from_tuples(case1)
    case2 = CaseCoeffs.from_tuples(case2)

    derived_row = []
    derived_col = []
    derived_primary_idx = []
    derived_sign = []

    block_start = 0
    for ell in range(lmin, lmax + 1):
        block_size = 2 * ell + 1

        for mp_local in range(block_size):
            mp = mp_local - ell
            for m_local in range(block_size):
                m = m_local - ell
                row = block_start + mp_local
                col = block_start + m_local

                is_primary = (mp + m > 0) or (mp + m == 0 and mp >= 0)
                if is_primary:
                    continue

                neg_mp_local = -mp + ell
                neg_m_local = -m + ell
                primary_row_idx = block_start + neg_mp_local
                primary_col_idx = block_start + neg_m_local

                derived_row.append(row)
                derived_col.append(col)
                derived_primary_idx.append(
                    primary_map[(primary_row_idx, primary_col_idx)]
                )
                derived_sign.append((-1) ** (mp - m))

        block_start += block_size

    derived_row = jnp.array(derived_row, dtype=jnp.int64)
    derived_col = jnp.array(derived_col, dtype=jnp.int64)
    derived_primary_idx = jnp.array(derived_primary_idx, dtype=jnp.int64)
    derived_sign = jnp.array(derived_sign, dtype=dtype)

    return WignerCoefficients(
        lmin=lmin,
        lmax=lmax,
        size=size,
        primary_row=primary_row,
        primary_col=primary_col,
        n_primary=n_primary,
        case1=case1,
        case2=case2,
        mp_plus_m=mp_plus_m,
        m_minus_mp=m_minus_mp,
        diagonal_mask=diagonal_mask,
        anti_diagonal_mask=anti_diagonal_mask,
        special_2m=special_2m,
        anti_diag_sign=anti_diag_sign,
        n_derived=n_derived,
        derived_row=derived_row,
        derived_col=derived_col,
        derived_primary_idx=derived_primary_idx,
        derived_sign=derived_sign,
    )


def _vectorized_horner(
    ratio: jax.Array, horner_coeffs: jax.Array, poly_len: jax.Array
) -> jax.Array:
    """Vectorized Horner polynomial evaluation for all elements simultaneously.

    Evaluates polynomials of varying lengths using masking.

    Args:
        ratio: The ratio term -(rb/ra)^2 or -(ra/rb)^2, shape (N,)
        horner_coeffs: Horner factors, shape (n_elements, max_poly_len)
        poly_len: Actual polynomial length per element, shape (n_elements,)

    Returns:
        Polynomial values of shape (N, n_elements)
    """
    n_elements, max_poly_len = horner_coeffs.shape
    result = jnp.ones((ratio.shape[0], n_elements), dtype=ratio.dtype)

    ratio = ratio[:, None]

    for i in range(max_poly_len - 1):
        coeff = horner_coeffs[:, i]
        mask = i < (poly_len - 1)
        factor = ratio * coeff[None]
        new_result = 1.0 + result * factor
        result = jnp.where(mask[None], new_result, result)

    return result


def _compute_case_magnitude(
    log_ra: jax.Array, log_rb: jax.Array, ratio: jax.Array, case: CaseCoeffs
) -> jax.Array:
    """Compute the real-valued magnitude factor for a general case.

    This is the common computation for both Case 1 (|Ra| >= |Rb|) and
    Case 2 (|Ra| < |Rb|), used by both complex and real-pair versions.

    Args:
        log_ra: Log of |Ra| magnitudes, shape (N,).
        log_rb: Log of |Rb| magnitudes, shape (N,).
        ratio: -(rb/ra)^2 for Case 1 or -(ra/rb)^2 for Case 2, shape (N,).
        case: CaseCoeffs with polynomial coefficients for this case.

    Returns:
        Magnitude factor of shape (N, n_primary), real-valued.
    """
    horner_sum = _vectorized_horner(ratio, case.horner, case.poly_len)
    ra_powers = jnp.exp(jnp.outer(log_ra, case.ra_exp))
    rb_powers = jnp.exp(jnp.outer(log_rb, case.rb_exp))
    magnitude = (case.sign * case.coeff) * ra_powers * rb_powers
    return magnitude * horner_sum


def wigner_d_matrix_real(
    q: jax.Array, lmax: int, lmin: int = 0
) -> tuple[jax.Array, jax.Array]:
    """Compute Wigner D matrices using real arithmetic only.

    q = (w, x, y, z) is decomposed into real/imaginary parts of Ra and Rb first:
        Ra = w + i*z  ->  (ra_re=w, ra_im=z)
        Rb = y + i*x  ->  (rb_re=y, rb_im=x)

    Args:
        q: Quaternion of shape (N, 4).
        lmax: Maximum angular momentum.
        lmin: Minimum angular momentum (default 0).

    Returns:
        Tuple (D_re, D_im) - real and imaginary parts of the complex
        block-diagonal matrices, each of shape (N, size, size)
    """
    _x64_was_enabled = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        dtype = jnp.float64
        with jax.ensure_compile_time_eval():
            coeffs = _precompute_wigner_coefficients(lmax, lmin, dtype=dtype)

        input_dtype = q.dtype
        q = jnp.asarray(q, dtype=dtype)

        ra_re, ra_im, rb_re, rb_im = q[..., 0], q[..., 3], q[..., 2], q[..., 1]

        n_batch = ra_re.shape[0]

        eps = jnp.finfo(dtype).eps
        eps_sq = eps * eps
        ra_sq = ra_re * ra_re + ra_im * ra_im
        rb_sq = rb_re * rb_re + rb_im * rb_im
        ra_small = ra_sq <= eps_sq
        rb_small = rb_sq <= eps_sq
        ra = jnp.sqrt(jnp.clip(ra_sq, min=eps_sq))
        rb = jnp.sqrt(jnp.clip(rb_sq, min=eps_sq))
        general_mask = ~ra_small & ~rb_small
        use_case1 = (ra >= rb) & general_mask
        use_case2 = (ra < rb) & general_mask

        safe_ra_re_phi = jnp.where(ra_small, jnp.ones_like(ra_re), ra_re)
        safe_ra_im_phi = jnp.where(ra_small, jnp.zeros_like(ra_im), ra_im)
        phia = jnp.atan2(safe_ra_im_phi, safe_ra_re_phi)

        safe_rb_re_phi = jnp.where(rb_small, jnp.ones_like(rb_re), rb_re)
        safe_rb_im_phi = jnp.where(rb_small, jnp.zeros_like(rb_im), rb_im)
        phib = jnp.atan2(safe_rb_im_phi, safe_rb_re_phi)

        phase = jnp.outer(phia, coeffs.mp_plus_m) + jnp.outer(phib, coeffs.m_minus_mp)
        exp_phase_re = jnp.cos(phase)
        exp_phase_im = jnp.sin(phase)

        safe_ra = jnp.clip(ra, min=eps)
        safe_rb = jnp.clip(rb, min=eps)
        log_ra = jnp.log(safe_ra)
        log_rb = jnp.log(safe_rb)

        result_re = jnp.zeros((n_batch, coeffs.n_primary), dtype=dtype)
        result_im = jnp.zeros((n_batch, coeffs.n_primary), dtype=dtype)

        log_mag_rb_power = jnp.outer(log_rb, coeffs.special_2m)
        rb_power_mag = jnp.exp(log_mag_rb_power)
        rb_power_phase = jnp.outer(phib, coeffs.special_2m)
        rb_power_re = rb_power_mag * jnp.cos(rb_power_phase)
        rb_power_im = rb_power_mag * jnp.sin(rb_power_phase)

        special_val_antidiag_re = coeffs.anti_diag_sign[None] * rb_power_re
        special_val_antidiag_im = coeffs.anti_diag_sign[None] * rb_power_im

        mask_antidiag = ra_small[:, None] & coeffs.anti_diagonal_mask[None]
        result_re = jnp.where(mask_antidiag, special_val_antidiag_re, result_re)
        result_im = jnp.where(mask_antidiag, special_val_antidiag_im, result_im)

        log_mag_ra_power = jnp.outer(log_ra, coeffs.special_2m)
        ra_power_mag = jnp.exp(log_mag_ra_power)
        ra_power_phase = jnp.outer(phia, coeffs.special_2m)
        ra_power_re = ra_power_mag * jnp.cos(ra_power_phase)
        ra_power_im = ra_power_mag * jnp.sin(ra_power_phase)

        mask_diag = (rb_small & ~ra_small)[:, None] & coeffs.diagonal_mask[None]
        result_re = jnp.where(mask_diag, ra_power_re, result_re)
        result_im = jnp.where(mask_diag, ra_power_im, result_im)

        ratio1 = -(rb * rb) / (safe_ra * safe_ra)
        real_factor1 = _compute_case_magnitude(log_ra, log_rb, ratio1, coeffs.case1)
        val1_re = real_factor1 * exp_phase_re
        val1_im = real_factor1 * exp_phase_im

        valid_case1 = coeffs.case1.poly_len > 0
        mask1 = use_case1[:, None] & valid_case1[None]
        result_re = jnp.where(mask1, val1_re, result_re)
        result_im = jnp.where(mask1, val1_im, result_im)

        ratio2 = -(ra * ra) / (safe_rb * safe_rb)
        real_factor2 = _compute_case_magnitude(log_ra, log_rb, ratio2, coeffs.case2)
        val2_re = real_factor2 * exp_phase_re
        val2_im = real_factor2 * exp_phase_im

        valid_case2 = coeffs.case2.poly_len > 0
        mask2 = use_case2[:, None] & valid_case2[None]
        result_re = jnp.where(mask2, val2_re, result_re)
        result_im = jnp.where(mask2, val2_im, result_im)

        d_re = jnp.zeros((n_batch, coeffs.size, coeffs.size), dtype=dtype)
        d_im = jnp.zeros((n_batch, coeffs.size, coeffs.size), dtype=dtype)

        batch_indices = jnp.repeat(
            jnp.arange(n_batch)[:, None], coeffs.n_primary, axis=1
        )
        row_expanded = jnp.repeat(coeffs.primary_row[None], n_batch, axis=0)
        col_expanded = jnp.repeat(coeffs.primary_col[None], n_batch, axis=0)
        d_re = d_re.at[batch_indices, row_expanded, col_expanded].set(result_re)
        d_im = d_im.at[batch_indices, row_expanded, col_expanded].set(result_im)

        if coeffs.n_derived > 0:
            primary_re = result_re[:, coeffs.derived_primary_idx]
            primary_im = result_im[:, coeffs.derived_primary_idx]

            derived_sign_expanded = coeffs.derived_sign[None]
            derived_re = derived_sign_expanded * primary_re
            derived_im = -derived_sign_expanded * primary_im

            batch_indices_d = jnp.repeat(
                jnp.arange(n_batch)[:, None], coeffs.n_derived, axis=1
            )
            row_expanded_d = jnp.repeat(coeffs.derived_row[None], n_batch, axis=0)
            col_expanded_d = jnp.repeat(coeffs.derived_col[None], n_batch, axis=0)
            d_re = d_re.at[batch_indices_d, row_expanded_d, col_expanded_d].set(
                derived_re
            )
            d_im = d_im.at[batch_indices_d, row_expanded_d, col_expanded_d].set(
                derived_im
            )

        d_re = jnp.asarray(d_re, input_dtype)
        d_im = jnp.asarray(d_im, input_dtype)

        return d_re, d_im
    finally:
        jax.config.update("jax_enable_x64", _x64_was_enabled)


def _compute_transform_sign(ell: int, m: int) -> int:
    """Compute the sign for the Euler-matching basis transformation.

    The transformation is a signed row permutation of Jd[ell] that
    converts axis-angle Wigner D matrices to match Euler Wigner D matrices.

    For even |m|: sign = (-1)^((l - |m|) / 2)
    For odd |m|, m < 0: sign = (-1)^((l + |m| + 1) // 2)
    For odd |m|, m > 0: sign = (-1)^((l + |m| + 1) // 2 + 1)
    """
    abs_m = abs(m)
    if abs_m % 2 == 0:
        return (-1) ** ((ell - abs_m) // 2)

    base = (ell + abs_m + 1) // 2
    if m < 0:
        return (-1) ** base

    return (-1) ** (base + 1)


def _build_euler_transform(ell: int, jd: jax.Array) -> jax.Array:
    """Build the basis transformation U for level ell.

    U transforms axis-angle Wigner D to match Euler Wigner D:
        D_euler = U @ D_axis @ U.T

    Args:
        ell: Angular momentum level.
        jd: Wigner d matrix at beta=pi/2 for level ell, shape (2*ell+1, 2*ell+1).

    Returns:
        Orthogonal transformation matrix U of shape (2*ell+1, 2*ell+1).
    """
    size = 2 * ell + 1
    u = jnp.zeros((size, size), dtype=jd.dtype)

    for i in range(size):
        m = i - ell
        abs_m = abs(m)
        if abs_m % 2 == 1:
            jd_row = (-m) + ell
        else:
            jd_row = i

        sign = _compute_transform_sign(ell, m)
        u = u.at[i, :].set(sign * jd[jd_row, :])

    return u


def _build_u_matrix(ell: int, dtype: Dtype = jnp.complex64) -> jax.Array:
    """Build complex-to-real spherical harmonic transformation matrix.

    Uses e3nn convention: m = (-ell, ..., +ell) at indices (0, ..., 2*ell).

    Args:
        ell: Angular momentum quantum number.
        dtype: Complex data type for the matrix (default: complex64).

    Returns:
        U matrix of shape (2*ell+1, 2*ell+1).
    """
    size = 2 * ell + 1
    sqrt2_inv = 1.0 / math.sqrt(2.0)

    u = jnp.zeros((size, size), dtype=dtype)

    for m in range(-ell, ell + 1):
        row = m + ell

        if m > 0:
            col_pos = m + ell
            col_neg = -m + ell
            sign = (-1) ** m
            u = u.at[row, col_pos].set(sign * sqrt2_inv)
            u = u.at[row, col_neg].set(sqrt2_inv)
        elif m == 0:
            u = u.at[row, ell].set(1.0)
        else:
            abs_m = abs(m)
            col_pos = abs_m + ell
            col_neg = -abs_m + ell
            sign = (-1) ** abs_m
            u = u.at[row, col_neg].set(1j * sqrt2_inv)
            u = u.at[row, col_pos].set(-sign * 1j * sqrt2_inv)

    return u


def _precompute_u_blocks(
    lmax: int, lmin: int, jd_list: list[jax.Array], dtype: Dtype = jnp.float32
) -> list[jax.Array]:
    """Private helper to precompute complex U transformation matrices.

    This combines the complex->real transformation with:
    - For l=1: The Cartesian permutation P (m-ordering -> x,y,z)
    - For l>=2: The Euler-matching basis transformation

    Args:
        lmax: Maximum angular momentum.
        lmin: Minimum angular momentum.
        jd_list: J_y generators (angular momentum operators) in the |ℓ,m⟩ basis.
        dtype: Real dtype, either float32 or float64 (float32 used by calling method).

    Returns:
        List of combined U matrices (complex) where U_blocks[i] corresponds to l=lmin+i.
    """
    assert dtype in (jnp.float32, jnp.float64), (
        f"dtype must be float32 or float64, got {dtype}"
    )
    complex_dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128

    u_combined = []
    for ell in range(lmin, lmax + 1):
        u_ell = _build_u_matrix(ell, complex_dtype)

        if ell == 0:
            u_combined.append(u_ell)
        elif ell == 1:
            u_combined.append(u_ell[(2, 0, 1)])
        else:
            jd = jnp.asarray(jd_list[ell], dtype=dtype)
            u_euler = jnp.asarray(_build_euler_transform(ell, jd), dtype=complex_dtype)
            u_combined.append(u_euler @ u_ell)

    return u_combined


def wigner_d_pair_to_real(
    d_re: jax.Array,
    d_im: jax.Array,
    jd_list: list[jax.Array],
    lmax: int,
    lmin: int = 0,
) -> jax.Array:
    """Transform Wigner D matrix from real-pair to real basis using real arithmetic.

    Args:
        d_re: Real part of complex Wigner D matrices, shape (N, size, size).
        d_im: Imaginary part of complex Wigner D matrices, shape (N, size, size).
        jd_list: J_y generators (angular momentum operators) in the |ℓ,m⟩ basis.
        lmax: Maximum angular momentum.
        lmin: Minimum angular momentum (default 0).

    Returns:
        Real Wigner D matrices of shape (N, size, size).
    """
    size = d_re.shape[1]
    dtype = d_re.dtype
    d_real = jnp.zeros((d_re.shape[0], size, size), dtype=dtype)

    with jax.ensure_compile_time_eval():
        u_blocks_complex = _precompute_u_blocks(lmax, lmin, jd_list, dtype=dtype)

    block_start = 0
    for idx, ell in enumerate(range(lmin, lmax + 1)):
        block_end = block_start + 2 * ell + 1
        block_slice = (
            slice(None),
            slice(block_start, block_end),
            slice(block_start, block_end),
        )
        block_start = block_end

        d_block_re = d_re[block_slice]
        d_block_im = d_im[block_slice]

        u_complex = u_blocks_complex[idx]
        u_re = jnp.asarray(u_complex.real, dtype=dtype)
        u_im = jnp.asarray(u_complex.imag, dtype=dtype)

        u_re_t = u_re.T
        u_im_t = u_im.T

        temp_re = jnp.matmul(d_block_re, u_re_t) + jnp.matmul(d_block_im, u_im_t)
        temp_im = jnp.matmul(d_block_im, u_re_t) - jnp.matmul(d_block_re, u_im_t)

        result_re = jnp.matmul(u_re, temp_re) - jnp.matmul(u_im, temp_im)

        d_real = d_real.at[block_slice].set(result_re)

    return d_real
