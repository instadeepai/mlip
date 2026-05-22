# Code converted from https://github.com/facebookresearch/fairchem
# Some parts of the code may remain identical, distributed under:
#
#     MIT License- Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Copyright 2025 InstaDeep Ltd
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


import math
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np


class CoefficientMapping:
    """This is a bookkeeping/indexing utility for truncated spherical-harmonic-
    coefficients. It enumerates all (l, m) pairs under (l_max, m_max), builds a
    permutation to reorder coefficients from l-major to m-major layout,
    caches index sets for fast truncation queries (l ≤ L, |m| ≤ M),
    and provides rescaling blocks used when forming rotation-invariant quantities
    under m-truncation. No learning happens here; it is purely structural support
    for SO(2)/SO(3)-equivariant operations."""

    def __init__(self, l_max: int, m_max: int):
        self.l_max = int(l_max)
        self.m_max = int(m_max)

        l_list = []
        m_abs_list = []
        m_complex_list = []

        for l_number in range(self.l_max + 1):
            m_max_l = min(self.m_max, l_number)
            m = np.arange(-m_max_l, m_max_l + 1, dtype=np.int64)
            m_complex_list.append(m)
            m_abs_list.append(np.abs(m))
            l_list.append(np.full_like(m, l_number, dtype=np.int64))

        self.l_harmonic_np = np.concatenate(l_list, axis=0)
        self.m_harmonic_np = np.concatenate(m_abs_list, axis=0)
        self.m_complex_np = np.concatenate(m_complex_list, axis=0)
        self.res_size: int = int(self.l_harmonic_np.shape[0])
        num_coeffs = self.res_size

        to_m = np.zeros((num_coeffs, num_coeffs), dtype=np.float32)
        m_size = np.zeros((self.m_max + 1,), dtype=np.int64)

        offset = 0
        for m in range(self.m_max + 1):
            idx_r, idx_i = self._complex_idx_np(
                m, -1, self.m_complex_np, self.l_harmonic_np
            )

            for idx_out, idx_in in enumerate(idx_r):
                to_m[offset + idx_out, idx_in] = 1.0
            offset += len(idx_r)
            m_size[m] = len(idx_r)

            for idx_out, idx_in in enumerate(idx_i):
                to_m[offset + idx_out, idx_in] = 1.0
            offset += len(idx_i)

        self.to_m_np = to_m
        self.m_size_np = m_size

        self._build_coefficient_idx_cache()

        self._rotate_inv_rescale: Dict[Tuple[int, int], np.ndarray] = {}

    def _complex_idx_np(
        self,
        m: int,
        l_max: int,
        m_complex: np.ndarray,
        l_harmonic: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if l_max == -1:
            l_max = self.l_max

        n = l_harmonic.shape[0]
        indices = np.arange(n, dtype=np.int64)

        mask_r = np.logical_and(l_harmonic <= l_max, m_complex == m)
        idx_r = indices[mask_r]

        if m != 0:
            mask_i = np.logical_and(l_harmonic <= l_max, m_complex == -m)
            idx_i = indices[mask_i]
        else:
            idx_i = np.zeros((0,), dtype=np.int64)

        return idx_r, idx_i

    def _build_coefficient_idx_cache(self):

        l_h = self.l_harmonic_np
        m_h = self.m_harmonic_np

        cache_rows: List[Tuple[np.ndarray, ...]] = []
        for l_number in range(self.l_max + 1):
            row: List[np.ndarray] = []
            for m in range(self.l_max + 1):
                mask = np.logical_and(l_h <= l_number, m_h <= m)
                idx = np.nonzero(mask)[0].astype(np.int32)
                row.append(idx)
            cache_rows.append(tuple(row))

        self._coefficient_idx_cache: Tuple[Tuple[np.ndarray, ...], ...] = tuple(
            cache_rows
        )

    def _build_rotate_inv_rescale(self):

        for l_number in range(self.l_max + 1):
            for m in range(self.l_max + 1):
                idx = self._coefficient_idx_cache[l_number][m]
                size_l = (l_number + 1) ** 2
                block = np.ones((1, size_l, size_l), dtype=np.float32)

                for l_sub in range(l_number + 1):
                    if l_sub <= m:
                        continue
                    start = l_sub * l_sub
                    length = 2 * l_sub + 1
                    rescale_factor = math.sqrt(length / (2 * m + 1))
                    block[:, start : start + length, start : start + length] *= (
                        rescale_factor
                    )

                block = block[:, :, idx]
                self._rotate_inv_rescale[(l_number, m)] = block

    @property
    def l_harmonic(self) -> jnp.ndarray:
        return jnp.array(self.l_harmonic_np, dtype=jnp.int32)

    @property
    def m_harmonic(self) -> jnp.ndarray:
        return jnp.array(self.m_harmonic_np, dtype=jnp.int32)

    @property
    def m_complex(self) -> jnp.ndarray:
        return jnp.array(self.m_complex_np, dtype=jnp.int32)

    @property
    def to_m(self) -> jnp.ndarray:
        return jnp.array(self.to_m_np, dtype=jnp.float32)

    @property
    def m_size(self) -> List[int]:
        return self.m_size_np.tolist()

    def coefficient_idx(self, l_max: int, m_max: int) -> jnp.ndarray:
        if l_max <= self.l_max and m_max <= self.l_max:
            idx_np = self._coefficient_idx_cache[l_max][m_max]
            return jnp.array(idx_np, dtype=jnp.int32)
        else:
            l_h = self.l_harmonic
            m_h = self.m_harmonic
            mask = jnp.logical_and(l_h <= l_max, m_h <= m_max)
            return jnp.nonzero(mask, size=self.res_size, fill_value=0)[0]

    def rotate_inv_rescale(self, l_number: int, m: int) -> jnp.ndarray:
        block_np = self._rotate_inv_rescale.get((l_number, m), None)
        if block_np is None:
            raise KeyError(f"rotate_inv_rescale not computed for (l={l_number}, m={m})")
        return jnp.array(block_np, dtype=jnp.float32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(l_max={self.l_max}, m_max={self.m_max})"
