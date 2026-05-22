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

import os
import re
from functools import lru_cache

import numpy as np

_DESIGN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Filenames follow the convention `<kind><TTT>.<NNNNN>` where:
#   - kind is `ss` for antipodal designs, `sf` otherwise
#   - TTT  is the design order (zero-padded)
#   - NNNNN is the number of points (zero-padded)
_FILENAME_RE = re.compile(r"^(ss|sf)(\d+)\.(\d+)$")


@lru_cache(maxsize=1)
def _index_designs() -> dict:
    """Index every design file in :data:`_DESIGN_DIR`.

    Returns:
        Mapping `{("ss" | "sf", order): (n_points, filename)}`. When the
        directory contains several files for the same `(kind, order)` pair,
        the one with the fewest points is kept.
    """
    index: dict[tuple[str, int], tuple[int, str]] = {}
    for fname in os.listdir(_DESIGN_DIR):
        match = _FILENAME_RE.match(fname)
        if match is None:
            continue
        kind, order_str, n_str = match.groups()
        key = (kind, int(order_str))
        n_points = int(n_str)
        if key not in index or n_points < index[key][0]:
            index[key] = (n_points, fname)
    return index


def get_spherical_design(
    t: int, antipodal: bool = False, use_float64: bool = False
) -> np.ndarray:
    """Load a spherical t-design and return it as a numpy array.

    The spherical t-designs were downloaded from
    https://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/index.html and follow
    the naming convention `<kind><TTT>.<NNNNN>`:

    - `ss<TTT>.<NNNNN>`: antipodal design of order `TTT` with `NNNNN`
      points (e.g. `ss003.00006`).
    - `sf<TTT>.<NNNNN>`: not necessarily antipodal design of order `TTT`
      with `NNNNN` points (e.g. `sf003.00008`).

    If no design is available for the requested order, the next available
    order (`>= t`) is used instead.

    Args:
        t: Required order of the spherical design.
        antipodal: If `True`, only antipodal designs (`ss` files) are
            considered. If `False` (default), both `ss` and `sf` files
            are considered and the design with the fewest points is returned.
        use_float64: Whether to load as `float64` (otherwise `float32`).

    Returns:
        The spherical t-design as a numpy array of shape `(n_points, 3)`.
    """
    index = _index_designs()

    allowed_kinds = ("ss",) if antipodal else ("ss", "sf")
    candidates = [
        (n_points, fname)
        for (kind, order), (n_points, fname) in index.items()
        if kind in allowed_kinds and order >= t
    ]
    if not candidates:
        kind_desc = "antipodal " if antipodal else ""
        raise ValueError(f"No {kind_desc}spherical design available for order >= {t}")

    _, fname = min(candidates, key=lambda item: item[0])
    design_path = os.path.join(_DESIGN_DIR, fname)

    dtype = np.float64 if use_float64 else np.float32
    return np.loadtxt(design_path, dtype=dtype).reshape(-1, 3)
