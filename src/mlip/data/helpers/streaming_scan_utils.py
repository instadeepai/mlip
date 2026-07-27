# Copyright 2025 Zhongguancun Academy
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

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TypeVar, Sequence

import numpy as np
import pydantic
from pydantic import BaseModel

from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
    readers_metadata,
)

logger = logging.getLogger("mlip")

T = TypeVar("T", bound=BaseModel)


def count_dynamic_batches(
    n_nodes: Sequence[int],
    n_edges: Sequence[int],
    n_edges_lr: Sequence[int],
    *,
    n_node: int,
    n_edge: int,
    n_graph: int,
    n_edge_long_range: int | None,
    skip_last_batch: bool,
) -> int:
    """Count the number of batches with dynamic batching."""
    valid_lr = n_edge_long_range if n_edge_long_range is not None else 0
    num_batches = 0
    acc_n = acc_e = acc_g = acc_lr = 0
    has_batch = False
    for nn, ne, nlr in zip(n_nodes, n_edges, n_edges_lr):
        if not has_batch:
            acc_n, acc_e, acc_g, acc_lr = nn, ne, 1, nlr
            has_batch = True
            continue
        if (
            acc_g + 1 > n_graph - 1
            or acc_n + nn > n_node - 1
            or acc_e + ne > n_edge
            or acc_lr + nlr > valid_lr
        ):
            num_batches += 1
            acc_n, acc_e, acc_g, acc_lr = nn, ne, 1, nlr
        else:
            acc_n += nn
            acc_e += ne
            acc_g += 1
            acc_lr += nlr
    if has_batch and not skip_last_batch:
        num_batches += 1
    return num_batches


class Histogram:
    """Online histogram for median / mean of non-negative integer samples."""

    def __init__(self, init_bins: int = 512):
        self.size = init_bins
        self.hist = np.zeros(self.size, dtype=np.int64)
        self.total_count = 0

    def _ensure_size(self, new_max_value: int):
        if new_max_value < self.size:
            return
        new_size = self.size
        while new_size <= new_max_value:
            new_size *= 2
        new_hist = np.zeros(new_size, dtype=np.int64)
        new_hist[: self.size] = self.hist
        self.hist = new_hist
        self.size = new_size

    def add(self, values: np.ndarray):
        """Add a batch of values to the histogram."""
        if values.size == 0:
            return
        max_v = int(values.max())
        self._ensure_size(max_v)
        cnt = np.bincount(values.astype(np.int64, copy=False), minlength=self.size)
        self.hist[: len(cnt)] += cnt
        self.total_count += int(values.size)

    def median(self) -> int:
        """Compute the median of the samples."""
        if self.total_count == 0:
            return 0
        midpoint = (self.total_count - 1) // 2
        cumulative = 0
        for idx, count in enumerate(self.hist):
            cumulative += int(count)
            if cumulative > midpoint:
                return int(idx)
        return int(len(self.hist) - 1)

    def mean(self) -> float:
        """Compute the mean of the samples."""
        if self.total_count == 0:
            return 0.0
        coeffs = np.arange(len(self.hist), dtype=np.float64)
        return float(np.dot(self.hist, coeffs) / self.total_count)


class IncrementalE0s:
    """Online least-squares accumulator for per-element reference energies (E0s).

    Each graph contributes composition counts and total energy to the normal
    equations ``AᵀA`` / ``Aᵀb``. New atomic numbers grow the matrices in place;
    memory is ``O(n_species²)``.
    """

    def __init__(self):
        self._z_to_i: dict[int, int] = {}
        self._zs: list[int] = []
        self._ata = None
        self._atb = None

    def _ensure_species(self, zs: np.ndarray):
        new = [int(z) for z in zs if int(z) not in self._z_to_i]
        if not new:
            return
        old_n = len(self._zs)
        for z in new:
            self._z_to_i[z] = len(self._zs)
            self._zs.append(z)
        n = len(self._zs)
        ata = np.zeros((n, n), dtype=np.float64)
        atb = np.zeros(n, dtype=np.float64)
        if old_n:
            ata[:old_n, :old_n] = self._ata
            atb[:old_n] = self._atb
        self._ata = ata
        self._atb = atb

    def add(self, atomic_numbers: np.ndarray, energy: float):
        """Accumulate one graph's composition and total energy."""
        zs, counts = np.unique(np.asarray(atomic_numbers), return_counts=True)
        self._ensure_species(zs)
        c = np.zeros(len(self._zs), dtype=np.float64)
        for z, cnt in zip(zs, counts):
            c[self._z_to_i[int(z)]] = float(cnt)
        self._ata += np.outer(c, c)
        self._atb += c * energy

    def solve(self) -> dict[int, float]:
        """Compute the per-element reference energies (E0s)."""
        if not self._zs:
            return {}
        try:
            e0s = np.linalg.lstsq(self._ata, self._atb, rcond=1e-8)[0]
            by_z = {z: float(e0s[i]) for i, z in enumerate(self._zs)}
            return dict(sorted(by_z.items()))
        except np.linalg.LinAlgError:
            logger.warning(
                "Failed to compute E0s using least squares regression, "
                "using the 0.0 for all atoms."
            )
            return dict.fromkeys(sorted(self._zs), 0.0)


class CachingHelper:
    """Helper for caching BatchingInfo."""

    def __init__(
        self,
        cache_dir: str | os.PathLike | None,
        readers: list[ChemicalSystemsReader] | ChemicalSystemsReader,
        batch_size: int,
        graph_cutoff_angstrom: float,
        long_range_cutoff_angstrom: float | None,
        max_n_node: int | None,
        max_n_edge: int | None,
        max_n_edge_long_range: int | None,
    ):
        if cache_dir is None:
            self.cache_dir = None
            return
        self.cache_dir = Path(cache_dir)

        payload = {
            "readers_metadata": readers_metadata(readers),
            "batch_size": batch_size,
            "graph_cutoff_angstrom": graph_cutoff_angstrom,
            "long_range_cutoff_angstrom": long_range_cutoff_angstrom,
            "max_n_node": max_n_node,
            "max_n_edge": max_n_edge,
            "max_n_edge_long_range": max_n_edge_long_range,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        fingerprint = hashlib.sha256(blob).hexdigest()[:16]
        self.cache_path = self.cache_dir / f"batching_info_{fingerprint}.json"

    def load(self, model_cls: type[T]) -> T | None:
        """Load cached model from ``cache_dir/batching_info_*.json``."""
        if self.cache_dir is None:
            return None
        path = self.cache_path
        if not path.is_file():
            return None
        try:
            info = model_cls.model_validate_json(path.read_text())
            logger.info("Reusing cached BatchingInfo from %s", path)
            return info
        except (OSError, pydantic.ValidationError) as exc:
            logger.warning("Failed to load BatchingInfo cache %s: %s", path, exc)
            return None

    def save(self, info: BaseModel) -> None:
        """Save model to ``cache_dir/batching_info_*.json``."""
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(info.model_dump_json(indent=4))
        logger.info("Wrote BatchingInfo cache to %s", self.cache_path)
