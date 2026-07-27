# Copyright 2026 Zhongguancun Academy
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

import bisect
import os
from typing import SupportsIndex

import numpy as np
from ase.db import connect
from ase.db.core import Database

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)
from mlip.data.chemical_systems_readers.chemical_systems_dataset import (
    ChemicalSystemsDataset,
)
from mlip.data.chemical_systems_readers.type_aliases import (
    ChemicalSystems,
)


class AselmdbReader(ChemicalSystemsReader, ChemicalSystemsDataset):
    """Implementation of a chemical systems reader that loads data from
    aselmdb format via the `ase` library."""

    def __init__(self, *args, **kwargs):
        # Initialize before super()/validation so __del__ is safe on failed construct.
        self._dbs: list[Database] | None = None
        self._db_ids: list[list[int]] | None = None
        self._idlen_cumulative: list[int] | None = None
        self._length: int | None = None
        super().__init__(*args, **kwargs)

    def _ensure_index(self) -> None:
        if self._dbs is None:
            self._dbs = [
                connect(path, type="aselmdb", readonly=True, use_lock_file=False)
                for path in self._paths
            ]

        if self._db_ids is not None:
            return

        db_ids: list[list[int]] = []
        for db in self._dbs:
            if hasattr(db, "ids"):
                ids = list(db.ids)
            else:
                ids = [int(row.id) for row in db.select()]
            if self.num_to_load is not None:
                ids = ids[: self.num_to_load]
            db_ids.append(ids)

        id_lens = [len(ids) for ids in db_ids]
        self._db_ids = db_ids
        self._idlen_cumulative = np.cumsum(id_lens).tolist() if id_lens else []
        self._length = int(sum(id_lens))

    def load(self) -> ChemicalSystems:
        """Load chemical systems from all aselmdb filepaths."""
        filepaths = self.filepaths
        if not isinstance(filepaths, list):
            filepaths = [filepaths]

        all_systems: ChemicalSystems = []
        for filepath in filepaths:
            all_systems.extend(self._load_single_file(filepath))
        return all_systems

    def __len__(self) -> int:
        self._ensure_index()
        return self._length

    def __getitem__(self, index: SupportsIndex) -> ChemicalSystem:
        self._ensure_index()

        idx = int(index)
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)

        db_idx = bisect.bisect(self._idlen_cumulative, idx)
        el_idx = idx if db_idx == 0 else idx - self._idlen_cumulative[db_idx - 1]

        atoms_row = self._dbs[db_idx]._get_row(self._db_ids[db_idx][el_idx])
        atoms = atoms_row.toatoms()
        if isinstance(atoms_row.data, dict):
            atoms.info.update(atoms_row.data)

        return ChemicalSystem.from_ase_atoms(
            atoms, property_name_mapping=self.property_name_mapping
        )

    def __getstate__(self):
        # DB connections are not picklable; reopen lazily after unpickle.
        state = self.__dict__.copy()
        state["_dbs"] = None
        return state

    def __del__(self):
        if self._dbs is None:
            return
        for db in self._dbs:
            if hasattr(db, "close"):
                db.close()
        self._dbs = None

    def _read_file(self, filepath: str | os.PathLike) -> ChemicalSystems:
        """Read a single aselmdb file and convert to chemical systems."""
        with connect(
            filepath, type="aselmdb", use_lock_file=False, readonly=True
        ) as db:
            systems = []
            for idx, row in enumerate(db.select()):
                atoms = row.toatoms()
                if isinstance(row.data, dict):
                    atoms.info.update(row.data)
                systems.append(
                    ChemicalSystem.from_ase_atoms(
                        atoms, property_name_mapping=self.property_name_mapping
                    )
                )
                if self.num_to_load is not None and idx >= self.num_to_load - 1:
                    break
        return systems
