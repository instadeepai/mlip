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

from typing import Sequence

import numpy as np


class AtomicNumberTable:
    """The atomic number table which is handling the mappings between atomic
    species (indexes) and atomic numbers.
    """

    def __init__(self, zs: Sequence[int]):
        """Constructor.

        Args:
            zs: A sorted and deduplicated sequence of atomic numbers Z.
        """
        zs = [int(z) for z in zs]
        # unique
        assert len(zs) == len(set(zs))
        # sorted
        assert zs == sorted(zs)

        # These are all atomic numbers that exist in the dataset
        self.zs = zs
        # We create a map to map the atomic number to the index in the zs list
        self.z_map = {z: i for i, z in enumerate(zs)}
        self.reverse_z_map = dict(enumerate(zs))

    def __len__(self) -> int:
        """Return the number of elements in the table."""
        return len(self.zs)

    def __str__(self):
        """Return a string representation of the table."""
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        """Maps an index (i.e., a value of atomic species) to an atomic number Z.

        Args:
            index: The index to map to the atomic number.

        Returns:
            The atomic number Z.
        """
        return self.reverse_z_map[index]

    def z_to_index(self, atomic_number: int) -> int:
        """Maps an atomic number Z to an index (i.e., a value of atomic species).

        Args:
            atomic_number: The atomic number Z.

        Returns:
            The index.
        """
        return self.z_map[atomic_number]

    def z_to_index_map(self, max_atomic_number: int) -> np.ndarray:
        """Returns a Z-to-index map that can be used by multiple numbers at once.

        Args:
            max_atomic_number: The size of the resulting map array, i.e.,
                               the maximum atomic number to consider.

        Returns:
            The map which is an array where the array index is the atomic number and
            the value at that index is the atomic index.

        """
        x = np.zeros(max_atomic_number + 1, dtype=np.int32)
        for i, z in enumerate(self.zs):
            x[z] = i
        return x
