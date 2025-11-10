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

import numpy as np


def get_no_pbc_cell(
    positions: np.ndarray, graph_cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Create a cell that contains all positions, with room to spare.

    Args:
        positions: A Nx3 array of the positions of the atoms in Angstrom.
        graph_cutoff: The maximum distance for an edge to be computed between two atoms
                      in Angstrom.

    Returns:
        A tuple of the cell, as an array of size 3,
        and a cell origin, as an array of size 3.
    """
    rmax = np.max(positions, axis=0)
    rmin = np.min(positions, axis=0)
    return np.diag(graph_cutoff * 4 + (rmax - rmin)), rmin - graph_cutoff * 2
