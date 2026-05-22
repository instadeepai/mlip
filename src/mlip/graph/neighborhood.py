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


import matscipy.neighbours
import numpy as np

FLAT_CELL_THRESHOLD_ANGSTROM = 1e-6


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


def _safe_matscipy_neighbour_list(**kwargs):
    """Forwards call to `matscipy.neighbours.neighbour_list(**kwargs)`.

    Handles two failure modes that can arise for molecules without a cell:
    A `SIGABRT` error in C if the molecule is almost completely flat in one dimension,
    or a `LinAlgError` due to singular matrix inversion in matscipy. In either case,
    we run with a padded cell instead.
    """

    _kwargs = kwargs.copy()
    cell = _kwargs.pop("cell", None)
    pbc = _kwargs.pop("pbc", None)
    positions = _kwargs.pop("positions")
    cutoff = _kwargs.pop("cutoff")

    def _run_with_padded_cell():
        padded_cell, cell_origin = get_no_pbc_cell(positions, cutoff)
        return matscipy.neighbours.neighbour_list(
            **_kwargs,
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            cell=padded_cell,
            cell_origin=cell_origin,
        )

    if not any(pbc) and cell is None:
        # Catch edge-case of almost-flat molecules that can cause a SIGABRT error.
        ranges = np.max(positions, axis=0) - np.min(positions, axis=0)
        if np.any(ranges < FLAT_CELL_THRESHOLD_ANGSTROM):
            return _run_with_padded_cell()

    try:
        return matscipy.neighbours.neighbour_list(**kwargs)
    except np.linalg.LinAlgError:
        if cell is not None or any(pbc):
            raise ValueError(
                "Neighbour list creation with matscipy failed due to "
                "singular matrix inversion."
            )
        return _run_with_padded_cell()


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: tuple[bool, bool, bool] | None = None,
    cell: np.ndarray | None = None,  # [3, 3]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the edge information for a given set of positions, including senders,
    receivers, and shift vectors.

    If `pbc` is `None` or `(False, False, False)`, then the shifts will be
    returned as zero.
    This is the default behavior. The cell is None as default and as a result, matscipy
    will compute the minimal cell size needed to fit the whole system. See matscipy's
    documentation for more information.

    Args:
        positions: The position matrix.
        cutoff: The distance cutoff for the edges in Angstrom.
        pbc: A tuple of bools representing if periodic boundary conditions exist in
             any of the spatial dimensions. Default is None, which means False in every
             direction.
        cell: The unit cell of the system given as a 3x3 matrix or as None (default),
              which means that matscipy will compute the minimal cell size needed to
              fit the whole system.

    Returns:
        A tuple of **senders** (starting indexes of atoms for each edge), **receivers**
        (ending indexes of atoms for each edge), and **shifts** (the shift vectors, see
        matscipy's documentation for more information. If PBCs are false,
        then we return shifts of zero).

    """
    if pbc is None:
        pbc = (False, False, False)

    if np.all(cell == 0.0):
        cell = None

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell is None or cell.shape == (3, 3)

    # See docstring of functions get_edge_relative_vectors() and
    # get_edge_vectors() on how senders and receivers are used
    senders, receivers, senders_unit_shifts = _safe_matscipy_neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
    )

    # If we are not having PBCs, then use shifts of zero
    shifts = senders_unit_shifts if any(pbc) else np.zeros((len(senders), 3))

    # See docstring of functions get_edge_relative_vectors() and
    # get_edge_vectors() on how these return values are used
    return senders, receivers, shifts
