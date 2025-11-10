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

from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np


def get_edge_vectors(
    positions: np.ndarray,  # [n_nodes, 3]
    senders: np.ndarray,  # [n_edges]
    receivers: np.ndarray,  # [n_edges]
    shifts: np.ndarray,  # [n_edges, 3]
    cell: Optional[np.ndarray],  # [n_graph, 3, 3]
    n_edge: np.ndarray,  # [n_graph]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute positions of sender and receiver nodes of each edge.

    With periodic boundary conditions (PBCs), the receiver position
    will remain unchanged and stay in the unit cell. The sender node's
    representative is translated from the unit cell to the nearest
    neighbouring cell, by subtracting the shift (an integer-valued
    vector counting lattice steps) multiplied by the 3x3 cell matrix.


    .. code-block:: python

        # Returns (vectors_senders, vectors_receivers)
        vectors_senders   = positions[i] - shifts @ cell
        vectors_receivers = positions[j]

    The shift vectors therefore describe the number of boundary crossings
    of the directed edge going from the sender to the receiver.


    Args:
        positions: The positions of the nodes.
        senders: The sender nodes of each edge.
                 Output `i` of `ase.neighborlist.primitive_neighbor_list`.
        receivers: The receiver nodes of each edge.
                   Output `j` of `ase.neighborlist.primitive_neighbor_list`.
        shifts: The shift vectors of each edge.
                Output `S` of `ase.neighborlist.primitive_neighbor_list`.
        cell: The cell of each graph. Array of shape ``[n_graph, 3, 3]``.
        n_edge: The number of edges of each graph. Array of shape ``[n_graph]``.

    Returns:
        The positions of the sender and receiver nodes of each edge.
    """
    vectors_senders = positions[senders]  # [n_edges, 3]
    vectors_receivers = positions[receivers]  # [n_edges, 3]

    if cell is not None:
        num_edges = receivers.shape[0]
        shifts = jnp.einsum(
            "ei,eij->ej",
            shifts,  # [n_edges, 3]
            jnp.repeat(
                cell,  # [n_graph, 3, 3]
                n_edge,  # [n_graph]
                axis=0,
                total_repeat_length=num_edges,
            ),  # [n_edges, 3, 3]
        )  # [n_edges, 3]
        vectors_senders -= shifts  # minus sign to match results with ASE

    return vectors_senders, vectors_receivers  # [n_edges, 3]


def get_edge_relative_vectors(
    positions: np.ndarray,  # [n_nodes, 3]
    senders: np.ndarray,  # [n_edges]
    receivers: np.ndarray,  # [n_edges]
    shifts: np.ndarray,  # [n_edges, 3]
    cell: Optional[np.ndarray],  # [n_graph, 3, 3]
    n_edge: np.ndarray,  # [n_graph]
) -> np.ndarray:
    """Compute the relative edge vectors from senders to receivers.

    With PBCs, sender nodes need to be translated from the unit cell
    to the receiver's nearest neighbouring cell. See :func:`get_edge_vectors`
    for more details.

    .. code-block:: python

        # Returns vectors
        vectors = positions[receivers] - positions[senders] + shifts @ cell
        # From the ASE docs:
        D = positions[j] - positions[i] + S.dot(cell)

    Args:
        positions: The positions of the system.
        senders: The sender indices of the edges, labelled `i` by ASE.
        receivers: The receiver indices of the edges, labelled `j` by ASE.
        shifts: The shift vectors as returned by the matscipy neighbour lists
                functionality, and labelled `S` by ASE.
        cell: The unit cells of each graph, an array of shape `[n_graph, 3, 3]`.
        n_edge: The number of edges for each graph, an array of shape `[n_graph]`.

    Returns:
        The relative edge vectors, labelled `D` by ASE.
    """
    vectors_senders, vectors_receivers = get_edge_vectors(
        positions=positions,
        senders=senders,
        receivers=receivers,
        shifts=shifts,
        cell=cell,
        n_edge=n_edge,
    )
    return vectors_receivers - vectors_senders
