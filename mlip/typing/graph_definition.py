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

from typing import Callable, NamedTuple, Optional, TypeAlias

import numpy as np

Positions: TypeAlias = np.ndarray  # [num_nodes, 3]
DisplacementVectors: TypeAlias = np.ndarray  # [num_edges, 3]
ShiftVectors: TypeAlias = np.ndarray  # [num_edges, 3]
AtomicSpecies: TypeAlias = np.ndarray  # [num_nodes]
Forces: TypeAlias = np.ndarray  # [num_nodes, 3]
Cell: TypeAlias = np.ndarray  # [num_graphs, 3, 3]
Stress: TypeAlias = np.ndarray  # [num_graphs, 3, 3]
Energy: TypeAlias = np.ndarray  # [num_graphs]
WeightingFactors: TypeAlias = np.ndarray  # [num_graphs]


class GraphNodes(NamedTuple):
    """Contains positions and forces, which can be ``None`` if there are no reference
    forces to store, for instance in an MD. Furthermore, contains the atomic species
    which are the elements mapped to an index according to which elements
    types are allowed. For example, if the model allows for "H", "C", and "S" as
    elements, the species will be either 0 for "H", 1 for "C", and 2 for "S".
    """

    positions: Positions
    species: AtomicSpecies
    forces: Optional[Forces] = None


class GraphEdges(NamedTuple):
    """Contains either the shift vectors or a displacement function. This info is in
    each case used to compute the edge vectors from the positions. If shifts are not
    None, the models will use them, otherwise check for a displacement function.

    The displacement function should be vmapped already, meaning it can take in a
    position matrix for senders and receivers and output the edge vector matrix.
    Moreover, the displacement function must be wrapped in ``jax.tree_util.Partial`` in
    order to be compatible with jitting.

    If the displacement function pathway is applied, stress cannot be calculated as a
    property. In the future, these two pathways may be unified.
    """

    shifts: Optional[ShiftVectors]
    displ_fun: Optional[Callable[[Positions, Positions], DisplacementVectors]] = None


class GraphGlobals(NamedTuple):
    """Contains energy and stress that are ``None`` if no reference exists, for example
    during MD. Furthermore, this holds the cell, which is the unit cell (i.e., box)
    for the system which is relevant if periodic boundary conditions are applied
    or for constant pressure simulations. Lastly, the weight vector contains
    weights for each subgraph in a graph batch that should be considered in
    the loss, for example, zero for dummy graphs in a batch.
    """

    cell: Cell
    weight: WeightingFactors
    energy: Optional[Energy] = None
    stress: Optional[Stress] = None
