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

from typing import Callable

import ase
import jax
import jraph
import numpy as np

from mlip.data.helpers.atomic_number_table import AtomicNumberTable
from mlip.typing.graph_definition import (
    GraphEdges,
    GraphGlobals,
    GraphNodes,
    ShiftVectors,
)


def create_graph_from_atoms(
    atoms: ase.Atoms,
    senders: np.ndarray,
    receivers: np.ndarray,
    displacement_fun: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    allowed_atomic_numbers: set[int],
    cell: np.ndarray | None = None,
    shifts: ShiftVectors | None = None,
) -> jraph.GraphsTuple:
    """Creates a graph for a group of atoms (i.e., a chemical system).

    This function will leave the shifts of the graph empty and will populate the
    displacement function in the graph object instead.

    Note: Only one of the two arguments "shifts" and "displacement_fun" should be passed
    to this function, the other one should be None.

    Args:
        atoms: The atoms of the system.
        senders: The sender indexes of the edges for the graph.
        receivers: The receiver indexes of the edges for the graph.
        displacement_fun: Optional function that takes in two position vectors and
                          returns the displacement vector between them.
        allowed_atomic_numbers: A set of allowed atomic numbers known to a model
                                such that the atomic species can be correctly assigned
                                in the graph.
        cell: The structure's box.
        shifts: Vectors defining which periodic box each node is in, so that one can
                compute the edge vectors from the positions.

    Returns:
        The graph representing the system.
    """
    if shifts is None:
        if displacement_fun is None:
            raise ValueError(
                "Both shifts and displacement_fun are None when creating graph."
                " One and only one should be passed."
            )
        vmapped_displ_fun = jax.tree_util.Partial(jax.vmap(displacement_fun))
    elif displacement_fun is None:
        vmapped_displ_fun = None
    else:
        raise ValueError(
            "Both shifts and displacement_fun are not None when creating"
            " graph. One and only one should be passed."
        )

    positions = atoms.get_positions()
    if cell is None:
        cell = np.identity(3, dtype=float)
        assert shifts is None or np.all(shifts == 0)

    z_table = AtomicNumberTable(sorted(allowed_atomic_numbers))
    z_map = z_table.z_to_index_map(max_atomic_number=120)
    atomic_species = z_map[atoms.numbers]

    return jraph.GraphsTuple(
        nodes=GraphNodes(
            positions=positions,
            forces=None,
            species=atomic_species,
        ),
        edges=GraphEdges(shifts=shifts, displ_fun=vmapped_displ_fun),
        globals=jax.tree.map(
            lambda x: x[None, ...],
            GraphGlobals(
                cell=cell,
                energy=np.array(0.0),
                stress=None,
                weight=np.asarray(1.0),
            ),
        ),
        receivers=receivers,
        senders=senders,
        n_edge=np.array([len(senders)]),
        n_node=np.array([len(atomic_species)]),
    )
