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

import jax.numpy as jnp
import jraph
import numpy as np
import pytest
from ase import Atoms
from matscipy.neighbours import neighbour_list

from mlip.data.helpers.edge_vectors import get_edge_relative_vectors
from mlip.data.helpers.neighborhood import get_neighborhood
from mlip.simulation.utils import create_graph_from_atoms
from mlip.typing.graph_definition import GraphEdges, GraphGlobals, GraphNodes


@pytest.fixture
def graph_manually_created_with_shifts():
    """Create a graph within a PBC box, with edges crossing boundaries.

        ----------
        | 2      |
        |        |
        | 0    1 |
        ----------

    Expected edges should be of length 0.1 when subtracting the lattice
    shift vectors.
    """
    z = 0.1
    positions = jnp.array([[0.05, 0.05, z], [4.95, 0.05, z], [0.05, 4.95, z]])
    cell = 5 * jnp.eye(3)[None, :]
    species = jnp.zeros((3,))
    # Directed edges: 0->1, 1->0, 0->2, 2->0
    senders = jnp.array([0, 1, 0, 2])
    receivers = jnp.array([1, 0, 2, 0])
    shifts = jnp.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]])

    return jraph.GraphsTuple(
        nodes=GraphNodes(
            positions=positions,
            forces=jnp.zeros((3, 3)),
            species=species,
        ),
        edges=GraphEdges(
            shifts=shifts,
            displ_fun=None,
        ),
        globals=GraphGlobals(
            cell=cell,
            energy=jnp.zeros((1, 1)),
            stress=jnp.zeros((1, 3, 3)),
            weight=jnp.ones((1, 1)),
        ),
        receivers=receivers,
        senders=senders,
        n_edge=jnp.array([4]),
        n_node=jnp.array([3]),
    )


@pytest.fixture
def graph_created_from_ase_atoms():
    z = 0.1
    atoms = Atoms(
        "H3",
        positions=[[0.05, 0.05, z], [4.95, 0.05, z], [0.05, 4.95, z]],
        pbc=True,
        cell=[5, 5, 5],
    )
    senders, receivers, shifts = neighbour_list(
        quantities="ijS",
        cell=atoms.cell,
        pbc=atoms.pbc,
        positions=atoms.positions,
        cutoff=0.11,
    )

    _displacement_fun = None
    graph = create_graph_from_atoms(
        atoms,
        senders,
        receivers,
        _displacement_fun,
        allowed_atomic_numbers=[1],
        cell=atoms.cell,
        shifts=shifts,
    )
    return graph


def test_graph_with_shifts_and_graph_from_atoms_is_equal(
    graph_manually_created_with_shifts, graph_created_from_ase_atoms
):
    graph_1 = graph_manually_created_with_shifts
    graph_2 = graph_created_from_ase_atoms

    assert jnp.allclose(graph_1.nodes.positions, graph_2.nodes.positions)
    assert jnp.allclose(graph_1.globals.cell, graph_2.globals.cell)
    assert jnp.allclose(graph_1.n_edge, graph_2.n_edge)
    assert jnp.allclose(graph_1.n_node, graph_2.n_node)
    sorted_indices_shifts = jnp.lexsort(graph_1.edges.shifts.T)
    sorted_indices_atoms = jnp.lexsort(graph_2.edges.shifts.T)

    assert jnp.allclose(
        graph_1.edges.shifts[sorted_indices_shifts],
        graph_2.edges.shifts[sorted_indices_atoms],
    )
    assert jnp.allclose(
        graph_1.senders[sorted_indices_shifts],
        graph_2.senders[sorted_indices_atoms],
    )
    assert jnp.allclose(
        graph_1.receivers[sorted_indices_shifts],
        graph_2.receivers[sorted_indices_atoms],
    )


def test_edge_relative_vectors_with_shifts(graph_manually_created_with_shifts):
    graph = graph_manually_created_with_shifts
    expect = jnp.array(
        [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.1, 0.0]]
    )
    result = get_edge_relative_vectors(
        graph.nodes.positions,
        graph.senders,
        graph.receivers,
        graph.edges.shifts,
        graph.globals.cell,
        graph.n_edge,
    )
    assert jnp.allclose(expect, result)


def test_matscipy_linalg_error_is_handled_automatically_in_pbc_false_case():
    # With these positions and the default cell that matscipy computes, there would
    # be a linear algebra error from numpy if we wouldn't explicitly handle it.
    positions = np.array([[1.0, 1.5, 0.0], [-1.0, 1.5, 0.0]])

    senders, receivers, shifts = get_neighborhood(positions, cutoff=5.0)

    assert senders.tolist() == [1, 0]
    assert receivers.tolist() == [0, 1]
    assert shifts.tolist() == [[0, 0, 0], [0, 0, 0]]
