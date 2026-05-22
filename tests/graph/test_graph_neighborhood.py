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

from unittest.mock import patch

import jax.numpy as jnp
import matscipy.neighbours
import numpy as np
import pytest
from ase import Atoms
from matscipy.neighbours import neighbour_list

from mlip.data.chemical_system import ChemicalSystem
from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.graph.neighborhood import get_neighborhood, get_no_pbc_cell


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
    # Directed edges: 0->1, 1->0, 0->2, 2->0
    senders = jnp.array([0, 1, 0, 2])
    receivers = jnp.array([1, 0, 2, 0])
    shifts = jnp.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]])

    return Graph(
        nodes=GraphNodes(
            positions=positions,
            forces=jnp.zeros((3, 3)),
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

    chem_system = ChemicalSystem.from_ase_atoms(atoms)
    return Graph.from_chemical_system(chem_system, graph_cutoff_angstrom=0.11)


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


def test_flat_molecule_uses_padded_cell_in_pbc_false_case():
    # Flat positions (z=0) triggers using get_no_pbc_cell before matscipy call.
    positions = np.array([[1.0, 1.5, 0.0], [-1.0, 1.5, 0.0]])

    with patch(
        "mlip.graph.neighborhood.get_no_pbc_cell", wraps=get_no_pbc_cell
    ) as mock_get_cell:
        senders, receivers, shifts = get_neighborhood(positions, cutoff=5.0)

    mock_get_cell.assert_called_once()
    assert senders.tolist() == [0, 1]
    assert receivers.tolist() == [1, 0]
    assert shifts.tolist() == [[0, 0, 0], [0, 0, 0]]


def test_matscipy_llinalg_error_is_handled_automatically_in_pbc_false_case(monkeypatch):
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    original = matscipy.neighbours.neighbour_list
    call_count = 0

    def mock_neighbour_list(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise np.linalg.LinAlgError("Singular matrix")
        return original(**kwargs)

    monkeypatch.setattr(matscipy.neighbours, "neighbour_list", mock_neighbour_list)

    senders, _, shifts = get_neighborhood(positions, cutoff=5.0)

    assert call_count == 2
    assert len(senders) > 0
    assert np.all(shifts == 0.0)


def test_no_pbc_graph_does_not_have_shifts(setup_system, mace_force_field):
    # With these positions and the default cell that matscipy computes, there would
    # be a linear algebra error from numpy if we wouldn't explicitly handle it.
    (
        atoms,
        _,
    ) = setup_system
    model_ff = mace_force_field
    graph_cutoff_angstrom = model_ff.cutoff_distance
    positions = atoms.positions

    senders, receivers, shifts = get_neighborhood(
        positions,
        graph_cutoff_angstrom,
        pbc=None,
        cell=None,
    )

    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs**2, axis=-1))
    expected_receivers, expected_senders = np.where(
        (0 < distances) & (distances < graph_cutoff_angstrom)
    )

    assert np.all(shifts == 0.0)
    assert len(senders) == 68

    expected_edges = []
    for s, r in zip(list(expected_senders), list(expected_receivers)):
        expected_edges.append((s, r))
    edges = []
    for s, r in zip(list(senders), list(receivers)):
        edges.append((s, r))
    assert sorted(expected_edges) == sorted(edges)


def test_no_pbc_cell_does_not_have_shifts(setup_system, mace_force_field) -> None:
    (
        atoms,
        _,
    ) = setup_system
    model_ff = mace_force_field
    graph_cutoff_angstrom = model_ff.cutoff_distance

    no_pbc_cell, no_pbc_cell_origin = get_no_pbc_cell(
        atoms.positions, graph_cutoff_angstrom
    )

    senders, shifts = neighbour_list(
        quantities="iS",
        cell=no_pbc_cell,
        cell_origin=no_pbc_cell_origin,
        pbc=np.array([False, False, False]),
        positions=atoms.positions,
        cutoff=graph_cutoff_angstrom,
    )
    assert np.all(shifts == 0.0)
    assert len(senders) == 68
