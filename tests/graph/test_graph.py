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

import dataclasses
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms

from mlip.data.chemical_system import ChemicalSystem
from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes
from mlip.graph.batching_helpers import batch_graphs, pad_with_graphs
from mlip.models.charge_utils import compute_long_range_interactions

WITH_SHIFTS_DISTANCE_CUTOFF = 0.11
WITH_SHIFTS_ALLOWED_ATOMIC_NUMBERS = [1]
WITH_SHIFTS_WEIGHT = 1.0
WITH_SHIFTS_LONG_RANGE_CUTOFF = 5.5


def _sort_edges_in_graph(graph: Graph) -> Graph:
    sorted_indices = jnp.lexsort(graph.edges.shifts.T)
    graph = graph.replace_edges(shifts=graph.edges.shifts[sorted_indices])
    return graph.replace(
        senders=graph.senders[sorted_indices],
        receivers=graph.receivers[sorted_indices],
    )


def _batch_graphs_with_padding(graphs: Iterator[Graph]) -> Graph:
    batched_graph = batch_graphs(graphs)
    padded_batched_graph = pad_with_graphs(
        batched_graph,
        n_node=jnp.sum(batched_graph.n_node) + 1,
        n_edge=jnp.sum(batched_graph.n_edge) + 1,
        n_graph=batched_graph.num_graphs + 1,
    )
    return padded_batched_graph


@pytest.fixture
def atoms_with_shifts():
    z = 0.1
    atoms = Atoms(
        "H3",
        positions=[[0.05, 0.05, z], [4.95, 0.05, z], [0.05, 4.95, z]],
        pbc=True,
        cell=[5, 5, 5],
    )
    return atoms


@pytest.fixture
def graph_with_shifts(atoms_with_shifts):
    """Create a graph within a PBC box, with edges crossing boundaries.

        ----------
        | 2      |
        |        |
        | 0    1 |
        ----------

    Expected edges should be of length 0.1 when subtracting the lattice
    shift vectors.
    """
    positions = jnp.array(atoms_with_shifts.positions)
    cell = jnp.array(atoms_with_shifts.cell.array)[None, :]
    numbers = jnp.array(atoms_with_shifts.numbers)

    # Directed edges: 0->1, 1->0, 0->2, 2->0
    senders = jnp.array([0, 1, 0, 2])
    receivers = jnp.array([1, 0, 2, 0])
    shifts = jnp.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]])

    graph = Graph(
        nodes=GraphNodes(
            positions=positions,
            atomic_numbers=numbers,
        ),
        edges=GraphEdges(
            shifts=shifts,
            displ_fun=None,
        ),
        globals=GraphGlobals(
            cell=cell,
            weight=np.array([WITH_SHIFTS_WEIGHT]),
            energy=np.array([0.0]),
        ),
        receivers=receivers,
        senders=senders,
        n_edge=jnp.array([4]),
        n_node=jnp.array([3]),
    )
    return _sort_edges_in_graph(graph)


@pytest.fixture
def graph_with_displ_fun(graph_with_shifts):
    shifts = graph_with_shifts.edges.shifts

    def displ_fun(vectors_senders, vectors_receivers, cell):
        edge_vectors = vectors_receivers - vectors_senders + shifts @ cell
        return edge_vectors

    return graph_with_shifts.replace_edges(shifts=None, displ_fun=displ_fun)


@pytest.fixture
def padded_graph_with_shifts(graph_with_shifts):
    # Pad the graph_with_shifts to 3 graphs, 10 nodes, 10 edges
    padded_graph = pad_with_graphs(graph_with_shifts, n_node=10, n_edge=10, n_graph=3)
    return padded_graph


@pytest.fixture
def chemical_system_with_shifts(atoms_with_shifts):
    return ChemicalSystem(
        positions=atoms_with_shifts.positions,
        atomic_numbers=atoms_with_shifts.numbers,
        cell=np.array(atoms_with_shifts.cell),
        pbc=atoms_with_shifts.pbc,
        weight=WITH_SHIFTS_WEIGHT,
    )


def test_graph_init_from_chemical_system(
    chemical_system_with_shifts, graph_with_shifts
):
    graph_from_chemical_system = Graph.from_chemical_system(
        chemical_system_with_shifts,
        graph_cutoff_angstrom=WITH_SHIFTS_DISTANCE_CUTOFF,
    )
    graph_from_chemical_system = _sort_edges_in_graph(graph_from_chemical_system)
    # Assert both graphs have the same field values
    jax.tree.map(
        np.testing.assert_allclose, graph_from_chemical_system, graph_with_shifts
    )


def test_node_mask(graph_with_shifts, padded_graph_with_shifts):
    # Test usage on non-padded graph
    mask = graph_with_shifts.node_mask()
    assert mask.shape[0] == np.sum(graph_with_shifts.n_node)
    assert np.all(mask)

    # Test usage on padded graph
    mask = padded_graph_with_shifts.node_mask()
    assert mask.shape[0] == np.sum(padded_graph_with_shifts.n_node)
    # Assert that non-padding nodes are True, and padding nodes are False
    assert np.all(mask[: np.sum(graph_with_shifts.n_node)])
    assert not np.any(mask[np.sum(graph_with_shifts.n_node) :])

    # Test usage on padded batched graphs
    batched_graph = _batch_graphs_with_padding([graph_with_shifts, graph_with_shifts])
    mask = batched_graph.node_mask()
    assert mask.shape[0] == np.sum(batched_graph.n_node)
    assert np.all(mask[: np.sum(graph_with_shifts.n_node) * 2])
    assert not np.any(mask[np.sum(graph_with_shifts.n_node) * 2 :])


def test_padded_graph_mask(graph_with_shifts, padded_graph_with_shifts):
    # Test usage on non-padded graph
    mask = graph_with_shifts.graph_mask()
    assert mask.shape[0] == graph_with_shifts.num_graphs
    assert np.all(mask)

    # Test usage on padded graph
    mask = padded_graph_with_shifts.graph_mask()
    # Assert that non-padding graphs are True, and padding graphs are False
    assert np.all(mask[: graph_with_shifts.num_graphs])
    assert not np.any(mask[graph_with_shifts.num_graphs :])

    # Test usage on padded batched graphs
    batched_graph = _batch_graphs_with_padding([graph_with_shifts, graph_with_shifts])
    mask = batched_graph.graph_mask()
    assert mask.shape[0] == batched_graph.num_graphs
    assert np.all(mask[: graph_with_shifts.num_graphs * 2])
    assert not np.any(mask[graph_with_shifts.num_graphs * 2 :])


def test_graph_edge_vectors(
    graph_with_shifts: Graph,
    graph_with_displ_fun: Graph,
):
    expect = jnp.array([
        [0.0, -0.1, 0.0],
        [-0.1, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
    ])
    result = graph_with_shifts.edge_vectors()
    assert jnp.allclose(expect, result)

    result = graph_with_displ_fun.edge_vectors()
    assert jnp.allclose(expect, result)


def test_replace_methods(graph_with_shifts: Graph):
    new_nodes = GraphNodes(
        positions=jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        forces=np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]),
    )
    new_edges = GraphEdges(
        shifts=jnp.array([[0.1, 0.0], [0.0, 0.1]]),
    )
    new_globals = GraphGlobals(
        cell=jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        energy=jnp.array([0.2]),
        stress=jnp.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]]),
        weight=jnp.array([1.0]),
    )
    new_graph = graph_with_shifts.replace_nodes(**dataclasses.asdict(new_nodes))
    new_graph = new_graph.replace_edges(**dataclasses.asdict(new_edges))
    new_graph = new_graph.replace_globals(**dataclasses.asdict(new_globals))
    jax.tree.map(np.testing.assert_allclose, new_graph.nodes, new_nodes)
    jax.tree.map(np.testing.assert_allclose, new_graph.edges, new_edges)
    jax.tree.map(np.testing.assert_allclose, new_graph.globals, new_globals)


def test_update_feature_methods(graph_with_shifts: Graph):
    feature_init_test = {"feat_1": 7, "feat_2": 11}
    feature_update = {"feat_2": 9, "feat_3": 13}
    feature_final = {"feat_1": 7, "feat_2": 9, "feat_3": 13}
    # manually add the features to the graph:
    graph_with_shifts = graph_with_shifts.replace_nodes(features=feature_init_test)
    graph_with_shifts = graph_with_shifts.replace_edges(features=feature_init_test)
    graph_with_shifts = graph_with_shifts.replace_globals(features=feature_init_test)

    new_graph = graph_with_shifts.update_node_features(**feature_update)
    new_graph = new_graph.update_edge_features(**feature_update)
    new_graph = new_graph.update_global_features(**feature_update)

    assert new_graph.nodes.features == feature_final
    assert new_graph.edges.features == feature_final
    assert new_graph.globals.features == feature_final


def test_graph_to_prediction(graph_with_shifts: Graph):
    dummy_energy = jnp.array([0.0])
    dummy_stress = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    dummy_pressure = jnp.array([0.0])
    dummy_forces = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    graph_with_shifts = graph_with_shifts.replace_globals(
        energy=dummy_energy, stress=dummy_stress, pressure=dummy_pressure
    )
    graph_with_shifts = graph_with_shifts.replace_nodes(forces=dummy_forces)
    prediction = graph_with_shifts.to_prediction()
    assert np.allclose(prediction.energy, dummy_energy)
    assert np.allclose(prediction.forces, dummy_forces)
    assert np.allclose(prediction.stress, dummy_stress)
    assert np.allclose(prediction.pressure, dummy_pressure)


def test_graph_edges_long_range_default():
    """An unspecified edges_long_range defaults to None."""
    graph = Graph(
        nodes=GraphNodes(
            positions=jnp.zeros((1, 3)),
            atomic_numbers=jnp.array([1]),
        ),
        edges=GraphEdges(shifts=jnp.zeros((0, 3)), displ_fun=None),
        globals=GraphGlobals(
            cell=jnp.zeros((1, 3, 3)),
            weight=jnp.array([1.0]),
            energy=jnp.array([0.0]),
        ),
        senders=jnp.zeros(0, dtype=jnp.int32),
        receivers=jnp.zeros(0, dtype=jnp.int32),
        n_edge=jnp.array([0]),
        n_node=jnp.array([1]),
    )

    assert graph.edges_long_range is None


@pytest.fixture
def graph_with_long_range_shifts(graph_with_shifts: Graph) -> Graph:
    """Reuse `graph_with_shifts` and attach long-range neighbours with PBC shifts.

    Long-range neighbours mirror the short-range ones for simplicity:
    same senders, receivers, and shift vectors as the short-range edges.
    The shifts are taken directly from the already-sorted edges so that
    senders[i] / receivers[i] / shifts[i] remain consistently aligned.
    """
    return graph_with_shifts.replace(
        senders_long_range=graph_with_shifts.senders,
        receivers_long_range=graph_with_shifts.receivers,
        n_edge_long_range=graph_with_shifts.n_edge,
        edges_long_range=GraphEdges(
            shifts=graph_with_shifts.edges.shifts, displ_fun=None
        ),
    )


def test_graph_long_range_edge_vectors(graph_with_long_range_shifts: Graph):
    """long_range_edge_vectors mirrors edge_vectors using edges_long_range.shifts."""
    # Same expected vectors as test_graph_edge_vectors — the long-range
    # neighbours are identical to the short-range ones in this fixture.
    expect = jnp.array([
        [0.0, -0.1, 0.0],
        [-0.1, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
    ])
    result = graph_with_long_range_shifts.long_range_edge_vectors()
    # Vectors come back ordered by senders_long_range/receivers_long_range, same
    # ordering as the short-range edges since we mirrored them.
    assert jnp.allclose(jnp.sort(result, axis=0), jnp.sort(expect, axis=0))


def test_graph_init_from_chemical_system_with_long_range_pbc(
    chemical_system_with_shifts,
):
    """from_chemical_system passes PBC and cell to the long-range neighborhood
    builder, populating edges_long_range.shifts with non-zero entries when the
    long-range cutoff exceeds the cell extent."""
    graph = Graph.from_chemical_system(
        chemical_system_with_shifts,
        graph_cutoff_angstrom=WITH_SHIFTS_DISTANCE_CUTOFF,
        long_range_cutoff_angstrom=WITH_SHIFTS_LONG_RANGE_CUTOFF,
    )

    assert graph.n_edge_long_range is not None
    assert graph.edges_long_range.shifts is not None
    assert graph.edges_long_range.shifts.shape == (
        int(graph.n_edge_long_range[0]),
        3,
    )
    # With a 5x5x5 cell and a 5.5 Å cutoff, at least one image edge crosses a
    # boundary, so at least one shift must be non-zero.
    assert np.any(np.asarray(graph.edges_long_range.shifts) != 0)

    # long_range_edge_vectors must agree with raw position deltas + the shift term.
    cell = np.asarray(graph.globals.cell[0])
    shifts_e = np.asarray(graph.edges_long_range.shifts)
    s = np.asarray(graph.senders_long_range)
    r = np.asarray(graph.receivers_long_range)
    pos = np.asarray(graph.nodes.positions)
    expected = pos[r] - (pos[s] - shifts_e @ cell)
    assert np.allclose(np.asarray(graph.long_range_edge_vectors()), expected)


def test_graph_init_from_chemical_system_long_range_off(
    chemical_system_with_shifts,
):
    """When long_range_cutoff_angstrom is None, edges_long_range and the
    long-range triple are all None."""
    graph = Graph.from_chemical_system(
        chemical_system_with_shifts,
        graph_cutoff_angstrom=WITH_SHIFTS_DISTANCE_CUTOFF,
    )

    assert graph.n_edge_long_range is None
    assert graph.senders_long_range is None
    assert graph.receivers_long_range is None
    assert graph.edges_long_range is None


def test_batch_graphs_concatenates_edges_long_range(
    graph_with_long_range_shifts: Graph,
):
    """batch_graphs concatenates edges_long_range.shifts across graphs."""
    g = graph_with_long_range_shifts
    batched = batch_graphs([g, g])

    assert batched.edges_long_range.shifts is not None
    assert batched.edges_long_range.shifts.shape == (
        2 * g.edges_long_range.shifts.shape[0],
        3,
    )
    # Concatenation order: g's shifts followed by g's shifts again.
    np.testing.assert_allclose(
        np.asarray(
            batched.edges_long_range.shifts[: g.edges_long_range.shifts.shape[0]]
        ),
        np.asarray(g.edges_long_range.shifts),
    )
    np.testing.assert_allclose(
        np.asarray(
            batched.edges_long_range.shifts[g.edges_long_range.shifts.shape[0] :]
        ),
        np.asarray(g.edges_long_range.shifts),
    )


def test_batch_graphs_no_long_range_keeps_none(
    graph_with_shifts: Graph,
):
    """batch_graphs on graphs without long-range interactions leaves
    edges_long_range as None."""
    batched = batch_graphs([graph_with_shifts, graph_with_shifts])

    assert batched.edges_long_range is None


def test_pad_with_graphs_extends_edges_long_range(
    graph_with_long_range_shifts: Graph,
):
    """pad_with_graphs zero-pads edges_long_range.shifts and the first
    n_edge_long_range entries of the padded interactions match the unpadded
    ones element-wise."""
    g = graph_with_long_range_shifts
    g = g.update_node_features(partial_charges=jnp.array([0.5, -0.5, 1.0]))

    n_real_long_range = int(g.n_edge_long_range[0])
    pad_n_edge_long_range = n_real_long_range + 3
    padded = pad_with_graphs(
        g,
        n_node=int(np.sum(g.n_node)) + 2,
        n_edge=int(np.sum(g.n_edge)) + 2,
        n_graph=g.num_graphs + 1,
        n_edge_long_range=pad_n_edge_long_range,
    )

    # edges_long_range.shifts has been padded to the requested size.
    assert padded.edges_long_range.shifts.shape == (pad_n_edge_long_range, 3)
    # Padding shifts are zero.
    np.testing.assert_allclose(
        np.asarray(padded.edges_long_range.shifts[n_real_long_range:]),
        np.zeros((3, 3)),
    )

    # The Coulomb interactions for the real long-range edges match exactly,
    # element-by-element. (Padded entries with sender=receiver=0 and zero
    # shifts have distance 0 and are non-zero in the envelope, so we do NOT
    # compare totals — only the real prefix.)
    interactions_padded = compute_long_range_interactions(padded)
    interactions_unpadded = compute_long_range_interactions(g)
    assert jnp.allclose(
        interactions_padded[:n_real_long_range],
        interactions_unpadded,
        atol=1e-6,
    )


def test_pad_with_graphs_no_long_range_keeps_none(
    graph_with_shifts: Graph,
):
    """pad_with_graphs on a graph without long-range interactions leaves
    edges_long_range as None."""
    padded = pad_with_graphs(
        graph_with_shifts,
        n_node=int(np.sum(graph_with_shifts.n_node)) + 2,
        n_edge=int(np.sum(graph_with_shifts.n_edge)) + 2,
        n_graph=graph_with_shifts.num_graphs + 1,
    )
    assert padded.edges_long_range is None
