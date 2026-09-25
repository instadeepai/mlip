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

from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp
import jax_md
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mlip.data import ChemicalSystem
from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.graph import Graph
from mlip.graph.graph import GraphEdges
from mlip.simulation.fep.alchemical_graph import AlchemicalGraph
from mlip.simulation.fep.alchemical_graph.masking import (
    get_alchemical_edge_mask,
    prune_edges_with_mask,
)


@dataclass
class AlchemicalEdgeMaskTestCase:
    senders: jnp.ndarray
    receivers: jnp.ndarray
    expected_mask: jnp.ndarray
    alchemical_atom_indices: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0, 3])
    )


def get_alchemical_edge_mask_test_cases() -> list[AlchemicalEdgeMaskTestCase]:
    return [
        # Case 1: Mix of boundary and non-boundary edges
        AlchemicalEdgeMaskTestCase(
            senders=jnp.array([0, 0, 0, 1, 1, 1]),
            receivers=jnp.array([1, 2, 3, 0, 2, 3]),
            expected_mask=jnp.array([1, 1, 0, 1, 0, 1]),
        ),
        # Case 2: All boundary edges
        AlchemicalEdgeMaskTestCase(
            senders=jnp.array([0, 0, 3, 3]),
            receivers=jnp.array([1, 2, 2, 1]),
            expected_mask=jnp.array([1, 1, 1, 1]),
        ),
        # Case 3: All non-boundary edges
        AlchemicalEdgeMaskTestCase(
            senders=jnp.array([0, 1, 3, 2]),
            receivers=jnp.array([3, 2, 0, 1]),
            expected_mask=jnp.array([0, 0, 0, 0]),
        ),
    ]


@pytest.mark.parametrize("case", get_alchemical_edge_mask_test_cases())
def test_get_alchemical_edge_mask(
    alchemical_graph: AlchemicalGraph, case: AlchemicalEdgeMaskTestCase
):
    """Test that the `get_alchemical_edge_mask` function is correct."""
    graph = alchemical_graph.replace(
        senders=case.senders, receivers=case.receivers
    ).replace_globals(alchemical_atom_indices=case.alchemical_atom_indices)
    output_mask, _ = get_alchemical_edge_mask(graph)
    assert_array_equal(output_mask, case.expected_mask.astype(bool))


@dataclass
class PruneEdgesWithMaskTestCase:
    mask: jnp.ndarray = field(
        default_factory=lambda: jnp.array([True, True, False, False, False])
    )
    senders: jnp.ndarray = field(default_factory=lambda: jnp.array([0, 0, 1, 2, 2]))
    receivers: jnp.ndarray = field(default_factory=lambda: jnp.array([1, 2, 3, 1, 0]))
    shifts: jnp.ndarray = field(
        default_factory=lambda: jnp.array([[0.0, 0.0, 0.0] * 5])
    )
    displ_fun: Callable | None = None
    expected_senders: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0, 0, 10, 10, 10])
    )
    expected_receivers: jnp.ndarray = field(
        default_factory=lambda: jnp.array([1, 2, 10, 10, 10])
    )
    expected_shifts: jnp.ndarray = field(
        default_factory=lambda: jnp.array([[0.0, 0.0, 0.0] * 5])
    )


def get_prune_edges_with_mask_test_cases() -> list[PruneEdgesWithMaskTestCase]:
    return [
        PruneEdgesWithMaskTestCase(),
        PruneEdgesWithMaskTestCase(
            shifts=None,
            displ_fun=jax_md.space.free()[0],
            expected_shifts=None,
        ),
        PruneEdgesWithMaskTestCase(
            mask=jnp.array([False, True, False, True, True]),
            expected_senders=np.array([0, 2, 2, 10, 10]),
            expected_receivers=np.array([2, 1, 0, 10, 10]),
        ),
    ]


@pytest.mark.parametrize("case", get_prune_edges_with_mask_test_cases())
def test_prune_edges_with_mask(
    alchemical_graph: Graph, case: PruneEdgesWithMaskTestCase
):
    """Test that the `prune_edges_with_mask` function is correct."""
    graph = alchemical_graph.replace(
        senders=case.senders,
        receivers=case.receivers,
        edges=GraphEdges(shifts=case.shifts, displ_fun=case.displ_fun),
        n_edge=jnp.array([len(case.senders)]),
    )
    new_graph = prune_edges_with_mask(graph, case.mask)
    assert_array_equal(new_graph.senders, case.expected_senders)
    assert_array_equal(new_graph.receivers, case.expected_receivers)
    assert_array_equal(new_graph.edges.shifts, case.expected_shifts)


def create_distant_waters_system(fully_connected: bool) -> Graph:
    """Create a system of two water molecules, 20A apart."""
    distance_cutoff_angstrom = 100.0 if fully_connected else 5.0
    system = ChemicalSystem(
        atomic_numbers=np.array([1, 8, 1, 1, 8, 1]),
        positions=np.concatenate([
            np.array([[-0.5, 0.0, 0.0], [0.0, 0.2, 0.0], [0.5, 0.0, 0.0]]),
            np.array([[-0.5, 0.0, 0.0], [0.0, 0.2, 0.0], [0.5, 0.0, 0.0]]) + 20.0,
        ]),
    )
    return Graph.from_chemical_system(system, distance_cutoff_angstrom)


@pytest.fixture
def distant_waters_connected() -> Graph:
    """Fully-connected system containing two distant water molecules."""
    return create_distant_waters_system(fully_connected=True)


@pytest.fixture
def distant_waters_disconnected() -> Graph:
    """Disconnected system containing two distant water molecules."""
    return create_distant_waters_system(fully_connected=False)


@pytest.fixture
def distant_waters_mask(distant_waters_connected: Graph) -> np.ndarray:
    """Create a mask of edges between first water and distant water."""
    graph = distant_waters_connected
    senders_in_first_water = jnp.isin(graph.senders, np.array([0, 1, 2]))
    receivers_in_first_water = jnp.isin(graph.receivers, np.array([0, 1, 2]))
    mask = ~jnp.logical_xor(senders_in_first_water, receivers_in_first_water)
    return mask.astype(bool)


def test_prune_edges_matches_neighbor_list_padding(
    distant_waters_connected: Graph, distant_waters_mask: np.ndarray
):
    """Test `prune_edges_with_mask` padding matches `jax_md.partition.neighbor_list`.

    `neighbor_list` is used to track edges and pad to a consistent shape. This test
    checks that the padding behaviour of `prune_edges_with_mask` matches this.
    """
    graph = distant_waters_connected
    neighbor_fun = jax_md.partition.neighbor_list(
        displacement_or_metric=jax_md.space.free()[0],
        box=jnp.nan,
        r_cutoff=5.0,  # No edges between distant waters.
        disable_cell_list=False,
        format=jax_md.partition.NeighborListFormat.Sparse,
        capacity_multiplier=2.5,  # To pad to N*N edges.
        custom_mask_function=None,
    )
    neighbors = neighbor_fun.allocate(graph.nodes.positions)
    senders_jax_md, receivers_jax_md = neighbors.idx[1, :], neighbors.idx[0, :]

    graph_pruned = prune_edges_with_mask(graph, distant_waters_mask)
    assert_array_equal(graph_pruned.senders, senders_jax_md)
    assert_array_equal(graph_pruned.receivers, receivers_jax_md)


def test_prune_edges_leaves_absent_shifts(
    distant_waters_connected: Graph, distant_waters_mask: np.ndarray
):
    assert distant_waters_connected.edges.shifts is None
    graph_pruned = prune_edges_with_mask(distant_waters_connected, distant_waters_mask)
    assert graph_pruned.edges.shifts is None


@pytest.mark.parametrize("translate_shifts", [0.0, 1.0])
def test_prune_edges_matches_dynamically_batch(
    distant_waters_connected: Graph,
    distant_waters_mask: np.ndarray,
    distant_waters_disconnected: Graph,
    translate_shifts: float,
):
    """Test `prune_edges_with_mask` padding matches `dynamically_batch`.

    `dynamically_batch` is used to add a dummy graph, treated as padding. This test
    checks that the padding behaviour of `prune_edges_with_mask` matches this.
    """

    # Add shifts to check they are treated consistently between the methods.
    def with_uniform_shifts(graph: Graph) -> Graph:
        return graph.replace(
            edges=GraphEdges(
                shifts=np.full((graph.senders.shape[0], 3), translate_shifts),
                displ_fun=None,
            ),
        )

    graph_connected = with_uniform_shifts(distant_waters_connected)
    graph_disconnected = with_uniform_shifts(distant_waters_disconnected)

    # Pad to shape given by the capacities, and add dummy graph
    graph_padded = next(
        dynamically_batch(
            [graph_disconnected],
            n_node=graph_connected.n_node[0] + 1,
            n_edge=graph_connected.n_edge[0] + 1,
            n_graph=2,
        )
    )
    # Remove dummy graph's edges
    senders_padded = graph_padded.senders[:-1]
    receivers_padded = graph_padded.receivers[:-1]
    shifts_padded = graph_padded.edges.shifts[:-1]

    graph_pruned = prune_edges_with_mask(graph_connected, distant_waters_mask)
    assert_array_equal(graph_pruned.senders, senders_padded)
    assert_array_equal(graph_pruned.receivers, receivers_padded)
    assert_array_equal(graph_pruned.edges.shifts, shifts_padded)
