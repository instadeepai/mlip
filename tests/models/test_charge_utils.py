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
import numpy as np
from ase import Atoms
from numpy.testing import assert_allclose

from mlip.data import ChemicalSystem
from mlip.graph import Graph
from mlip.models.charge_utils import (
    EPSILON_0,
    compute_long_range_interactions,
    correct_partial_charge_feature,
    physnet_envelope_function,
)
from mlip.utils.safe_norm import safe_norm


def test_correct_partial_charge_feature(setup_system):
    _, graph = setup_system
    n_nodes = graph.n_node[0]
    graph = graph.update_node_features(partial_charges=jnp.ones((n_nodes,)))
    graph = graph.replace_globals(charge=jnp.array([1.0]))
    graph = correct_partial_charge_feature(graph)
    corrected_partial_charges = graph.nodes.features["partial_charges"]
    assert_allclose(jnp.sum(corrected_partial_charges), 1.0, atol=1e-5, rtol=1e-4)


def test_compute_long_range_interactions(setup_system):
    atoms, _ = setup_system
    chemical_system = ChemicalSystem.from_ase_atoms(atoms)
    graph = Graph.from_chemical_system(
        chemical_system,
        graph_cutoff_angstrom=3.0,
        long_range_cutoff_angstrom=10.0,
    )
    assert graph.n_edge_long_range is not None
    n_long_range_edges = int(graph.n_edge_long_range[0])
    assert n_long_range_edges > 0

    # add random partial charges to the graph
    n_nodes = int(graph.n_node[0])
    rng = np.random.default_rng(0)
    partial_charges = jnp.asarray(rng.standard_normal(n_nodes), dtype=jnp.float32)
    graph = graph.update_node_features(partial_charges=partial_charges)

    # compute long range interactions
    interactions = compute_long_range_interactions(graph)

    # shape matches the number of long range edges and all values are finite
    assert interactions.shape == (n_long_range_edges,)
    assert jnp.all(jnp.isfinite(interactions))


def test_compute_long_range_interactions_uses_minimum_image_distance():
    """In a periodic cell, the Coulomb sum must use minimum-image distances.

    Two atoms separated by 0.6 along x in a 1-Å cubic cell. Without PBC the
    raw distance is 0.6 (not the minimum). With PBC the minimum-image distance
    is 0.4 (across the boundary).
    """
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]],
        pbc=True,
        cell=[1.0, 1.0, 1.0],
    )
    chemical_system = ChemicalSystem.from_ase_atoms(atoms)
    graph = Graph.from_chemical_system(
        chemical_system,
        graph_cutoff_angstrom=0.5,
        long_range_cutoff_angstrom=0.5,
    )
    graph = graph.update_node_features(partial_charges=jnp.asarray([1.0, -1.0]))

    interactions = compute_long_range_interactions(graph)

    # All long-range edges must reflect the minimum-image distance of 0.4 Å,
    # with charge product -1.0 for every directed edge.
    expected_distance = 0.4
    k_e = 1.0 / (4.0 * jnp.pi * EPSILON_0)
    expected_per_edge = (
        k_e
        * (-1.0)
        * float(physnet_envelope_function(jnp.asarray([expected_distance]))[0])
    )

    assert interactions.shape[0] >= 1
    assert jnp.all(jnp.isfinite(interactions))
    assert jnp.allclose(interactions, expected_per_edge, atol=1e-6)


def test_compute_long_range_interactions_non_pbc_matches_raw_distance(
    setup_system,
):
    """For a non-PBC system with LRI enabled, the new implementation must
    produce results identical to the pre-change formula
    `safe_norm(positions[r] - positions[s])`."""
    atoms, _ = setup_system
    chemical_system = ChemicalSystem.from_ase_atoms(atoms)
    graph = Graph.from_chemical_system(
        chemical_system,
        graph_cutoff_angstrom=3.0,
        long_range_cutoff_angstrom=10.0,
    )
    n_nodes = int(graph.n_node[0])
    rng = np.random.default_rng(0)
    partial_charges = jnp.asarray(rng.standard_normal(n_nodes), dtype=jnp.float32)
    graph = graph.update_node_features(partial_charges=partial_charges)

    actual = compute_long_range_interactions(graph)

    # Reference computation via the pre-change formula.
    positions = graph.nodes.positions
    s = graph.senders_long_range
    r = graph.receivers_long_range
    distances_ref = safe_norm(positions[r] - positions[s], axis=-1)
    k_e = 1.0 / (4.0 * jnp.pi * EPSILON_0)
    charge_interactions = partial_charges[s] * partial_charges[r]
    expected = k_e * charge_interactions * physnet_envelope_function(distances_ref)

    assert jnp.allclose(actual, expected, atol=1e-6)
