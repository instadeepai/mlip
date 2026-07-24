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

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from mlip.simulation.fep.alchemical_graph import AlchemicalGraph
from mlip.simulation.fep.models import repulsive_potentials
from mlip.simulation.fep.models.repulsive_potentials import (
    SoftcoreLennardJonesPotential,
    SoftcoreWCAPotential,
)


def _lennard_jones_potential(
    edge_vectors: jnp.ndarray,
    edges_sigma: jnp.ndarray,
    edges_epsilon: jnp.ndarray,
) -> jnp.ndarray:
    """Standard Lennard-Jones potential.

    Used to comparison with the SCLJ potential at repulsion_weight = 1.0.
    """
    sigma, epsilon = edges_sigma, edges_epsilon
    edge_distances = jnp.linalg.norm(edge_vectors, axis=1)
    pairwise_potential = (
        4 * epsilon * ((sigma / edge_distances) ** 12 - (sigma / edge_distances) ** 6)
    )
    return pairwise_potential


def _wca_potential(
    edge_vectors: jnp.ndarray,
    edges_sigma: jnp.ndarray,
    edges_epsilon: jnp.ndarray,
) -> jnp.ndarray:
    """Standard WCA potential: LJ shifted by epsilon, truncated at r_min=2^(1/6)*sigma.

    Used to check `SoftcoreWCAPotential` at repulsion_weight = 1.0.
    """
    lj = _lennard_jones_potential(edge_vectors, edges_sigma, edges_epsilon)
    edge_distances = jnp.linalg.norm(edge_vectors, axis=1)
    r_min = 2 ** (1 / 6) * edges_sigma
    return jnp.where(edge_distances < r_min, lj + edges_epsilon, 0.0)


POTENTIALS = {
    "softcore_lennard_jones": SoftcoreLennardJonesPotential,
    "softcore_wca": SoftcoreWCAPotential,
}
REFERENCE_AT_FULL_WEIGHT = {
    "softcore_lennard_jones": _lennard_jones_potential,
    "softcore_wca": _wca_potential,
}


@dataclass
class RepulsivePotentialTestCase:
    potential_name: str
    repulsion_weight: float
    expected_energies: jnp.ndarray
    senders: jnp.ndarray = field(default_factory=lambda: jnp.array([0, 0, 2, 3]))
    receivers: jnp.ndarray = field(default_factory=lambda: jnp.array([1, 2, 0, 5]))


def get_softcore_repulsive_potential_test_cases() -> list[RepulsivePotentialTestCase]:
    return [
        RepulsivePotentialTestCase(
            potential_name="softcore_lennard_jones",
            repulsion_weight=0.0,
            expected_energies=[0, 0, 0, 0],
        ),
        RepulsivePotentialTestCase(
            potential_name="softcore_lennard_jones",
            repulsion_weight=0.5,
            expected_energies=[0.184504, 0.035031, 0.035031, -0.00077559],
        ),
        RepulsivePotentialTestCase(
            potential_name="softcore_lennard_jones",
            repulsion_weight=1.0,
            expected_energies=[90.90315, 0.488774, 0.488774, -0.0016090],
        ),
        RepulsivePotentialTestCase(
            potential_name="softcore_wca",
            repulsion_weight=0.0,
            expected_energies=[0, 0, 0, 0],
        ),
        RepulsivePotentialTestCase(
            potential_name="softcore_wca",
            repulsion_weight=0.5,
            expected_energies=[0.189107, 0.038492, 0.038492, 0.0],
        ),
        RepulsivePotentialTestCase(
            potential_name="softcore_wca",
            repulsion_weight=1.0,
            expected_energies=[90.91236, 0.495695, 0.495695, 0.0],
        ),
    ]


@pytest.mark.parametrize("case", get_softcore_repulsive_potential_test_cases())
def test_repulsive_potentials(
    alchemical_graph: AlchemicalGraph,
    case: RepulsivePotentialTestCase,
):
    """Test that each softcore repulsive potential is correct.

    The `expected_energies` were computed using the potentials themselves, hence
    are not ground-truth values, but prevent changing behaviour.
    """
    graph = alchemical_graph.replace(senders=case.senders, receivers=case.receivers)
    potential = POTENTIALS[case.potential_name]()

    energies = potential.compute_pairwise_energies(graph, case.repulsion_weight)
    assert_allclose(energies, jnp.array(case.expected_energies), rtol=1e-4)

    if case.repulsion_weight == 1.0:
        edge_vectors = graph.edge_vectors()
        edges_sigma, edges_epsilon = repulsive_potentials._get_edges_sigma_epsilon(
            graph, potential.sigma_map, potential.epsilon_map
        )
        reference_energies = REFERENCE_AT_FULL_WEIGHT[case.potential_name](
            edge_vectors, edges_sigma, edges_epsilon
        )
        assert_allclose(energies, reference_energies, rtol=1e-4)


def test_per_subgraph_energies_masks_padding_edges(alchemical_graph: AlchemicalGraph):
    """`_per_subgraph_energies` should zero padding edges and sum per real subgraph."""
    # Change graph to have 3 subgraphs, where the last is a dummy graph.
    graph = alchemical_graph.replace(
        senders=jnp.array([0, 2, 4]),
        receivers=jnp.array([1, 3, 4]),
        n_node=jnp.array([2, 2, 1]),
        n_edge=jnp.array([1, 1, 1]),
    )
    potential = SoftcoreLennardJonesPotential()
    repulsion_weight = jnp.array([0.5, 1.0, 1.0])

    per_subgraph_energy = potential._per_subgraph_energies(graph, repulsion_weight)

    edge_energies = potential.compute_pairwise_energies(graph, repulsion_weight)
    assert_allclose(
        per_subgraph_energy, jnp.array([edge_energies[0], edge_energies[1], 0.0])
    )
