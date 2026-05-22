from unittest.mock import MagicMock

import jax
import jax.numpy as jnp

from mlip.graph import GraphEdges
from mlip.models.predictors.energy_heads import (
    coulomb_energy_computation_head,
    standard_energy_computation_head,
)


def test_standard_energy_computation_head():
    """Test that the standard head sums per-node energies."""
    graph = MagicMock()
    graph.nodes.features = {"energy": jnp.array([1.0, 2.0, 4.0])}
    graph.n_node = jnp.array([2, 1])

    result = standard_energy_computation_head(graph)

    assert result.shape == (2,)
    assert jnp.allclose(result, jnp.array([3.0, 4.0]))


def test_coulomb_energy_computation_head(setup_system):
    _, graph = setup_system
    rng = jax.random.PRNGKey(0)
    # add random partial charges to the graph, add random long range senders and
    # receivers
    graph = graph.update_node_features(
        partial_charges=jax.random.randint(rng, (graph.n_node[0],), -3, 3)
    )
    graph = graph.replace(
        senders_long_range=jax.random.randint(
            rng, (graph.n_edge[0],), 0, graph.n_node[0]
        ),
        receivers_long_range=jax.random.randint(
            rng, (graph.n_edge[0],), 0, graph.n_node[0]
        ),
        n_edge_long_range=jnp.array([graph.n_edge[0]]),
        edges_long_range=GraphEdges(
            shifts=jnp.zeros((int(graph.n_edge[0]), 3)),
            displ_fun=None,
        ),
    )
    graph = graph.update_node_features(energy=graph.globals.energy)

    result = coulomb_energy_computation_head(graph)
    assert result.shape == (1,)
