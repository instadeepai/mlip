import flax.linen as nn
import jax.numpy as jnp
import jraph
import numpy as np
import pydantic
import pytest

from mlip.data import ChemicalSystem, DatasetInfo
from mlip.data.helpers import create_graph_from_chemical_system
from mlip.models import ForceField
from mlip.models.mlip_network import MLIPNetwork
from mlip.typing import GraphNodes


class QuadraticMLIP(MLIPNetwork):
    """A dummy MLIPNetwork class.

    Might be reused for numerical checks in Hessian predictors.
    """

    class Config(pydantic.BaseModel):
        stiffness: list[float]
        length: list[float]

    config: Config
    dataset_info: DatasetInfo

    @nn.compact
    def __call__(self, vectors, species, senders, receivers):
        stiffness = jnp.array(self.config.stiffness)
        length = jnp.array(self.config.length)
        specie = species[senders]
        rij = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        spring_terms = 0.5 * stiffness[specie] * (rij - length[specie]) ** 2
        node_energies = jnp.zeros(species.shape[0])
        node_energies = node_energies.at[receivers].add(spring_terms)
        return node_energies


@pytest.fixture
def salt_graph() -> tuple[jraph.GraphsTuple, DatasetInfo]:
    # NaCl CFC lattice
    salt = ChemicalSystem(
        atomic_numbers=np.array([11, 17]),
        atomic_species=np.array([0, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.6, 0.5, 0.5]]),
        cell=np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        pbc=(True, True, True),
    )

    # slightly smaller than lattice width
    cutoff = 0.95

    graph = create_graph_from_chemical_system(
        salt,
        cutoff,
        batch_it_with_minimal_dummy=True,
    )

    return graph, DatasetInfo(
        atomic_energies_map={11: 0.0, 17: 0.0},
        cutoff_distance_angstrom=cutoff,
    )


def test_pressure_is_positive(salt_graph):
    """Assert p > 0."""
    graph, dataset_info = salt_graph
    cfg = QuadraticMLIP.Config(
        stiffness=[2.0, 2.0],
        length=[0.87, 0.87],
    )
    predictor = ForceField.from_mlip_network(
        QuadraticMLIP(cfg, dataset_info),
        seed=2,
        predict_stress=True,
    )
    prediction = predictor(graph)
    assert np.all(prediction.pressure[:-1] > 0)


def test_virial_is_origin_independent(salt_graph):
    graph_1, dataset_info = salt_graph
    cfg = QuadraticMLIP.Config(
        stiffness=[2.0, 2.0],
        length=[0.87, 0.87],
    )
    predictor = ForceField.from_mlip_network(
        QuadraticMLIP(cfg, dataset_info),
        seed=2,
        predict_stress=True,
    )
    virial_1 = predictor(graph_1).stress_virial

    # Translate structure
    translation = np.array([0.2, 0.3, 0.1])
    graph_2 = graph_1._replace(
        nodes=GraphNodes(
            positions=graph_1.nodes.positions + translation,
            species=graph_1.nodes.species,
        )
    )
    virial_2 = predictor(graph_2).stress_virial

    # Virial stress independent of origin
    assert jnp.allclose(virial_1[0], virial_2[0], atol=1e-5, rtol=1e-5)
