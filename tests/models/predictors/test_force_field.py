import numpy as np
import pytest
from pydantic import BaseModel

from mlip.data import ChemicalSystem
from mlip.graph import Graph
from mlip.models.force_field import ForceField
from mlip.models.inference_context import InferenceContext
from mlip.models.predictors.conservative_predictor import ConservativePredictor
from mlip.models.predictors.energy_heads import (
    coulomb_energy_computation_head,
    standard_energy_computation_head,
)
from mlip.typing import Prediction
from mlip.typing.properties import Properties


class MockModelConfig(BaseModel):
    use_coulomb_term: bool = False
    predict_partial_charges: bool = False


@pytest.fixture
def force_field(quadratic_mlip):
    return ForceField.from_mlip_network(
        quadratic_mlip,
        Properties(energy=True, forces=True, stress=True),
        seed=2,
    )


@pytest.fixture
def salt_graph():
    """Mimic the `salt_graph` fixture from `conftest.py` as a `Graph`."""
    salt = ChemicalSystem(
        atomic_numbers=np.array([11, 17]),
        atomic_species=np.array([0, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.6, 0.5, 0.5]]),
        cell=np.eye(3),
        pbc=(True, True, True),
    )
    return Graph.from_chemical_system(salt, 0.95)


def test_from_mlip_network(force_field):
    assert isinstance(force_field, ForceField)
    assert isinstance(force_field.predictor, ConservativePredictor)


def test_get_energy_head():

    coulomb_config = MockModelConfig(
        use_coulomb_term=True, predict_partial_charges=True
    )
    # Coulomb-trained model returns the Coulomb head regardless of whether the
    # user surfaces partial_charges in the Prediction object.
    assert (
        ForceField.get_energy_head(coulomb_config, Properties(partial_charges=True))
        is coulomb_energy_computation_head
    )
    assert (
        ForceField.get_energy_head(coulomb_config, Properties(partial_charges=False))
        is coulomb_energy_computation_head
    )

    standard_config = MockModelConfig(
        use_coulomb_term=False, predict_partial_charges=False
    )
    assert (
        ForceField.get_energy_head(standard_config, Properties())
        is standard_energy_computation_head
    )


def test_validate_properties():
    ForceField.validate_properties(
        Properties(energy=True, forces=False),
        Properties(energy=True, forces=True),
    )
    with pytest.raises(ValueError):
        ForceField.validate_properties(
            Properties(energy=True, forces=True),
            Properties(energy=True, forces=False),
        )


def test_calculate(force_field, salt_graph):
    graph = force_field.calculate(salt_graph)
    assert isinstance(graph, Graph)
    assert graph.globals.energy is not None
    assert graph.nodes.forces is not None
    assert graph.globals.stress is not None
    assert graph.globals.pressure is not None


def test_predict(force_field, salt_graph):
    prediction = force_field.predict(salt_graph)
    assert isinstance(prediction, Prediction)
    assert prediction.energy is not None
    assert prediction.forces is not None
    assert prediction.stress is not None
    assert prediction.pressure is not None


def test_replace_config_method_of_force_field(force_field):
    new_length = [0.91] * len(force_field.config.length)
    updated_ff = force_field.replace_config(
        length=new_length, add_atomic_energies=False
    )

    assert updated_ff.params is force_field.params
    assert updated_ff.dataset_info is force_field.dataset_info
    assert (
        updated_ff.predictor.required_properties
        is force_field.predictor.required_properties
    )
    assert (
        updated_ff.predictor.mlip_network.available_properties
        == force_field.predictor.mlip_network.available_properties
    )
    assert updated_ff.predictor.energy_head is force_field.predictor.energy_head

    assert updated_ff is not force_field

    assert updated_ff.config.stiffness == force_field.config.stiffness
    assert updated_ff.config.length != force_field.config.length
    assert (
        updated_ff.config.add_atomic_energies != force_field.config.add_atomic_energies
    )

    assert not updated_ff.config.add_atomic_energies
    assert updated_ff.config.length == new_length


def test_prepare_experts_for_inference_rejects_non_moe_model(quadratic_mlip):
    force_field = ForceField.from_mlip_network(
        quadratic_mlip,
        Properties(energy=True, forces=True, stress=True),
        seed=2,
        inference_context=InferenceContext(charge=1),
    )

    with pytest.raises(ValueError, match="does not use mixture-of-experts"):
        force_field.prepare_experts_for_inference()


def test_replace_inference_context_attaches_resolved_context(force_field):
    assert force_field.inference_context is None

    updated_ff = force_field.replace_inference_context(InferenceContext(charge=1))

    assert updated_ff is not force_field
    assert force_field.inference_context is None
    assert updated_ff.inference_context == InferenceContext(charge=1)
    assert updated_ff.params is force_field.params
    assert updated_ff.predictor is force_field.predictor


def test_replace_inference_context_resolves_partial_dataset_name(
    quadratic_mlip, multi_head_dataset_info
):
    multi_head_mlip = type(quadratic_mlip)(
        quadratic_mlip.config, multi_head_dataset_info
    )
    force_field = ForceField.from_mlip_network(
        multi_head_mlip,
        Properties(energy=True, forces=True, stress=True),
        seed=2,
    )

    updated_ff = force_field.replace_inference_context(
        InferenceContext(dataset_name="dataset_1")
    )

    assert updated_ff.inference_context.dataset_name == "dataset_1"
    assert updated_ff.inference_context.dataset_idx == 1
