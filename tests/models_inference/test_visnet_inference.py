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


from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mlip.data import ChemicalSystem
from mlip.graph import Graph
from mlip.models.force_field import ForceField
from mlip.models.visnet.network import Visnet


def test_visnet_outputs_correct_forces_and_energies_for_single_graph(
    setup_system, visnet_force_field
):
    _, graph = setup_system
    visnet_ff = visnet_force_field

    result = jax.jit(visnet_ff)(graph)

    assert list(result.energy) == pytest.approx([-25.650845], abs=1e-3)
    expected_forces = np.array([
        [-7.2968267e-03, -2.9765684e-03, -8.4474469e-03],
        [9.5504019e-03, 2.7035497e-02, -8.0040090e-02],
        [1.0298559e-02, 2.8945699e-03, -4.3731583e-03],
        [-2.0968132e-02, 1.9943571e-02, 6.1610069e-02],
        [2.1050731e-03, -7.8989938e-03, 4.7375535e-05],
        [-7.6804560e-04, 7.1482547e-03, 8.8765882e-03],
        [1.6644197e-02, -1.8109651e-02, 9.8979529e-03],
        [-7.0810374e-03, 5.0333040e-03, 7.4181352e-03],
        [2.9910570e-03, -7.5949985e-03, 2.5546132e-04],
        [-5.4752436e-03, -2.5474990e-02, 4.7551133e-03],
    ])
    assert np.allclose(np.array(result.forces), expected_forces, atol=5e-5)

    assert result.stress is not None and np.any(result.stress != 0.0)
    assert result.pressure is not None and np.any(result.pressure != 0.0)


def test_visnet_with_use_remat_matches_without(
    setup_system, visnet_config, dataset_info, visnet_force_field
):
    _, graph = setup_system
    no_remat_config = visnet_config.model_copy(update={"use_remat": False})
    no_remat_model = Visnet(no_remat_config, dataset_info)
    no_remat_ff = ForceField(
        replace(visnet_force_field.predictor, mlip_network=no_remat_model),
        visnet_force_field.params,
    )

    remat_config = visnet_config.model_copy(update={"use_remat": True})
    remat_model = Visnet(remat_config, dataset_info)
    remat_ff = ForceField(
        replace(visnet_force_field.predictor, mlip_network=remat_model),
        visnet_force_field.params,
    )

    no_remat_result = jax.jit(no_remat_ff)(graph)
    remat_result = jax.jit(remat_ff)(graph)

    assert jnp.allclose(no_remat_result.energy, remat_result.energy, atol=1e-5)
    assert jnp.allclose(no_remat_result.forces, remat_result.forces, atol=1e-3)


def test_visnet_model_versions_consistent(
    setup_system, visnet_force_field, visnet_force_field_v1
):
    _, graph = setup_system

    result_v1 = jax.jit(visnet_force_field_v1)(graph)
    result_v2 = jax.jit(visnet_force_field)(graph)

    assert jnp.allclose(result_v1.energy, result_v2.energy, atol=1e-6)
    assert jnp.allclose(result_v1.forces, result_v2.forces, atol=1e-6)
    assert jnp.allclose(result_v1.stress, result_v2.stress, atol=1e-6)


def test_visnet_grad_params(setup_system, visnet_force_field, pad_graph):
    _, graph = setup_system
    graph = pad_graph(graph, 4, 34, 92)
    _apply = jax.jit(visnet_force_field.predictor.apply)
    params = visnet_force_field.params

    energy_loss = lambda p, g: jnp.sum(_apply(p, g).globals.energy)  # noqa: E731
    forces_loss = lambda p, g: jnp.sum(_apply(p, g).nodes.forces ** 2)  # noqa: E731
    stress_loss = lambda p, g: jnp.sum(_apply(p, g).globals.stress ** 2)  # noqa: E731

    for loss_fn in [energy_loss, forces_loss, stress_loss]:
        backprop = jax.grad(loss_fn)
        params_grad = backprop(params, graph)
        leaves_grad = jax.tree.leaves(params_grad)

        for p in leaves_grad:
            assert not jnp.any(jnp.isnan(p)), f"NaN for {loss_fn.__name__}"
            assert not jnp.any(jnp.isinf(p)), f"Inf for {loss_fn.__name__}"


def test_visnet_predicts_partial_charges(
    setup_system, partial_charges_visnet_force_field
):
    _, graph = setup_system
    graph = graph.replace_globals(charge=jnp.array([1.0]))
    out_graph = jax.jit(partial_charges_visnet_force_field.calculate)(graph)
    result = out_graph.to_prediction()

    assert result.partial_charges is not None and np.any(result.partial_charges != 0.0)
    assert result.partial_charges.shape == (graph.n_node[0])

    # Assert partial charges are corrected to total charge.
    pred_total_charge = jnp.sum(result.partial_charges)
    ref_total_charge = out_graph.globals.charge[0]
    assert_allclose(pred_total_charge, ref_total_charge, atol=1e-5, rtol=1e-4)


def test_visnet_uses_coulomb_term(
    setup_system, lri_visnet_force_field, visnet_force_field
):
    atoms, _ = setup_system
    graph = Graph.from_chemical_system(
        ChemicalSystem.from_ase_atoms(atoms),
        graph_cutoff_angstrom=3.0,
        long_range_cutoff_angstrom=5.0,
    )
    graph = graph.replace_globals(charge=jnp.array([1.0]))
    lri_out_graph = jax.jit(lri_visnet_force_field.calculate)(graph)
    lri_result = lri_out_graph.to_prediction()

    out_graph = jax.jit(visnet_force_field.calculate)(graph)
    result = out_graph.to_prediction()

    assert not jnp.allclose(lri_result.energy, result.energy, atol=1e-6)
    assert not jnp.allclose(lri_result.forces, result.forces, atol=1e-6)


def test_visnet_with_total_charge_embedding(
    setup_system,
    total_charge_embedding_visnet_force_field,
    visnet_force_field,
):
    _, graph = setup_system
    graph = graph.replace_globals(charge=jnp.array([1.0]))

    # Apply the FF with total charge embedding
    out_graph_with_charge_embedding = jax.jit(
        total_charge_embedding_visnet_force_field.calculate
    )(graph)
    result_with_charge_embedding = out_graph_with_charge_embedding.to_prediction()

    # Apply the FF without total charge embedding
    out_graph_no_charge_embedding = jax.jit(visnet_force_field.calculate)(graph)
    result_no_charge_embedding = out_graph_no_charge_embedding.to_prediction()

    # Assert that energies and forces are different between the two (as expected)
    assert not jnp.allclose(
        result_with_charge_embedding.energy,
        result_no_charge_embedding.energy,
        atol=1e-6,
    ), "Energies should differ due to total charge embedding"
    assert not jnp.allclose(
        result_with_charge_embedding.forces,
        result_no_charge_embedding.forces,
        atol=1e-6,
    ), "Forces should differ due to total charge embedding"
