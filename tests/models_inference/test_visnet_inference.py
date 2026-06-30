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

    assert list(result.energy) == pytest.approx([-22.16494], abs=1e-3)
    expected_forces = np.array([
        [3.257973, -4.439838, 3.969158],
        [0.37949374, 2.85621, -5.7438984],
        [-1.3700594, -5.9856634, 2.890955],
        [0.574846, -3.7950482, 2.9814265],
        [-0.23810202, 2.011692, -0.6061833],
        [-0.30511624, 0.7987462, -1.2056352],
        [-0.8425261, 2.8831577, -0.23028827],
        [0.23945636, 0.98294044, -1.0757343],
        [-0.7405577, 1.8445723, -0.72196805],
        [-0.9554075, 2.8432314, -0.25783217],
    ])
    assert np.allclose(np.array(result.forces), expected_forces, atol=5e-5)

    assert result.stress is not None and np.any(result.stress != 0.0)
    assert result.pressure is not None and np.any(result.pressure != 0.0)


def test_visnet_v1_vs_legacy_v2_consistent(
    setup_system, legacy_visnet_force_field, visnet_force_field_v1
):
    """The legacy ViSNet path (``use_legacy_visnet=True``) reproduces the v1 model.

    The legacy path mirrors the mlip <= 0.2.1 behaviour and is kept only for
    backward compatibility. This test asserts the legacy path matches the v1
    implementation so old checkpoints reproduce exactly.
    """
    _, graph = setup_system

    result_v1 = jax.jit(visnet_force_field_v1)(graph)
    result_legacy_v2 = jax.jit(legacy_visnet_force_field)(graph)

    assert jnp.allclose(result_v1.energy, result_legacy_v2.energy, atol=1e-6)
    assert jnp.allclose(result_v1.forces, result_legacy_v2.forces, atol=1e-6)
    assert jnp.allclose(result_v1.stress, result_legacy_v2.stress, atol=1e-6)

    assert list(result_v1.energy) == pytest.approx([-25.650845], abs=1e-3)
    expected_forces = np.array([
        [-7.2968043e-03, -2.9765312e-03, -8.4473826e-03],
        [9.5504513e-03, 2.7035501e-02, -8.0039799e-02],
        [1.0298509e-02, 2.8945347e-03, -4.3731909e-03],
        [-2.0968061e-02, 1.9943487e-02, 6.1609928e-02],
        [2.1050749e-03, -7.8989770e-03, 4.7364811e-05],
        [-7.6801440e-04, 7.1482048e-03, 8.8765305e-03],
        [1.6644116e-02, -1.8109571e-02, 9.8979194e-03],
        [-7.0810574e-03, 5.0332304e-03, 7.4180965e-03],
        [2.9909867e-03, -7.5950362e-03, 2.5542988e-04],
        [-5.4751989e-03, -2.5474846e-02, 4.7551086e-03],
    ])
    assert np.allclose(np.array(result_v1.forces), expected_forces, atol=5e-5)


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
