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

    assert list(result.energy) == pytest.approx([-34.26291], rel=1e-4, abs=1e-3)
    expected_forces = np.array([
        [-0.19485164, 0.17425559, -0.14922298],
        [-0.12619817, -0.12542307, 0.72570837],
        [0.099122413, 0.27193841, -0.080755413],
        [-0.27341801, 0.99893421, -0.25957197],
        [-0.031038556, 0.19091985, -0.0081819445],
        [0.14821723, -0.19310783, -0.13136150],
        [0.49087793, -0.53768378, 0.10519141],
        [0.056637540, -0.22338006, -0.15273261],
        [-0.083986148, 0.17313249, -0.020567395],
        [-0.085362561, -0.72958577, -0.028505936],
    ])
    assert np.allclose(np.array(result.forces), expected_forces, rtol=1e-3, atol=5e-5)

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

    assert jnp.allclose(result_v1.energy, result_legacy_v2.energy, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(result_v1.forces, result_legacy_v2.forces, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(result_v1.stress, result_legacy_v2.stress, rtol=1e-5, atol=1e-6)

    assert list(result_v1.energy) == pytest.approx([-34.622456], rel=1e-4, abs=1e-3)
    expected_forces = np.array([
        [-0.040895950, -0.083107330, 0.10013036],
        [-0.43358433, 0.52376187, 1.1150863],
        [0.037093583, -0.057199750, 0.11814658],
        [0.90424383, -1.6308402, -1.5468105],
        [0.13796556, -0.057864763, -0.047500961],
        [0.067220338, 0.066016957, -0.10671128],
        [-0.54202390, 0.58150411, 0.23701070],
        [-0.043468732, 0.029171161, -0.13226275],
        [-0.046945579, -0.11972234, -0.090459287],
        [-0.039604779, 0.74828029, 0.35337073],
    ])
    assert np.allclose(
        np.array(result_v1.forces), expected_forces, rtol=1e-3, atol=5e-5
    )


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

    assert jnp.allclose(
        no_remat_result.energy, remat_result.energy, rtol=1e-5, atol=1e-5
    )
    assert jnp.allclose(
        no_remat_result.forces, remat_result.forces, rtol=1e-3, atol=1e-4
    )


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
