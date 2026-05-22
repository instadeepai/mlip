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
from mlip.graph.batching_helpers import (
    batch_graphs,
    homogenize_graph_fields,
    pad_with_graphs,
)


def test_nequip_outputs_correct_forces_and_energies_for_single_graph(
    setup_system,
    nequip_force_field,
):
    _, graph = setup_system
    nequip_ff = nequip_force_field

    result = jax.jit(nequip_ff)(graph)

    assert list(result.energy) == pytest.approx([42.09434], abs=1e-3)
    expected_forces = np.array([
        [85.43408, -34.709274, 22.736664],
        [-27.643398, 80.936386, 2.8183188],
        [-50.41974, -80.04627, -8.698838],
        [3.3170373, -2.7197578, -10.36809],
        [-19.162735, -43.858597, -37.10254],
        [1.0938892, 45.787735, 25.715027],
        [-8.596391, 9.238947, 3.2427912],
        [-37.445065, 33.05269, 16.969948],
        [53.844803, -19.67092, -20.328377],
        [-0.42247772, 11.989063, 5.0151005],
    ])
    assert np.allclose(np.array(result.forces), expected_forces, atol=5e-3)

    assert result.stress is not None and np.any(result.stress != 0.0)
    assert result.pressure is not None and np.any(result.pressure != 0.0)


def test_nequip_model_versions_consistent(
    setup_system, nequip_force_field, nequip_force_field_v1
):
    _, graph = setup_system

    result_v1 = jax.jit(nequip_force_field_v1)(graph)
    result_v2 = jax.jit(nequip_force_field)(graph)

    assert_allclose(result_v1.energy, result_v2.energy, atol=1e-4)
    assert_allclose(result_v1.forces, result_v2.forces, atol=5e-4, rtol=5e-4)
    assert_allclose(result_v1.stress, result_v2.stress, atol=5e-4, rtol=5e-4)


def test_nequip_grad_params(setup_system, nequip_force_field, pad_graph):
    _, graph = setup_system
    graph = pad_graph(graph, 4, 34, 92)
    _apply = jax.jit(nequip_force_field.predictor.apply)
    params = nequip_force_field.params

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


def test_multi_head_nequip(
    setup_system,
    nequip_force_field,
    multi_head_nequip_force_field,
):
    """A multi-head NequIP model produces valid energy, forces, and stress."""
    jitted_single_head_ff = jax.jit(nequip_force_field)
    jitted_multi_head_ff = jax.jit(multi_head_nequip_force_field)

    _, graph = setup_system
    result_single = jitted_single_head_ff(graph)

    _, graph = setup_system
    result_multi = jitted_multi_head_ff(graph)

    assert result_multi.energy is not None and np.any(result_multi.energy != 0.0)
    assert result_multi.forces is not None and np.any(result_multi.forces != 0.0)
    assert result_multi.stress is not None and np.any(result_multi.stress != 0.0)
    assert not np.any(np.isnan(result_multi.energy))
    assert not np.any(np.isnan(result_multi.forces))
    assert not np.any(np.isnan(result_multi.stress))

    assert_allclose(result_single.energy, result_multi.energy, atol=1e-5, rtol=1e-4)

    # Now add the dataset_idx 0 on the graph and assert the same result
    _, graph = setup_system
    graph = graph.replace_globals(dataset_idx=jnp.array([0]))

    result_with_idx = jitted_multi_head_ff(graph)

    assert result_with_idx.energy is not None and np.any(result_with_idx.energy != 0.0)
    assert_allclose(result_with_idx.energy, result_multi.energy, atol=1e-5, rtol=1e-4)


def test_nequip_predicts_partial_charges(
    setup_system, partial_charges_nequip_force_field
):
    _, graph = setup_system
    graph = graph.replace_globals(charge=jnp.array([1.0]))
    out_graph = jax.jit(partial_charges_nequip_force_field.calculate)(graph)
    result = out_graph.to_prediction()

    assert result.partial_charges is not None and np.any(result.partial_charges != 0.0)
    assert result.partial_charges.shape == (graph.n_node[0])

    # Assert partial charges are corrected to total charge.
    pred_total_charge = jnp.sum(result.partial_charges)
    ref_total_charge = out_graph.globals.charge[0]
    assert_allclose(pred_total_charge, ref_total_charge, atol=1e-5, rtol=1e-4)


def test_nequip_uses_coulomb_term(
    setup_system, lri_nequip_force_field, nequip_force_field
):
    atoms, _ = setup_system
    graph = Graph.from_chemical_system(
        ChemicalSystem.from_ase_atoms(atoms),
        graph_cutoff_angstrom=3.0,
        long_range_cutoff_angstrom=5.0,
    )
    graph = graph.replace_globals(charge=jnp.array([1.0]))
    lri_out_graph = jax.jit(lri_nequip_force_field.calculate)(graph)
    lri_result = lri_out_graph.to_prediction()

    out_graph = jax.jit(nequip_force_field.calculate)(graph)
    result = out_graph.to_prediction()

    assert not jnp.allclose(lri_result.energy, result.energy, atol=1e-6)
    assert not jnp.allclose(lri_result.forces, result.forces, atol=1e-6)


def test_multi_head_nequip_batched(
    setup_system,
    salt_graph,
    multi_head_nequip_force_field,
):
    """Batched forward pass with mixed dataset_idx selects correct heads."""
    ff = replace(multi_head_nequip_force_field, inference_context=None)

    _, graph_a = setup_system
    graph_a = graph_a.replace_globals(dataset_idx=jnp.array([0]))

    graph_b = salt_graph
    graph_b = graph_b.replace_globals(dataset_idx=jnp.array([1]))

    graph_a, graph_b = homogenize_graph_fields([graph_a, graph_b])
    batched = batch_graphs([graph_a, graph_b])
    batched = pad_with_graphs(batched, n_node=50, n_edge=200, n_graph=4)

    result = jax.jit(ff)(batched)

    energy = np.asarray(result.energy).flatten()
    # First two graphs are real, rest are padding.
    assert not np.any(np.isnan(energy[:2]))
    assert not np.any(np.isnan(result.forces))
    assert not np.any(np.isnan(np.asarray(result.stress)[:2]))

    # The two real graphs should get different energies (different heads + e0s).
    assert not np.isclose(energy[0].item(), energy[1].item())
