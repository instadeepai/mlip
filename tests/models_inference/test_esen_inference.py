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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mlip.data import ChemicalSystem
from mlip.graph import Graph


def test_esen_outputs_correct_forces_and_energies_for_single_graph(
    setup_system,
    esen_force_field,
):
    _, graph = setup_system
    esen_ff = esen_force_field

    result = jax.jit(esen_ff)(graph)

    assert list(result.energy) == pytest.approx([97.18425], rel=1e-4, abs=1e-3)
    expected_forces = np.array([
        [0.8912778, 0.048211426, 0.2944538],
        [-0.9329636, 1.6486659, -0.032038093],
        [-0.60205215, -0.5056275, -0.5119401],
        [0.5625181, -0.23489952, -0.19773436],
        [0.042880505, -0.9129689, -0.093554974],
        [-0.19032158, 0.05515289, 0.72733825],
        [-0.8467916, 0.15661353, -0.7663468],
        [-0.100945726, 0.16474894, 0.86770296],
        [0.56167263, -0.7224676, 0.12419792],
        [0.6147257, 0.30257082, -0.41207868],
    ])
    assert np.allclose(np.array(result.forces), expected_forces, rtol=1e-3, atol=5e-4)

    assert result.stress is not None and np.any(result.stress != 0.0)
    assert result.pressure is not None and np.any(result.pressure != 0.0)


def test_esen_grad_params(setup_system, esen_force_field, pad_graph):
    _, graph = setup_system
    graph = pad_graph(graph, 4, 34, 92)
    _apply = jax.jit(esen_force_field.predictor.apply)
    params = esen_force_field.params

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


def test_esen_predicts_partial_charges(setup_system, partial_charges_esen_force_field):
    _, graph = setup_system
    graph = graph.replace_globals(charge=jnp.array([1.0]))
    out_graph = jax.jit(partial_charges_esen_force_field.calculate)(graph)
    result = out_graph.to_prediction()

    assert result.partial_charges is not None and np.any(result.partial_charges != 0.0)
    assert result.partial_charges.shape == (graph.n_node[0])

    # Assert partial charges are corrected to total charge.
    pred_total_charge = jnp.sum(result.partial_charges)
    ref_total_charge = out_graph.globals.charge[0]
    assert_allclose(pred_total_charge, ref_total_charge, atol=1e-5, rtol=1e-4)


def test_esen_uses_coulomb_term(setup_system, lri_esen_force_field, esen_force_field):
    atoms, _ = setup_system
    graph = Graph.from_chemical_system(
        ChemicalSystem.from_ase_atoms(atoms),
        graph_cutoff_angstrom=3.0,
        long_range_cutoff_angstrom=5.0,
    )
    graph = graph.replace_globals(charge=jnp.array([1.0]))
    lri_out_graph = jax.jit(lri_esen_force_field.calculate)(graph)
    lri_result = lri_out_graph.to_prediction()

    out_graph = jax.jit(esen_force_field.calculate)(graph)
    result = out_graph.to_prediction()

    assert not jnp.allclose(lri_result.energy, result.energy, atol=1e-6)
    assert not jnp.allclose(lri_result.forces, result.forces, atol=1e-6)
