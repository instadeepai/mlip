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

import logging
from copy import deepcopy

import ase
import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator

from mlip.data import (
    ASEAtomsReader,
    GraphDatasetBuilderConfig,
    SingleGraphDatasetBuilder,
)
from mlip.data.graph_dataset import GraphDataset
from mlip.inference import run_batched_inference


def _graph_dataset_from_atoms(
    atoms: list[ase.Atoms], batch_size=1, dataset_info=True, prefetch=False
) -> GraphDataset:
    builder_config = GraphDatasetBuilderConfig(
        graph_cutoff_angstrom=3.0,
        max_n_node=None,
        max_n_edge=None,
        batch_size=batch_size,
        should_shuffle=False,
    )
    for system in atoms:
        if system.get_calculator() is None:
            calculator = SinglePointCalculator(system)
            system.set_calculator(calculator)
    atoms_reader = ASEAtomsReader(atoms)

    builder = SingleGraphDatasetBuilder(
        atoms_reader, builder_config, dataset_info=dataset_info
    )
    graph_dataset = builder.get_dataset(prefetch=prefetch)

    return graph_dataset


@pytest.mark.parametrize(
    "use_single_structure,from_atoms,batch_size",
    [(True, False, 1), (False, False, 3), (False, True, 4)],
)
def test_batched_inference_works_correctly(
    setup_system,
    quadratic_force_field,
    caplog,
    use_single_structure,
    from_atoms,
    batch_size,
):
    atoms, _ = setup_system
    caplog.set_level(logging.INFO)
    num_structures = 7
    structures = []
    for _ in range(num_structures - 1):
        structures.append(deepcopy(atoms))

    # Last structure is a little bit by deleting first atom
    structures.append(
        ase.Atoms(
            numbers=structures[-1].numbers[1:],
            positions=structures[-1].positions[1:, :],
        )
    )

    if use_single_structure:
        structures = structures[:1]

    if not from_atoms:
        graph_dataset = _graph_dataset_from_atoms(
            structures, batch_size, quadratic_force_field.dataset_info
        )
        result = run_batched_inference(graph_dataset, quadratic_force_field)
    else:
        result = run_batched_inference(structures, quadratic_force_field, batch_size=4)

        graph_dataset = _graph_dataset_from_atoms(
            structures, batch_size=3, dataset_info=quadratic_force_field.dataset_info
        )
        result = run_batched_inference(graph_dataset, quadratic_force_field)

        assert len(result) == len(structures)
        assert isinstance(result[0].energy, float)
        assert result[-1].forces.shape == (len(atoms) - 1, 3)
        assert result[-1].energy == pytest.approx(0.15324, abs=1e-3)
        assert result[-1].forces[0][0] == pytest.approx(0.00614, abs=1e-3)

        # First 6 energies should be the same
        for i in range(1, num_structures - 1):
            assert result[i].energy == pytest.approx(result[0].energy)

        # First 6 forces should be the same
        for i in range(1, num_structures - 1):
            assert np.allclose(result[i].forces, result[0].forces)

        # Asserting correct values in logs
        assert f"on {num_structures} structure(s) in 3 batches" in caplog.text
        for i in [1, 2, 3]:
            assert f"Batch {i} completed." in caplog.text

    # These can be tested in both scenarios
    assert result[0].forces.shape == (len(atoms), 3)
    assert result[0].stress is not None
    assert result[0].stress is not None
    assert result[0].energy == pytest.approx(0.19749, abs=1e-3)
    assert result[0].forces[0][0] == pytest.approx(-0.04018, abs=1e-3)


def test_batched_inference_with_graph_without_edges(
    setup_system, quadratic_force_field
):
    atoms, _ = setup_system
    positions = np.array([[0, 0, 0], [0, 0, 10]])
    atomic_numbers = np.array([6, 6])
    atoms_without_edges = ase.Atoms(positions=positions, numbers=atomic_numbers)

    structures = [atoms_without_edges, atoms]
    graph_dataset = _graph_dataset_from_atoms(
        structures, batch_size=2, dataset_info=quadratic_force_field.dataset_info
    )
    result = run_batched_inference(graph_dataset, quadratic_force_field, batch_size=2)

    # For first structure, no energy if no edges
    assert result[0].energy == 0.0
    assert not result[0].forces.any()

    # For second structure, normal result
    assert result[1].energy == pytest.approx(0.19749, abs=1e-3)
    assert result[1].forces[0][0] == pytest.approx(-0.04018, abs=1e-3)


def test_batched_inference_predicts_partial_charges(
    setup_system, partial_charges_mace_force_field
):
    atoms, _ = setup_system
    num_structures = 7
    total_charge = 1.0
    structures = [deepcopy(atoms) for _ in range(num_structures - 1)]
    structures.append(
        ase.Atoms(
            numbers=structures[-1].numbers[1:],
            positions=structures[-1].positions[1:, :],
        )
    )
    for system in structures:
        system.info["charge"] = total_charge

    graph_dataset = _graph_dataset_from_atoms(
        structures,
        batch_size=3,
        dataset_info=partial_charges_mace_force_field.dataset_info,
    )
    result = run_batched_inference(graph_dataset, partial_charges_mace_force_field)

    assert len(result) == num_structures

    for i in range(num_structures - 1):
        assert result[i].partial_charges is not None
        assert result[i].partial_charges.shape == (len(atoms),)
        assert np.any(result[i].partial_charges != 0.0)

    # Trimmed structure must have one fewer atom worth of charges, verifying
    # per-structure slicing in run_batched_inference.
    assert result[-1].partial_charges is not None
    assert result[-1].partial_charges.shape == (len(atoms) - 1,)

    # Identical inputs must yield identical charge predictions across batches.
    for i in range(1, num_structures - 1):
        assert np.allclose(result[i].partial_charges, result[0].partial_charges)

    # Charge conservation: corrected charges must sum to the total charge.
    for r in result:
        assert np.sum(r.partial_charges) == pytest.approx(total_charge, abs=1e-5)


def test_batched_inference_with_hessian_predictor(
    setup_system, quadratic_hessian_force_field
):
    atoms, _ = setup_system
    structures = []
    num_structures = 5
    for _ in range(num_structures - 1):
        structures.append(deepcopy(atoms))

    structures.append(
        ase.Atoms(
            numbers=structures[-1].numbers[1:],
            positions=structures[-1].positions[1:, :],
        )
    )

    graph_dataset = _graph_dataset_from_atoms(structures, batch_size=3)
    result = run_batched_inference(graph_dataset, quadratic_hessian_force_field)

    assert len(result) == num_structures
    assert result[-1].forces.shape == (len(atoms) - 1, 3)
    assert result[-1].hessian.shape == (len(atoms) - 1, 3, len(atoms) - 1, 3)
