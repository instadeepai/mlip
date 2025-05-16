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

from mlip.inference import run_batched_inference


def test_batched_inference_works_correctly(setup_system_and_mace_model, caplog):
    atoms, _, _, mace_ff = setup_system_and_mace_model
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

    result = run_batched_inference(structures, mace_ff, batch_size=3)

    assert len(result) == num_structures
    assert isinstance(result[0].energy, float)
    assert result[0].forces.shape == (len(atoms), 3)
    assert result[-1].forces.shape == (len(atoms) - 1, 3)
    assert result[0].stress is None
    assert result[0].stress_cell is None
    assert result[0].stress_forces is None
    assert result[0].pressure is None

    assert result[0].energy == pytest.approx(-0.11254195, abs=1e-3)
    assert result[-1].energy == pytest.approx(-0.06119524, abs=1e-3)
    assert result[0].forces[0][0] == pytest.approx(0.04921325, abs=1e-3)
    assert result[-1].forces[0][0] == pytest.approx(4.61012e-3, abs=1e-3)

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
