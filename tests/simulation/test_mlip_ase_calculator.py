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

import numpy as np
import pytest
from ase import Atoms

from mlip.models import ForceField
from mlip.models.inference_context import InferenceContext
from mlip.models.mace.network import Mace
from mlip.simulation.ase.mlip_ase_calculator import MLIPForceFieldASECalculator
from mlip.typing.properties import Properties

EDGE_CAPACITY_MULTIPLIER = 1.5
PROPS = Properties(energy=True, forces=True)


@pytest.fixture
def water():
    return Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])


def _calculator(atoms, force_field):
    return MLIPForceFieldASECalculator(atoms, EDGE_CAPACITY_MULTIPLIER, force_field)


def test_calculator_applies_the_inference_context(mace_config, dataset_info, water):
    force_field = ForceField.from_mlip_network(
        Mace(
            config=mace_config.model_copy(update={"predict_partial_charges": True}),
            dataset_info=dataset_info,
        ),
        PROPS,
        seed=0,
    ).replace_inference_context(InferenceContext(charge=0))

    atoms = water
    atoms.calc = _calculator(atoms, force_field)

    assert np.isfinite(atoms.get_potential_energy())
    assert np.all(np.isfinite(atoms.get_forces()))
