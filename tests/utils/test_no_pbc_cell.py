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
from matscipy.neighbours import neighbour_list

from mlip.utils.no_pbc_cell import get_no_pbc_cell


def test_no_pbc_cell_does_not_have_shifts(setup_system_and_mace_model) -> None:
    atoms, _, _, model_ff = setup_system_and_mace_model
    graph_cutoff_angstrom = model_ff.cutoff_distance

    no_pbc_cell, no_pbc_cell_origin = get_no_pbc_cell(
        atoms.positions, graph_cutoff_angstrom
    )

    senders, shifts = neighbour_list(
        quantities="iS",
        cell=no_pbc_cell,
        cell_origin=no_pbc_cell_origin,
        pbc=np.array([False, False, False]),
        positions=atoms.positions,
        cutoff=graph_cutoff_angstrom,
    )
    assert np.all(shifts == 0.0)
    assert len(senders) == 68
