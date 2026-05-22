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
import numpy as np

from mlip.graph.batching_helpers import pad_with_graphs


def test_models_perform_node_masking_correctly(
    setup_system,
    mace_force_field,
    visnet_force_field,
    nequip_force_field,
    esen_force_field,
):
    graph = pad_with_graphs(setup_system[1], 15, 70, 2)
    models = [
        mace_force_field,
        visnet_force_field,
        nequip_force_field,
        esen_force_field,
    ]

    assert list(graph.n_node) == [10, 5]

    for model in models:
        result = jax.jit(model)(graph)
        real_forces = result.forces[:10]
        padded_forces = result.forces[10:]
        assert np.all((real_forces != 0.0) & ~np.isnan(real_forces))
        assert np.all(padded_forces == 0.0)
