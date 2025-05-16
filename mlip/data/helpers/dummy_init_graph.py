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
import jraph
import numpy as np

from mlip.typing import GraphEdges, GraphGlobals, GraphNodes


def get_dummy_graph_for_model_init() -> jraph.GraphsTuple:
    """Generates a simple dummy graph that can be used for model initialization.

    Returns:
        The dummy graph.
    """
    return jraph.GraphsTuple(
        nodes=GraphNodes(
            positions=np.zeros((1, 3)),
            forces=np.zeros((1, 3)),
            species=np.array([0]),
        ),
        edges=GraphEdges(shifts=np.zeros((1, 3)), displ_fun=None),
        globals=jax.tree.map(
            lambda x: x[None, ...],
            GraphGlobals(
                cell=np.zeros((3, 3)),
                energy=np.array(0.0),
                stress=np.zeros((3, 3)),
                weight=np.asarray(1.0),
            ),
        ),
        receivers=np.array([0]),
        senders=np.array([0]),
        n_edge=np.array([1]),
        n_node=np.array([1]),
    )
