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

from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes


def get_dummy_graph_for_model_init() -> Graph:
    """Generates a simple dummy graph that can be used for model initialization.

    Returns:
        The dummy graph.
    """
    graph = Graph(
        nodes=GraphNodes(
            positions=np.zeros((1, 3)),
            forces=np.zeros((1, 3)),
            atomic_numbers=np.array([1]),
            partial_charges=np.array([0.0]),
        ),
        edges=GraphEdges(shifts=np.zeros((1, 3)), displ_fun=None),
        globals=jax.tree.map(
            lambda x: x[None, ...],
            GraphGlobals(
                cell=np.zeros((3, 3)),
                energy=np.array(0.0),
                stress=np.zeros((3, 3)),
                weight=np.asarray(1.0),
                charge=np.asarray(0, dtype=np.int32),
                spin_multiplicity=np.asarray(1, dtype=np.int32),
                dataset_idx=np.asarray(0, dtype=np.int32),
                is_dummy_for_init=np.asarray(True, dtype=bool),
                dipole_moment=np.zeros((3)),
            ),
        ),
        receivers=np.array([0]),
        senders=np.array([0]),
        n_edge=np.array([1]),
        n_node=np.array([1]),
        n_edge_long_range=np.array([1]),
        senders_long_range=np.array([0]),
        receivers_long_range=np.array([0]),
        edges_long_range=GraphEdges(shifts=np.zeros((1, 3)), displ_fun=None),
    )
    return graph
