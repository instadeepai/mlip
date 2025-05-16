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

import jax
import jraph
import numpy as np
from tqdm_loggable.auto import tqdm

from mlip.data.helpers.edge_vectors import get_edge_relative_vectors

LOG_PROPORTION = 10

logger = logging.getLogger("mlip")


def compute_avg_num_neighbors(graphs: list[jraph.GraphsTuple]) -> float:
    """Computes the averages number of neighbors for a given list of graphs.

    Args:
        graphs: The list of graphs to process.

    Returns:
        The average (i.e., mean) number of neighbors.
    """
    num_neighbors = []
    assert len(graphs) > 0

    for graph in tqdm(graphs, desc="Average number of neighbors computation"):
        _, counts = np.unique(graph.receivers, return_counts=True)
        num_neighbors.append(counts)

    return np.mean(np.concatenate(num_neighbors)).item()


def compute_avg_min_neighbor_distance(graphs: list[jraph.GraphsTuple]) -> float:
    """Computes the average minimum neighbor distance for a given list of graphs.

    Args:
        graphs: The list of graphs to process.

    Returns:
        The average (i.e., mean) minimum neighbor distance.
    """
    min_neighbor_distances = []
    assert len(graphs) > 0
    log_interval = max(1, len(graphs) // LOG_PROPORTION)

    for i, graph in enumerate(graphs):
        jit_get_edge_relative_vectors = jax.jit(get_edge_relative_vectors)
        vectors = jit_get_edge_relative_vectors(
            graph.nodes.positions,
            graph.senders,
            graph.receivers,
            graph.edges.shifts,
            graph.globals.cell,
            graph.n_edge,
        )
        length = np.linalg.norm(vectors, axis=-1)
        min_neighbor_distances.append(length.min())

        if (i + 1) % log_interval == 0 or i == len(graphs) - 1:
            percentage = ((i + 1) / len(graphs)) * 100
            logger.info("Processed %.0f%% of data", percentage)

    return np.mean(min_neighbor_distances).item()
