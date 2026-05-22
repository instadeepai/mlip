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

from jax import Array

from mlip.graph import Graph
from mlip.models.charge_utils import compute_long_range_interactions
from mlip.utils.jax_utils import scatter_sum


def standard_energy_computation_head(
    graph: Graph, deterministic: bool = False
) -> Array:
    """Sums per-node 'energy' features in the graph and returns the total energy.

    Args:
        graph: The graph output by the MLIP network, containing node 'energy' features.
        deterministic: Whether to use a deterministic implementation of segment_sum.
            If True, ensures deterministic energy outputs at the cost of speed.

    Returns:
        Array of total energies, one for each graph in the batch.
    """
    graph_energies = scatter_sum(
        graph.nodes.features["energy"],
        num_elements_per_segment=graph.n_node,
        deterministic=deterministic,
    )
    return graph_energies


def coulomb_energy_computation_head(graph: Graph, deterministic: bool = False) -> Array:
    """Computes the energy prediction of the model by adding the long range interactions
    to the standard energy.

    Args:
        graph: The graph output by the MLIP network, containing node 'energy' features.
        deterministic: Whether to use a deterministic implementation of segment_sum.
            If True, ensures deterministic energy outputs at the cost of speed.

    Returns:
        Array of total energies, one for each graph in the batch.
    """
    standard_energies = standard_energy_computation_head(graph, deterministic)
    long_range_interactions = compute_long_range_interactions(graph)

    # Accumulate electrostatic interactions by graph
    long_range_energies = scatter_sum(
        long_range_interactions,
        num_elements_per_segment=graph.n_edge_long_range,
        deterministic=deterministic,
    )
    return standard_energies + long_range_energies
