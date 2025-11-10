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

import jraph
import numpy as np

logger = logging.getLogger("mlip")


def compute_average_e0s_from_graphs(
    graphs: list[jraph.GraphsTuple],
) -> dict[int, float]:
    """Compute average energy contribution of each element by least squares.

    Args:
        graphs: The graphs for which to compute the average energy
                contribution of each element

    Returns:
        The atomic energies dictionary which is the mapping of atomic species to
        the average energy contribution of each element.
    """
    num_graphs = len(graphs)
    unique_species = sorted(set(np.concatenate([g.nodes.species for g in graphs])))
    num_unique_species = len(unique_species)

    species_count = np.zeros((num_graphs, num_unique_species))
    energies = np.zeros(num_graphs)

    for i in range(num_graphs):
        energies[i] = graphs[i].globals.energy
        for j, species_number in enumerate(unique_species):
            species_count[i, j] = np.count_nonzero(
                graphs[i].nodes.species == species_number
            )

    try:
        e0s = np.linalg.lstsq(species_count, energies, rcond=1e-8)[0]
        atomic_energies = {}
        for i, species_number in enumerate(unique_species):
            atomic_energies[species_number] = e0s[i]

    except np.linalg.LinAlgError:
        logger.warning(
            "Failed to compute E0s using "
            "least squares regression, using the 0.0 for all atoms."
        )
        atomic_energies = dict.fromkeys(unique_species, 0.0)

    return atomic_energies
