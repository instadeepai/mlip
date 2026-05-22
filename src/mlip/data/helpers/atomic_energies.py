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

import numpy as np

from mlip.graph.graph import AtomicNumbers, Graph

logger = logging.getLogger("mlip")


def compute_average_e0s_from_graphs(
    graphs: list[Graph],
) -> dict[int, float]:
    """Compute average energy contribution of each element by least squares.

    Args:
        graphs: The graphs for which to compute the average energy
                contribution of each element.

    Returns:
        A dictionary mapping atomic number to the average energy contribution
        of that element.
    """
    num_graphs = len(graphs)
    unique_atomic_numbers = sorted(
        set(np.concatenate([g.nodes.atomic_numbers for g in graphs]))
    )
    num_unique = len(unique_atomic_numbers)

    element_count = np.zeros((num_graphs, num_unique))
    energies = np.zeros(num_graphs)

    for i in range(num_graphs):
        energies[i] = np.asarray(graphs[i].globals.energy).item()
        for j, z in enumerate(unique_atomic_numbers):
            element_count[i, j] = np.count_nonzero(graphs[i].nodes.atomic_numbers == z)

    try:
        e0s = np.linalg.lstsq(element_count, energies, rcond=1e-8)[0]
        atomic_energies = {z: e0s[i] for i, z in enumerate(unique_atomic_numbers)}

    except np.linalg.LinAlgError:
        logger.warning(
            "Failed to compute E0s using "
            "least squares regression, using the 0.0 for all atoms."
        )
        atomic_energies = dict.fromkeys(unique_atomic_numbers, 0.0)

    return atomic_energies


def _convert_energy_to_formation_energy(
    energy: float, atomic_numbers: AtomicNumbers, atomic_energies_map: dict[int, float]
) -> float:
    """Converts an energy to a formation energy by subtracting the
    sum of atomic energies for that system.
    """
    sum_atomic_energies = sum(
        atomic_energies_map.get(int(z), 0.0) for z in atomic_numbers
    )
    return energy - sum_atomic_energies


def remove_e0s_from_graphs(
    graphs: list[Graph], atomic_energies_map: dict[int, float]
) -> list[Graph]:
    """Removes the atomic energies from a list of graphs.

    Important note: This function just assumes atomic energies of zero for elements
    in the graphs that are not in the map. Hence, it does not fail in this case.
    The compatibility check of a dataset with a dataset info is happening elsewhere in
    the dataset processing.

    Args:
        graphs: The list of graphs.
        atomic_energies_map: The dictionary mapping atomic numbers to their atomic
                             energies.

    Returns:
        The list of graphs with updated `graph.globals.energy` fields.
    """
    return [
        g.replace_globals(
            energy=_convert_energy_to_formation_energy(
                g.globals.energy, g.nodes.atomic_numbers, atomic_energies_map
            )
        )
        for g in graphs
    ]
