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
import time
from typing import Optional

import jraph
import pydantic
from ase import Atom

from mlip.data.helpers.atomic_energies import compute_average_e0s_from_graphs
from mlip.data.helpers.atomic_number_table import AtomicNumberTable
from mlip.data.helpers.neighbor_analysis import (
    compute_avg_min_neighbor_distance,
    compute_avg_num_neighbors,
)

logger = logging.getLogger("mlip")


class DatasetInfo(pydantic.BaseModel):
    """Pydantic dataclass holding information computed from the dataset that is
    (potentially) required by the models.

    Attributes:
        atomic_energies_map: A dictionary mapping the atomic numbers to the
                             computed average atomic energies for that element.
        cutoff_distance_angstrom: The graph cutoff distance that was
                          used in the dataset in Angstrom.
        avg_num_neighbors: The mean number of neighbors an atom has in the dataset.
        avg_r_min_angstrom: The mean minimum edge distance for a structure in the
                            dataset.
        scaling_mean: The mean used for the rescaling of the dataset values, the
                      default being 0.0.
        scaling_stdev: The standard deviation used for the rescaling of the dataset
                       values, the default being 1.0.
    """

    atomic_energies_map: dict[int, float]
    cutoff_distance_angstrom: float
    avg_num_neighbors: float = 1.0
    avg_r_min_angstrom: Optional[float] = None
    scaling_mean: float = 0.0
    scaling_stdev: float = 1.0

    def __str__(self):
        atomic_energies_map_with_symbols = {
            Atom(num).symbol: value for num, value in self.atomic_energies_map.items()
        }
        return (
            f"Atomic Energies: {atomic_energies_map_with_symbols}, "
            f"Avg. num. neighbors: {self.avg_num_neighbors:.2f}, "
            f"Avg. r_min: {self.avg_r_min_angstrom:.2f}, "
            f"Graph cutoff distance: {self.cutoff_distance_angstrom}"
        )


def compute_dataset_info_from_graphs(
    graphs: list[jraph.GraphsTuple],
    cutoff_distance_angstrom: float,
    z_table: AtomicNumberTable,
    avg_num_neighbors: Optional[float] = None,
    avg_r_min_angstrom: Optional[float] = None,
) -> DatasetInfo:
    """Computes the dataset info from graphs, typically training set graphs.

    Args:
        graphs: The graphs.
        cutoff_distance_angstrom: The graph distance cutoff in Angstrom to
                                  store in the dataset info.
        z_table: The atomic numbers table needed to produce the correct atomic energies
                 map keys.
        avg_num_neighbors: The optionally pre-computed average number of neighbors. If
                           provided, we skip recomputing this.
        avg_r_min_angstrom: The optionally pre-computed average miminum radius. If
                            provided, we skip recomputing this.


    Returns:
        The dataset info object populated with the computed data.
    """
    start_time = time.perf_counter()
    logger.info(
        "Starting to compute mandatory dataset statistics: this may take some time..."
    )
    if avg_num_neighbors is None:
        logger.debug("Computing average number of neighbors...")
        avg_num_neighbors = compute_avg_num_neighbors(graphs)
        logger.debug("Average number of neighbors: %.1f", avg_num_neighbors)
    if avg_r_min_angstrom is None:
        logger.debug("Computing average min neighbor distance...")
        avg_r_min_angstrom = compute_avg_min_neighbor_distance(graphs)
        logger.debug("Average min. node distance (Angstrom): %.1f", avg_r_min_angstrom)

    atomic_energies_map = {
        z_table.index_to_z(idx): energy
        for idx, energy in compute_average_e0s_from_graphs(graphs).items()
    }

    logger.debug(
        "Computation of average atomic energies"
        " and dataset statistics completed in %.2f seconds.",
        time.perf_counter() - start_time,
    )

    return DatasetInfo(
        atomic_energies_map=atomic_energies_map,
        cutoff_distance_angstrom=cutoff_distance_angstrom,
        avg_num_neighbors=avg_num_neighbors,
        avg_r_min_angstrom=avg_r_min_angstrom,
        scaling_mean=0.0,
        scaling_stdev=1.0,
    )
