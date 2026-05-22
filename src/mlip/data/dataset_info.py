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

import numpy as np
import pydantic
from ase import Atom
from pydantic import field_validator, model_validator

from mlip.data import GraphDatasetBuilderConfig
from mlip.data.helpers.atomic_energies import compute_average_e0s_from_graphs
from mlip.data.helpers.neighbor_analysis import (
    compute_avg_min_neighbor_distance,
    compute_avg_num_neighbors,
)
from mlip.graph import Graph

logger = logging.getLogger("mlip")


class DatasetInfo(pydantic.BaseModel):
    """Information computed from the dataset that is required by the models.

    Only the per-dataset-identity fields (`dataset_name` and
    `atomic_energies_map`) accept a list form — one entry per dataset — in
    multi-dataset mode (`GraphDatasetBuilder` in `MULTI` mode). Every other
    statistic stays scalar: it is either required to match across datasets
    (`graph_cutoff_angstrom`, `long_range_cutoff_angstrom`), aggregated
    before being stored (`num_graphs`, `avg_num_neighbors`,
    `avg_r_min_angstrom`, `total_charge_set`), or inherited from the
    pretrained entry on the fine-tuning path.

    Attributes:
        dataset_name: Name of the dataset, or names of the datasets in multi-dataset
            settings. Defaults to `None`.
        num_graphs: Total number of graphs in the dataset used to compute the
            statistics below (summed across datasets in multi-dataset mode).
            Defaults to None.
        atomic_energies_map: Mapping from atomic number to average atomic energy.
            When using multiple datasets, this is a list of such mappings.
        graph_cutoff_angstrom: Graph cutoff distance in Ångström used to build
            the neighbor lists. Must match across datasets in multi-dataset mode.
        total_charge_set: Set of total charge values supported by the dataset;
            union across datasets in multi-dataset mode. Defaults to `None`.
        long_range_cutoff_angstrom: Long range cutoff distance in Ångström used to
            build the long range neighbor lists. Defaults to `None`, meaning no long
            range graph will be built, preventing any long range interactions
            computations.
        avg_num_neighbors: Mean number of neighbors per atom, weighted by
            `num_graphs` across datasets in multi-dataset mode. Defaults to `1.0`.
        avg_r_min_angstrom: Mean of the per-structure minimum edge distances
            in Ångström, weighted across datasets. `None` when not computed.
        scaling_mean: Mean used for energy rescaling. Defaults to `0.0`.
        scaling_stdev: Standard deviation used for energy rescaling.
            Defaults to `1.0`.
        atomic_energies_removed: Whether the atomic energies were subtracted
                                 from the dataset(s) by the building process. This
                                 information is required by the training loop class
                                 to adjust the force field settings accordingly.
                                 Default is `False`.
    """

    dataset_name: str | list[str] | None = None
    num_graphs: int | None = None
    atomic_energies_map: dict[int, float] | list[dict[int, float]]
    total_charge_set: set[int] | None = None
    graph_cutoff_angstrom: float
    long_range_cutoff_angstrom: float | None = None
    avg_num_neighbors: float = 1.0
    avg_r_min_angstrom: float | None = None
    scaling_mean: float = 0.0
    scaling_stdev: float = 1.0
    atomic_energies_removed: bool = False

    @property
    def allowed_atomic_numbers(self) -> list[int]:
        """List of sorted atomic numbers supported by the dataset."""
        if isinstance(self.atomic_energies_map, list):
            all_keys: set[int] = set()
            for m in self.atomic_energies_map:
                all_keys.update(m.keys())
            return sorted(all_keys)
        return sorted(self.atomic_energies_map.keys())

    @property
    def available_total_charges(self) -> list[int]:
        """List of sorted total charge values supported by the dataset."""
        if self.total_charge_set is None:
            return []
        return sorted(set(self.total_charge_set))

    @field_validator("atomic_energies_map")
    @classmethod
    def validate_atomic_numbers(
        cls, v: dict[int, float] | list[dict[int, float]]
    ) -> dict[int, float] | list[dict[int, float]]:
        maps = v if isinstance(v, list) else [v]
        for dataset_map in maps:
            for z in dataset_map.keys():
                try:
                    Atom(z).symbol
                except IndexError:
                    raise ValueError("%s is not a valid atomic number" % z)
        return v

    @model_validator(mode="after")
    def _validate_list_consistency(self) -> "DatasetInfo":
        """In multi-dataset mode both `atomic_energies_map` and (if provided)
        `dataset_name` must be lists of the same length. Mixing scalar and
        list forms is rejected."""
        map_is_list = isinstance(self.atomic_energies_map, list)
        name_is_list = isinstance(self.dataset_name, list)
        if map_is_list and self.dataset_name is not None and not name_is_list:
            raise ValueError(
                "`atomic_energies_map` is a list but `dataset_name` is a "
                "scalar — fields must be all lists or all scalars."
            )
        if name_is_list and not map_is_list:
            raise ValueError(
                "`dataset_name` is a list but `atomic_energies_map` is a "
                "scalar — fields must be all lists or all scalars."
            )
        if (
            map_is_list
            and name_is_list
            and len(self.atomic_energies_map) != len(self.dataset_name)
        ):
            raise ValueError(
                f"`atomic_energies_map` (len {len(self.atomic_energies_map)}) "
                f"and `dataset_name` (len {len(self.dataset_name)}) must have "
                "the same length."
            )
        # Reject duplicate dataset names — `InferenceContext.resolve` maps a
        # name to the first matching index, so duplicates would make later
        # heads unreachable by name.
        if name_is_list and len(set(self.dataset_name)) != len(self.dataset_name):
            raise ValueError(
                f"`dataset_name` contains duplicates ({self.dataset_name!r}); "
                "each entry must be unique so InferenceContext can resolve "
                "names to distinct heads."
            )
        return self

    def __str__(self):
        if isinstance(self.atomic_energies_map, dict):
            atomic_energies_map_with_symbols = {
                Atom(num).symbol: value
                for num, value in self.atomic_energies_map.items()
            }
        else:
            atomic_energies_map_with_symbols = [
                {Atom(num).symbol: value for num, value in single_map.items()}
                for single_map in self.atomic_energies_map
            ]
        return (
            f"Atomic Energies: {atomic_energies_map_with_symbols}, "
            f"Total Charges Supported: {self.total_charge_set}, "
            f"Graph Cutoff (Å): {self.graph_cutoff_angstrom}, "
            f"Long Range Cutoff (Å): {self.long_range_cutoff_angstrom}, "
            f"Avg Num Neighbors: {self.avg_num_neighbors}, "
            f"Avg R Min (Å): {self.avg_r_min_angstrom}, "
            f"Scaling Mean: {self.scaling_mean}, "
            f"Scaling Stdev: {self.scaling_stdev}, "
            f"Atomic Energies Removed: {self.atomic_energies_removed}"
        )


def compute_dataset_info_from_graphs(
    graphs: list[Graph],
    graph_cutoff_angstrom: float,
    avg_num_neighbors: float | None = None,
    avg_r_min_angstrom: float | None = None,
    long_range_cutoff_angstrom: float | None = None,
) -> DatasetInfo:
    """Computes the dataset info from graphs, typically training set graphs.

    Args:
        graphs: The graphs.
        graph_cutoff_angstrom: The graph distance cutoff in Angstrom to
                                  store in the dataset info.
        avg_num_neighbors: The optionally pre-computed average number of neighbors. If
                           provided, we skip recomputing this.
        avg_r_min_angstrom: The optionally pre-computed average miminum radius. If
                            provided, we skip recomputing this.
        long_range_cutoff_angstrom: The long range distance cutoff in Angstrom to
                                    store in the dataset info. If None, long range
                                    interactions are not computed.

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

    atomic_energies_map = compute_average_e0s_from_graphs(graphs)

    total_charge_set = set()
    for graph in graphs:
        if graph.globals.charge is not None:
            total_charge_set.update(np.asarray(graph.globals.charge).astype(int))
    if len(total_charge_set) == 0:
        total_charge_set = None

    logger.debug(
        "Computation of average atomic energies"
        " and dataset statistics completed in %.2f seconds.",
        time.perf_counter() - start_time,
    )

    return DatasetInfo(
        num_graphs=len(graphs),
        atomic_energies_map=atomic_energies_map,
        graph_cutoff_angstrom=graph_cutoff_angstrom,
        avg_num_neighbors=avg_num_neighbors,
        avg_r_min_angstrom=avg_r_min_angstrom,
        scaling_mean=0.0,
        scaling_stdev=1.0,
        long_range_cutoff_angstrom=long_range_cutoff_angstrom,
        total_charge_set=total_charge_set,
    )


def check_compatibility_of_ds_info(
    builder_config: GraphDatasetBuilderConfig, dataset_info: DatasetInfo
) -> None:
    """Check that a preset `DatasetInfo` is consistent with the builder config."""
    if builder_config.graph_cutoff_angstrom != dataset_info.graph_cutoff_angstrom:
        raise ValueError(
            "Got inconsistent cutoff distances: "
            "the builder config and the preset dataset_info have different "
            "cutoff values. Either create a fresh dataset_info or fix "
            "the config to match the trained model."
        )
