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

import functools
import logging
from typing import Optional, TypeAlias

import jax
import jraph
import numpy as np
from tqdm_loggable.auto import tqdm

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.chemical_systems_readers import CombinedReader
from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)
from mlip.data.chemical_systems_readers.utils import (
    filter_systems_with_unseen_atoms_and_assign_atomic_species,
)
from mlip.data.configs import GraphDatasetBuilderConfig
from mlip.data.dataset_info import DatasetInfo, compute_dataset_info_from_graphs
from mlip.data.helpers.atomic_number_table import AtomicNumberTable
from mlip.data.helpers.data_prefetching import (
    ParallelGraphDataset,
    PrefetchIterator,
    create_prefetch_iterator,
)
from mlip.data.helpers.graph_creation import create_graph_from_chemical_system
from mlip.data.helpers.graph_dataset import GraphDataset

GraphDatasetsOrPrefetchedIterators: TypeAlias = (
    tuple[GraphDataset, GraphDataset, GraphDataset]
    | tuple[PrefetchIterator, PrefetchIterator, PrefetchIterator]
)

logger = logging.getLogger("mlip")


class DatasetsHaveNotBeenProcessedError(Exception):
    """Exception to be raised if dataset info is not available yet."""


class DevicesNotProvidedForPrefetchingError(Exception):
    """Exception to be raised if devices are not provided
    even though prefetching of data was requested.
    """


class GraphDatasetBuilder:
    """Main class handling the construction and preprocessing of the graph dataset.

    The key idea is that a user provides a
    :class:`~mlip.data.chemical_systems_readers.chemical_systems_reader.ChemicalSystemsReader`
    subclass that loads a dataset from disk into
    :class:`~mlip.data.chemical_system.ChemicalSystem` dataclasses and then
    ``GraphDatasetBuilder`` converts these further to ``jraph`` graphs and the
    dataset info dataclass.
    """

    Config = GraphDatasetBuilderConfig

    def __init__(
        self,
        reader: ChemicalSystemsReader | CombinedReader,
        dataset_config: GraphDatasetBuilderConfig,
        dataset_info: DatasetInfo | None = None,
    ):
        """Constructor.

        Args:
            reader: The data reader that loads a dataset into
                         :class:`~mlip.data.chemical_system.ChemicalSystem`
                         dataclasses
            dataset_config: The pydantic config.
            dataset_info: Leave `None` to create initial training datasets.
                          Otherwise, pass the `.dataset_info` of a trained
                          `ForceField` for downstream tasks like batched
                          inference, finetuning, or distillation.

        Note:
            Evaluating a trained model on new data can lead to inconsistent results if:

            * a wrong `AtomicNumberTable` is used to map atomic numbers to specie
              indices,
            * a different `graph_cutoff_angstrom` is used to generate the graphs
              (to a lesser extent).

            Therefore, it is important to pass the `dataset_info` of a trained
            model to prepare a dataset of batched graphs in downstream tasks.
        """
        self._reader = reader
        self._config = dataset_config
        self._dataset_info: Optional[DatasetInfo] = dataset_info
        self._datasets: Optional[dict[str, Optional[GraphDataset]]] = None
        # Sanity check when DatasetInfo is passed from the outside
        cutoff = self._config.graph_cutoff_angstrom
        if dataset_info and cutoff != dataset_info.cutoff_distance_angstrom:
            raise ValueError(
                "GraphDatasetBuilder got inconsistent cutoff distances: "
                "pass `None` as dataset_info to create a fresh dataset, or fix "
                "dataset_config if you want to reuse the atomic number table "
                "of a trained model."
            )

    def prepare_datasets(self) -> None:
        """Prepares the datasets.

        This includes loading it into ChemicalSystem objects via the chemical
        systems reader, and then producing the graph datasets and the
        dataset info object.
        """
        if self._dataset_info is None:
            # Assign atomic species from observed set of elements
            train_systems, valid_systems, test_systems = self._reader.load()
            z_table = self._construct_z_table(train_systems)

        else:
            # Assign atomic species from prescribed z_table
            z_table = self._retrieve_z_table(self._dataset_info)
            post_process_fun = functools.partial(
                filter_systems_with_unseen_atoms_and_assign_atomic_species,
                z_table=z_table,
            )
            train_systems, valid_systems, test_systems = self._reader.load(
                post_process_fun
            )

        train_graph_dataset, valid_graph_dataset, test_graph_dataset = (
            self._create_graph_datasets_from_chemical_systems(
                train_systems, valid_systems, test_systems
            )
        )

        logger.debug(
            "Number of graphs in training set: %s", len(train_graph_dataset.graphs)
        )
        logger.debug(
            "Number of graphs in validation set: %s", len(valid_graph_dataset.graphs)
        )
        logger.debug("Number of graphs in test set: %s", len(test_graph_dataset.graphs))

        if self._dataset_info is None:
            self._dataset_info = compute_dataset_info_from_graphs(
                train_graph_dataset.graphs,
                self._config.graph_cutoff_angstrom,
                z_table,
                self._config.avg_num_neighbors,
                self._config.avg_r_min_angstrom,
            )

        self._datasets = {
            "train": train_graph_dataset,
            "valid": valid_graph_dataset,
            "test": test_graph_dataset,
        }

        if self._config.use_formation_energies:
            self._convert_energies_to_formation_energies(z_table)

    def get_splits(
        self, prefetch: bool = False, devices: Optional[list[jax.Device]] = None
    ) -> GraphDatasetsOrPrefetchedIterators:
        """Returns the training, validation, and test dataset splits.

        Args:
            prefetch: Whether to run the data prefetching and return PrefetchIterators.
            devices: Devices for parallel prefetching. Must be given if prefetch=True.

        Returns:
            A tuple of training, validation, and test datasets. If prefetch=False,
            these are of type GraphDataset, otherwise of type PrefetchIterator.
        """
        if self._datasets is None:
            raise DatasetsHaveNotBeenProcessedError(
                "Datasets are not available yet. Run prepare_datasets() first."
            )

        if prefetch:
            if devices is None:
                raise DevicesNotProvidedForPrefetchingError(
                    "Please provide the devices argument when prefetch=True."
                )
            return self._get_prefetched_iterators(devices)
        return (
            self._datasets["train"],
            self._datasets["valid"],
            self._datasets["test"],
        )

    @property
    def dataset_info(self) -> DatasetInfo:
        """Getter for the dataset info.

        Will raise exception if dataset info not available yet.
        """
        if self._dataset_info is None:
            raise DatasetsHaveNotBeenProcessedError(
                "Dataset info not available yet. Run prepare_datasets() first."
            )
        return self._dataset_info

    @staticmethod
    def _filter_out_bad_graphs(
        graphs: list[jraph.GraphsTuple],
    ) -> list[jraph.GraphsTuple]:
        """Filter out graphs. This function currently only removes
        empty graphs.

        Args:
            graphs: the list of graphs.

        Returns:
            The filtered sublist of graphs
        """

        def filter_empty_graphs(
            graphs: list[jraph.GraphsTuple],
        ) -> list[jraph.GraphsTuple]:
            filtered_graphs, num_discarded_graphs = [], 0
            for graph in graphs:
                if graph.n_edge.sum() == 0:
                    num_discarded_graphs += 1
                else:
                    filtered_graphs.append(graph)
            if num_discarded_graphs > 0:
                logger.warning(
                    "Discarded %s empty graphs due to having no edges",
                    num_discarded_graphs,
                )
            return filtered_graphs

        graphs = filter_empty_graphs(graphs)
        return graphs

    def _create_graph_datasets_from_chemical_systems(
        self,
        train_systems: list[ChemicalSystem],
        valid_systems: list[ChemicalSystem],
        test_systems: list[ChemicalSystem],
    ) -> tuple[GraphDataset, GraphDataset, GraphDataset]:
        _cfg = self._config
        graph_datasets = {}

        # Train graphs will be returned as None if not calculated below
        max_n_node, max_n_edge, train_graphs = (
            self._determine_autofill_batch_limitations(train_systems)
        )

        for key, systems, should_shuffle in [
            ("train", train_systems, True),
            ("valid", valid_systems, False),
            ("test", test_systems, False),
        ]:
            if key == "train" and train_graphs is not None:
                graphs = train_graphs  # here: train graphs have been computed above
            else:
                graphs = [
                    create_graph_from_chemical_system(
                        system, _cfg.graph_cutoff_angstrom
                    )
                    for system in tqdm(systems, desc=f"{key} graph creation")
                ]
                graphs = self._filter_out_bad_graphs(graphs)

            graph_dataset = GraphDataset(
                graphs=graphs,
                max_n_node=max_n_node,
                max_n_edge=max_n_edge,
                batch_size=_cfg.batch_size,
                should_shuffle=should_shuffle,
            )
            graph_datasets[key] = graph_dataset

        return (
            graph_datasets["train"],
            graph_datasets["valid"],
            graph_datasets["test"],
        )

    @staticmethod
    def _retrieve_z_table(dataset_info: DatasetInfo) -> AtomicNumberTable:
        """Reconstruct the `AtomicNumberTable` of the given `dataset_info`."""
        return AtomicNumberTable(sorted(dataset_info.atomic_energies_map.keys()))

    @staticmethod
    def _construct_z_table(train_systems: list[ChemicalSystem]) -> AtomicNumberTable:
        """Construct a fresh `AtomicNumberTable` from list of train systems."""
        return AtomicNumberTable(
            sorted(
                set(np.concatenate([system.atomic_numbers for system in train_systems]))
            )
        )

    def _get_prefetched_iterators(
        self, devices: list[jax.Device]
    ) -> tuple[PrefetchIterator, PrefetchIterator, PrefetchIterator]:
        _cfg = self._config
        num_devices = len(devices)

        device_shard_fn = functools.partial(
            jax.tree.map,
            lambda x: jax.device_put_sharded(list(x), devices),
        )

        prefetched_iterators = {}

        for key, dataset in self._datasets.items():
            parallel_dataset = ParallelGraphDataset(dataset, num_devices)
            prefetched_iterator = create_prefetch_iterator(
                create_prefetch_iterator(
                    parallel_dataset,
                    prefetch_count=_cfg.num_batch_prefetch,
                ),
                prefetch_count=_cfg.batch_prefetch_num_devices,
                preprocess_fn=device_shard_fn,
            )
            prefetched_iterators[key] = prefetched_iterator

        return (
            prefetched_iterators["train"],
            prefetched_iterators["valid"],
            prefetched_iterators["test"],
        )

    def _determine_autofill_batch_limitations(
        self, train_systems: list[ChemicalSystem]
    ) -> tuple[int, int, Optional[list[jraph.GraphsTuple]]]:
        _cfg = self._config

        # Autofill max_n_node and max_n_edge if they are set to None
        if _cfg.max_n_node is None:
            max_n_node, max_num_atoms = self._get_median_and_max_num_atoms(
                train_systems
            )
            if _cfg.batch_size * max_n_node < max_num_atoms:
                logger.debug("Largest graph does not fit into batch -> resizing it.")
                max_n_node = int(np.ceil(max_num_atoms / _cfg.batch_size))

            logger.debug(
                "The batching parameter max_n_node has been computed to be %s.",
                max_n_node,
            )
        else:
            max_n_node = _cfg.max_n_node

        if _cfg.max_n_edge is None:
            train_graphs, num_discarded_graphs = [], 0
            for system in tqdm(train_systems, desc="Graph creation"):
                graph = create_graph_from_chemical_system(
                    system, _cfg.graph_cutoff_angstrom
                )
                if graph.n_edge.sum() == 0:
                    num_discarded_graphs += 1
                else:
                    train_graphs.append(graph)
            if num_discarded_graphs > 0:
                logger.warning(
                    "Discarded %s empty graphs due to having no edges",
                    num_discarded_graphs,
                )
            median_n_nei, max_total_edges = (
                self._get_median_num_neighbors_and_max_total_edges(train_graphs)
            )
            max_n_edge = median_n_nei * max_n_node // 2

            if max_n_edge * _cfg.batch_size * 2 < max_total_edges:
                logger.debug("Largest graph does not fit into batch -> resizing it.")
                max_n_edge = int(np.ceil(max_total_edges / (2 * _cfg.batch_size)))

            logger.debug(
                "The batching parameter max_n_edge has been computed to be %s.",
                max_n_edge,
            )
        else:
            train_graphs = None
            max_n_edge = _cfg.max_n_edge

        return max_n_node, max_n_edge, train_graphs

    @staticmethod
    def _get_median_and_max_num_atoms(
        chemical_systems: list[ChemicalSystem],
    ) -> tuple[int, int]:
        num_atoms = [system.atomic_numbers.shape[0] for system in chemical_systems]
        return int(np.ceil(np.median(num_atoms))), max(num_atoms)

    @staticmethod
    def _get_median_num_neighbors_and_max_total_edges(
        graphs: list[jraph.GraphsTuple],
    ) -> tuple[int, int]:
        num_neighbors = []
        current_max = 0

        for graph in graphs:
            _, counts = np.unique(graph.receivers, return_counts=True)
            current_max = max(current_max, counts.sum())
            num_neighbors.append(counts)

        median = int(np.ceil(np.median(np.concatenate(num_neighbors)).item()))
        return median, current_max

    def _convert_energies_to_formation_energies(
        self, z_table: AtomicNumberTable
    ) -> None:
        for dataset in self._datasets.values():
            dataset.graphs = [
                self._convert_energy_to_formation_energy(graph, z_table)
                for graph in dataset.graphs
            ]

    def _convert_energy_to_formation_energy(
        self, graph: jraph.GraphsTuple, z_table: AtomicNumberTable
    ) -> jraph.GraphsTuple:
        sum_atomic_energies = sum(
            self.dataset_info.atomic_energies_map.get(z_table.index_to_z(key), 0.0)
            for key in graph.nodes.species
        )
        formation_energy = graph.globals.energy - np.array(sum_atomic_energies)
        return graph._replace(globals=graph.globals._replace(energy=formation_energy))
