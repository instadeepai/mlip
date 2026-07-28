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
import numpy as np
from tqdm_loggable.auto import tqdm

from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)
from mlip.data.configs import GraphDatasetBuilderConfig
from mlip.data.dataset_info import (
    DatasetInfo,
    check_compatibility_of_ds_info,
    compute_dataset_info_from_graphs,
)
from mlip.data.graph_dataset import GraphDataset
from mlip.data.helpers.atomic_energies import remove_e0s_from_graphs
from mlip.data.helpers.data_prefetching import (
    PrefetchIterator,
    wrap_dataset_with_prefetch,
)
from mlip.data.helpers.exceptions import DatasetsHaveNotBeenProcessedError
from mlip.data.helpers.filtering_utils import (
    filter_systems_by_allowed_elements,
    filter_systems_by_allowed_total_charges,
    filter_systems_by_excluded_elements,
    filter_systems_by_excluded_total_charges,
    filter_systems_without_partial_charges,
    set_system_none_charges_to_zero,
)
from mlip.data.streaming_graph_dataset import StreamingGraphDataset
from mlip.data.chemical_systems_readers.chemical_systems_dataset import (
    map_dataset_from_readers,
    filter_excluded_indices,
)
from mlip.data.helpers.streaming_scan import (
    scan_chemical_map_dataset,
)
from mlip.data.helpers.type_aliases import (
    GraphPostProcessingFunction,
    SystemsPreprocessingFunction,
)
from mlip.graph import Graph
from mlip.utils.multihost import create_device_mesh

logger = logging.getLogger("mlip")


class SingleGraphDatasetBuilder:
    """Builds a single :class:`GraphDataset` from one or more readers.

    Handles loading chemical systems, converting them to graphs,
    auto-filling batch dimensions, and optionally computing
    :class:`DatasetInfo`.

    When ``builder_config.keep_in_memory`` is ``False``, returns a
    :class:`~mlip.data.streaming_graph_dataset.StreamingGraphDataset`
    that creates graphs on demand via Grain and packs them with dynamic batching.
    """

    def __init__(
        self,
        readers: ChemicalSystemsReader | list[ChemicalSystemsReader],
        builder_config: GraphDatasetBuilderConfig,
        dataset_info: DatasetInfo | bool,
        shuffle: bool = False,
    ):
        """Constructor.

        Args:
            readers: The data reader(s) that load a dataset into
                     :class:`~mlip.data.chemical_system.ChemicalSystem`
                     dataclasses.
            builder_config: The pydantic config.
            dataset_info: Pass `True` to compute dataset info from the graphs.
                          Pass `False` to skip dataset info computation.
                          Pass a :class:`DatasetInfo` instance to use a
                          pre-computed one (e.g. from a trained model).
        """
        self._dataset = None
        self._readers = readers if isinstance(readers, list) else [readers]
        self._builder_config = builder_config
        self._dataset_info = dataset_info
        self._shuffle = shuffle
        self._graphs: list[Graph] | None = None

        if not dataset_info and self._builder_config.use_formation_energies:
            raise ValueError(
                "Cannot use formation energies if dataset info is neither "
                "requested nor provided."
            )

        if isinstance(dataset_info, DatasetInfo):
            check_compatibility_of_ds_info(self._builder_config, dataset_info)

    def _preprocessing_fns_from_config(self) -> list[SystemsPreprocessingFunction]:
        """Return preprocessing functions implied by the builder config flags.

        Config-derived functions are applied before any caller-supplied
        `systems_preprocessing` so that normalization (e.g. setting None
        charges to zero) and filtering happen on the raw loaded systems.
        """
        fns: list[SystemsPreprocessingFunction] = []
        if self._builder_config.set_none_charges_to_zero:
            # Apply this before filtering by charges.
            fns.append(set_system_none_charges_to_zero)
        if self._builder_config.remove_systems_without_partial_charges:
            fns.append(filter_systems_without_partial_charges)
        if self._builder_config.allowed_atomic_numbers is not None:
            fns.append(
                filter_systems_by_allowed_elements(
                    self._builder_config.allowed_atomic_numbers
                )
            )
        if self._builder_config.allowed_charges is not None:
            fns.append(
                filter_systems_by_allowed_total_charges(
                    self._builder_config.allowed_charges
                )
            )
        if self._builder_config.excluded_atomic_numbers is not None:
            fns.append(
                filter_systems_by_excluded_elements(
                    self._builder_config.excluded_atomic_numbers
                )
            )
        if self._builder_config.excluded_charges is not None:
            fns.append(
                filter_systems_by_excluded_total_charges(
                    self._builder_config.excluded_charges
                )
            )
        return fns

    def _prepare_graphs(
        self, systems_preprocessing: list[SystemsPreprocessingFunction]
    ) -> None:
        """Load chemical systems from all readers, apply preprocessing, and
        convert them to :class:`Graph` objects.

        Args:
            systems_preprocessing: A list of functions applied sequentially to
                the loaded chemical systems after config-derived preprocessing.
        """
        systems = []
        for reader in self._readers:
            systems.extend(reader.load())

        all_preprocessing = self._preprocessing_fns_from_config() + list(
            systems_preprocessing or []
        )
        for preprocess_function in all_preprocessing:
            systems = preprocess_function(systems)

        self._graphs = [
            Graph.from_chemical_system(
                chemical_system=system,
                graph_cutoff_angstrom=self._builder_config.graph_cutoff_angstrom,
                long_range_cutoff_angstrom=self._builder_config.long_range_cutoff_angstrom,
            )
            for system in tqdm(systems, desc="graph creation")
        ]

    def get_dataset(
        self,
        prefetch: bool = False,
        mesh: jax.sharding.Mesh | None = None,
        systems_preprocessing: list[SystemsPreprocessingFunction] | None = None,
        graph_postprocessing: list[GraphPostProcessingFunction] | None = None,
    ) -> GraphDataset | StreamingGraphDataset | PrefetchIterator:
        """Build and return the dataset.

        Loads systems, converts to graphs, builds a :class:`GraphDataset`
        (or a streaming Grain-backed dataset), optionally computes
        :class:`DatasetInfo`, and wraps in a prefetch iterator if requested.

        Args:
            prefetch: Whether to wrap the dataset in a :class:`PrefetchIterator`. By
                default, this is set to `False`.
            mesh: Device mesh for data parallelism. If `None` and
                `prefetch` is `True`, a default mesh is created.
            systems_preprocessing: Optional list of functions applied
                sequentially to the loaded chemical systems.
            graph_postprocessing: Optional list of batch-level post-processing
                functions passed to :class:`GraphDataset`.

        Returns:
            A :class:`GraphDataset`, :class:`StreamingGraphDataset`, or
            :class:`PrefetchIterator`.
        """
        if self._builder_config.keep_in_memory:
            self._build_dataset(
                systems_preprocessing=systems_preprocessing,
                graph_postprocessing=graph_postprocessing,
            )
        else:
            self._build_streaming_dataset(
                systems_preprocessing=systems_preprocessing,
                graph_postprocessing=graph_postprocessing,
            )

        if prefetch:
            if mesh is None:
                mesh = create_device_mesh()

            return wrap_dataset_with_prefetch(
                dataset=self._dataset,
                mesh=mesh,
                num_batch_prefetch_host=self._builder_config.num_batch_prefetch_host,
                num_batch_prefetch_device=self._builder_config.num_batch_prefetch_device,
            )

        return self._dataset

    def _build_dataset(
        self,
        systems_preprocessing: list[SystemsPreprocessingFunction] | None,
        graph_postprocessing: list[GraphPostProcessingFunction] | None,
    ):
        self._prepare_graphs(systems_preprocessing=systems_preprocessing)
        max_n_node, max_n_edge, max_n_edge_long_range = (
            self._determine_autofill_batch_dimensions(self._graphs)
        )
        self._dataset = GraphDataset(
            graphs=self._graphs,
            max_n_node=max_n_node,
            max_n_edge=max_n_edge,
            max_n_edge_long_range=max_n_edge_long_range,
            batch_size=self._builder_config.batch_size,
            shuffle=self._shuffle,
            graph_postprocessing=graph_postprocessing,
            homogenize=self._builder_config.homogenize,
        )

        if (
            self._dataset_info is True
        ):  # And not preset which would also evaluate to True
            self._dataset_info = compute_dataset_info_from_graphs(
                graphs=self._graphs,
                graph_cutoff_angstrom=self._builder_config.graph_cutoff_angstrom,
                long_range_cutoff_angstrom=self._builder_config.long_range_cutoff_angstrom,
                avg_num_neighbors=self._builder_config.avg_num_neighbors,
                avg_r_min_angstrom=self._builder_config.avg_r_min_angstrom,
            )

        if self._builder_config.use_formation_energies:
            self._dataset.graphs = remove_e0s_from_graphs(
                self._dataset.graphs, self._dataset_info.atomic_energies_map
            )
            self._dataset_info = self._dataset_info.model_copy(
                update={"atomic_energies_removed": True}
            )

    def _build_streaming_dataset(
        self,
        systems_preprocessing: list[SystemsPreprocessingFunction] | None,
        graph_postprocessing: list[GraphPostProcessingFunction] | None,
    ) -> StreamingGraphDataset | PrefetchIterator:
        """Build a Grain-backed streaming dataset with dynamic batching."""
        if self._builder_config.homogenize:
            raise NotImplementedError(
                "homogenize=True is not supported with keep_in_memory=False yet."
            )

        all_preprocessing = self._preprocessing_fns_from_config() + list(
            systems_preprocessing or []
        )
        chemical_dataset = map_dataset_from_readers(self._readers)

        # Always scan (or load cache): needed for exclude_ids even when max_n_* are set.
        batching_info = scan_chemical_map_dataset(
            chemical_dataset,
            self._readers,
            graph_cutoff_angstrom=self._builder_config.graph_cutoff_angstrom,
            long_range_cutoff_angstrom=self._builder_config.long_range_cutoff_angstrom,
            preprocessing_fns=all_preprocessing,
            batch_size=self._builder_config.batch_size,
            max_n_node=self._builder_config.max_n_node,
            max_n_edge=self._builder_config.max_n_edge,
            max_n_edge_long_range=self._builder_config.max_n_edge_long_range,
            cache_dir=self._builder_config.batching_cache_dir,
            num_workers=self._builder_config.num_workers,
        )
        if self._dataset_info is True:
            self._dataset_info = batching_info.dataset_info
            updates: dict[str, float] = {}
            if self._builder_config.avg_num_neighbors is not None:
                updates["avg_num_neighbors"] = self._builder_config.avg_num_neighbors
            if self._builder_config.avg_r_min_angstrom is not None:
                updates["avg_r_min_angstrom"] = self._builder_config.avg_r_min_angstrom
            if updates:
                self._dataset_info = self._dataset_info.model_copy(update=updates)

        chemical_dataset = filter_excluded_indices(
            chemical_dataset, batching_info.exclude_ids
        )

        max_n_node = batching_info.max_n_node
        max_n_edge = batching_info.max_n_edge
        max_n_edge_long_range = batching_info.max_n_edge_long_range

        atomic_energies_map = None
        if self._builder_config.use_formation_energies:
            atomic_energies_map = self._dataset_info.atomic_energies_map
            self._dataset_info = self._dataset_info.model_copy(
                update={"atomic_energies_removed": True}
            )

        self._dataset = StreamingGraphDataset(
            chemical_dataset,
            batch_size=self._builder_config.batch_size,
            max_n_node=max_n_node,
            max_n_edge=max_n_edge,
            max_n_edge_long_range=max_n_edge_long_range,
            graph_cutoff_angstrom=self._builder_config.graph_cutoff_angstrom,
            long_range_cutoff_angstrom=self._builder_config.long_range_cutoff_angstrom,
            preprocessing_fns=all_preprocessing,
            atomic_energies_map=atomic_energies_map,
            shuffle=self._shuffle,
            graph_postprocessing=graph_postprocessing,
            worker_count=self._builder_config.num_workers,
            num_nodes=batching_info.num_nodes,
            num_batches=batching_info.num_batches,
            # exclude_ids already applied; leftover drops should error, and
            # mid-epoch resume can slice without draining reads.
            raise_exc_if_graphs_discarded=True,
        )

    @property
    def dataset_info(self) -> DatasetInfo | None:
        """The computed or preset :class:`DatasetInfo`, or `None` if
        dataset info computation was disabled.

        Returns immediately when dataset info was disabled (`False`) or
        was provided as a preset — in those cases :meth:`get_dataset`
        does not need to have been called.

        Raises:
            DatasetsHaveNotBeenProcessedError: If dataset info computation
                was requested (`dataset_info=True`) but :meth:`get_dataset`
                has not been called yet.
        """
        if self._dataset_info is False:
            return None
        if isinstance(self._dataset_info, DatasetInfo):
            return self._dataset_info
        raise DatasetsHaveNotBeenProcessedError(
            "Dataset info not available yet. Run get_dataset() first."
        )

    @staticmethod
    def _get_median_and_max_n_node(graphs: list[Graph]) -> tuple[int, int]:
        """Return `(median_n_node, max_n_node)` across all graphs."""
        num_atoms = [graph.n_node[0] for graph in graphs]
        return int(np.ceil(np.median(num_atoms))), max(num_atoms)

    @staticmethod
    def _get_median_num_neighbors_and_max_total_edges(
        graphs: list[Graph],
    ) -> tuple[int, int]:
        """Return `(median_num_neighbors, max_total_edges)` across all graphs."""
        num_neighbors = []
        current_max = 0

        for graph in graphs:
            counts = np.bincount(np.asarray(graph.receivers))
            current_max = max(current_max, counts.sum())
            num_neighbors.append(counts)

        median = int(np.ceil(np.median(np.concatenate(num_neighbors)).item()))
        return median, current_max

    @staticmethod
    def _get_median_num_neighbors_and_max_total_edges_long_range(
        graphs: list[Graph],
    ) -> tuple[int, int]:
        """Return `(median_num_neighbors_long_range, max_total_edges_long_range)`
        across all graphs for long range interactions,
        """
        num_neighbors = []
        current_max = 0

        for graph in graphs:
            counts = np.bincount(np.asarray(graph.receivers_long_range))
            current_max = max(current_max, counts.sum())
            num_neighbors.append(counts)

        median = int(np.ceil(np.median(np.concatenate(num_neighbors)).item()))
        return median, current_max

    @staticmethod
    def _find_max_n_node(graphs: list[Graph], batch_size: int) -> int:
        """Compute `max_n_node`, and resize if largest graph exceeds capacity.

        `max_n_node` is computed as the median number of nodes across all graphs, which
        is then multiplied by `batch_size` to determine the number of nodes returned in
        a batch (+1 for padding). If this estimate isn't sufficient to fit the largest
        graph, the value of `max_n_node` is adjusted to make it sufficient.
        """
        max_n_node, max_num_atoms = (
            SingleGraphDatasetBuilder._get_median_and_max_n_node(graphs)
        )
        if batch_size * max_n_node < max_num_atoms:
            logger.debug("Largest graph does not fit into batch -> resizing it.")
            max_n_node = int(np.ceil(max_num_atoms / batch_size))

        return max_n_node

    @staticmethod
    def _find_max_n_edge(graphs: list[Graph], max_n_node: int, batch_size: int) -> int:
        """Compute `max_n_edge`, and resize if largest graph exceeds capacity.

        `max_n_edge` is computed as ceil(median_neighbors_per_node * max_n_node / 2),
        giving an estimate of the typical edge count for a graph of max_n_node atoms.
        This is multiplied by batch_size * 2 to determine the total edge capacity of a
        batch (+1 for padding). If this estimate isn't sufficient to fit the largest
        graph, the value of `max_n_edge` is adjusted to make it sufficient.
        """
        median_n_nei_per_node, max_total_edges = (
            SingleGraphDatasetBuilder._get_median_num_neighbors_and_max_total_edges(
                graphs
            )
        )
        max_n_edge = int(np.ceil(median_n_nei_per_node * max_n_node / 2))

        if max_n_edge * batch_size * 2 < max_total_edges:
            logger.debug("Largest graph does not fit into batch -> resizing it.")
            max_n_edge = int(np.ceil(max_total_edges / (2 * batch_size)))

        return max_n_edge

    @staticmethod
    def _find_max_n_edge_long_range(
        graphs: list[Graph], max_n_node: int, batch_size: int
    ) -> int:
        """Compute `max_n_edge_long_range`, resizing if the largest graph
        exceeds the batch capacity.
        """
        median_n_nei_long_range, max_total_edges_long_range = (
            SingleGraphDatasetBuilder._get_median_num_neighbors_and_max_total_edges_long_range(
                graphs
            )
        )
        max_n_edge_long_range = median_n_nei_long_range * max_n_node // 2

        if max_n_edge_long_range * batch_size * 2 < max_total_edges_long_range:
            logger.debug(
                "Largest long range graph does not fit into batch -> resizing it."
            )
            max_n_edge_long_range = int(
                np.ceil(max_total_edges_long_range / (2 * batch_size))
            )

        return max_n_edge_long_range

    def _determine_autofill_batch_dimensions(
        self, graphs: list[Graph]
    ) -> tuple[int, int, int | None]:
        """Find good `max_n_node` and `max_n_edge` parameters from graphs."""
        return self._determine_autofill_batch_dimensions_static(
            graphs, self._builder_config
        )

    @staticmethod
    def _determine_autofill_batch_dimensions_static(
        graphs: list[Graph], config: GraphDatasetBuilderConfig
    ) -> tuple[int, int, int | None]:
        """Find good `max_n_node` and `max_n_edge` parameters from graphs."""
        # Autofill max_n_node to either:
        #   - median of n_node
        #   - max(n_node) / batch_size, if above too small for largest graph
        if config.max_n_node is None:
            max_n_node = SingleGraphDatasetBuilder._find_max_n_node(
                graphs, config.batch_size
            )
            logger.info(
                "The batching parameter max_n_node has been computed to be %s.",
                max_n_node,
            )
        else:
            max_n_node = config.max_n_node

        # Autofill max_n_edge to either:
        #   - num_neighbours * max_n_node
        #   - max(n_edge) / (2 * batch_size)
        if config.max_n_edge is None:
            max_n_edge = SingleGraphDatasetBuilder._find_max_n_edge(
                graphs, max_n_node, config.batch_size
            )
            logger.info(
                "The batching parameter max_n_edge has been computed to be %s.",
                max_n_edge,
            )
        else:
            max_n_edge = config.max_n_edge

        # Same logic is applied to compute max_n_edge_long_range when required
        if config.long_range_cutoff_angstrom is not None:
            if config.max_n_edge_long_range is None:
                max_n_edge_long_range = (
                    SingleGraphDatasetBuilder._find_max_n_edge_long_range(
                        graphs, max_n_node, config.batch_size
                    )
                )
                logger.info(
                    (
                        "The batching parameter max_n_edge_long_range has been computed"
                        " to be %s."
                    ),
                    max_n_edge_long_range,
                )
            else:
                max_n_edge_long_range = config.max_n_edge_long_range
        else:
            max_n_edge_long_range = None

        return max_n_node, max_n_edge, max_n_edge_long_range
