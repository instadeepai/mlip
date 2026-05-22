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
from collections import defaultdict
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np

from mlip.data.configs import GraphDatasetBuilderConfig
from mlip.data.dataset_info import DatasetInfo, check_compatibility_of_ds_info
from mlip.data.graph_dataset import GraphDataset
from mlip.data.helpers.atomic_energies import remove_e0s_from_graphs
from mlip.data.helpers.data_prefetching import (
    PrefetchIterator,
    wrap_dataset_with_prefetch,
)
from mlip.data.helpers.exceptions import DatasetsHaveNotBeenProcessedError
from mlip.data.helpers.filtering_utils import (
    filter_systems_by_charges,
    filter_systems_by_elements,
)
from mlip.data.helpers.type_aliases import (
    FlatReadersDict,
    GraphPostProcessingFunction,
    NestedReadersDict,
    SystemsPreprocessingFunction,
)
from mlip.data.single_graph_dataset_builder import SingleGraphDatasetBuilder
from mlip.graph import Graph
from mlip.utils.multihost import create_device_mesh

logger = logging.getLogger("mlip")

TRAIN_SPLIT_KEY = "train"
REPLAY_DATASET_KEY = "replay"


class BuilderMode(Enum):
    """Available modes for :class:`GraphDatasetBuilder`.

    Attributes:
        CUSTOM: Expects a flat readers dict keyed by arbitrary split names,
            e.g. `{"split_a": reader, "split_b": reader}`. Computes dataset
            info independently for each split, unless a preset dataset info
            is provided.
        TRAINING: Expects a flat readers dict that must contain a `"train"`
            key, e.g. `{"train": reader, "val": reader, "test": reader}`.
            Computes dataset info only from the `"train"` split.
        MULTI: Expects a nested readers dict keyed first by dataset name,
            then by split name, e.g.
            `{"oc20": {"train": reader, "val": reader}, "omat": {"train": reader}}`.
            Computes dataset info from each dataset's `"train"` split and
            combines them. If a `"replay"` dataset exists, it reuses a
            preset dataset info for that dataset.
    """

    CUSTOM = "custom"
    TRAINING = "training"
    MULTI = "multi"


class GraphDatasetBuilder:
    """Orchestrates building multiple dataset splits using
    :class:`SingleGraphDatasetBuilder`.

    Supports three modes (see :class:`BuilderMode`): `CUSTOM`,
    `TRAINING`, and `MULTI`, each differing in how dataset info is
    computed and how readers are organized.
    """

    Config = GraphDatasetBuilderConfig

    def __init__(
        self,
        readers: FlatReadersDict | NestedReadersDict,
        builder_config: GraphDatasetBuilderConfig,
        mode: BuilderMode | str = BuilderMode.TRAINING,
        dataset_info: DatasetInfo | None = None,
    ):
        """Constructor.

        Args:
            readers: A flat or nested dictionary of readers keyed by split
                name (e.g. `{"train": readers, "valid": readers}`) for
                `CUSTOM`/`TRAINING` modes, or by dataset name then split
                name for `MULTI` mode.
            builder_config: Configuration for graph construction, including
                batch size, cutoff distance, and batch dimension limits.
            mode: The build mode. Defaults to `BuilderMode.TRAINING`.
            dataset_info: An optional preset :class:`DatasetInfo`. Required
                for `MULTI` mode when a `"replay"` dataset is present.
        """
        self._dataset_info = None
        self.datasets = None
        self._readers = readers
        self._builder_config = builder_config
        if isinstance(mode, str):
            mode = BuilderMode(mode)
        self._mode = mode
        self._preset_dataset_info = dataset_info

        # Make sure the single builder does not take care of formation energies
        # as we take care of it in this class
        self._single_builder_config = self._builder_config.model_copy(
            update={"use_formation_energies": False}
        )

        if dataset_info is not None:
            check_compatibility_of_ds_info(self._builder_config, dataset_info)

        if mode == BuilderMode.CUSTOM:
            if not isinstance(readers, dict) or any(
                isinstance(v, dict) for v in readers.values()
            ):
                raise TypeError(
                    "CUSTOM mode expects a flat readers dict"
                    " (dict[str, ChemicalSystemsReader])."
                )
            if len(readers) == 0:
                raise ValueError("Readers dict must not be empty.")

        elif mode == BuilderMode.TRAINING:
            if not isinstance(readers, dict) or any(
                isinstance(v, dict) for v in readers.values()
            ):
                raise TypeError(
                    "TRAINING mode expects a flat readers dict"
                    " (dict[str, ChemicalSystemsReader])."
                )
            if TRAIN_SPLIT_KEY not in readers:
                raise ValueError("TRAINING mode requires a 'train' key in readers.")

        elif mode == BuilderMode.MULTI:
            if not all(isinstance(v, dict) for v in readers.values()):
                raise TypeError(
                    "MULTI mode expects a nested readers dict"
                    " (dict[str, dict[str, ChemicalSystemsReader]])."
                )
            if len(readers) < 2:
                raise ValueError("MULTI mode requires at least 2 dataset keys.")
            if REPLAY_DATASET_KEY in readers and dataset_info is None:
                raise ValueError(
                    "MULTI mode with a 'replay' dataset requires a preset dataset_info."
                )

    def get_datasets(
        self,
        prefetch: bool = False,
        mesh: jax.sharding.Mesh | None = None,
        systems_preprocessing: list[SystemsPreprocessingFunction] | None = None,
        graph_postprocessing: list[GraphPostProcessingFunction] | None = None,
    ) -> dict[str, GraphDataset | PrefetchIterator]:
        """Build all dataset splits according to the configured mode.

        Args:
            prefetch: Whether to wrap each dataset in a
                :class:`PrefetchIterator`. Default is `False`.
            mesh: Device mesh for data parallelism. If `None` and
                `prefetch` is `True`, a default mesh is created.
            systems_preprocessing: Optional list of functions applied
                sequentially to the loaded chemical systems.
            graph_postprocessing: Optional list of batch-level post-processing
                functions passed to :class:`GraphDataset`.

        Returns:
            A dictionary mapping split names to datasets or prefetch iterators.
            Also sets `self.dataset_info`.
        """

        self.datasets = {}

        if self._mode == BuilderMode.CUSTOM:
            self._handle_custom_build_mode(
                systems_preprocessing=systems_preprocessing,
                graph_postprocessing=graph_postprocessing,
            )

        elif self._mode == BuilderMode.TRAINING:
            self._handle_training_build_mode(
                systems_preprocessing=systems_preprocessing,
                graph_postprocessing=graph_postprocessing,
            )

        elif self._mode == BuilderMode.MULTI:
            self._handle_multi_build_mode(
                systems_preprocessing=systems_preprocessing,
                graph_postprocessing=graph_postprocessing,
            )

        # As the final step, wrap in prefetch iterators if required.
        if prefetch:
            if mesh is None:
                mesh = create_device_mesh()
            self.datasets = self._wrap_datasets_with_prefetch(self.datasets, mesh)

        return self.datasets

    @property
    def dataset_info(self) -> DatasetInfo | dict[str, DatasetInfo] | None:
        """The computed or preset :class:`DatasetInfo`, or `None` if
        dataset info computation was disabled.

        In `CUSTOM` mode with a preset :class:`DatasetInfo`, the post-build
        value equals the preset, so it is returned immediately without
        requiring :meth:`get_datasets` to have been called. Other modes
        derive the value from the graphs and still require a prior build.

        Raises:
            DatasetsHaveNotBeenProcessedError: If dataset info is not yet
                available because :meth:`get_datasets` has not been called.
        """
        if self._dataset_info is not None:
            return self._dataset_info
        if self._preset_dataset_info is not None and self._mode == BuilderMode.CUSTOM:
            return self._preset_dataset_info
        raise DatasetsHaveNotBeenProcessedError(
            "Dataset info not available yet. Run get_datasets() first."
        )

    def _handle_custom_build_mode(
        self,
        systems_preprocessing: list[SystemsPreprocessingFunction] | None = None,
        graph_postprocessing: list[GraphPostProcessingFunction] | None = None,
    ) -> None:
        self._dataset_info = {}
        compute_ds_info = self._preset_dataset_info is None

        for key, reader in self._readers.items():
            builder = SingleGraphDatasetBuilder(
                reader, self._single_builder_config, compute_ds_info
            )
            self.datasets[key] = builder.get_dataset(
                prefetch=False,
                systems_preprocessing=systems_preprocessing,
                graph_postprocessing=graph_postprocessing,
            )

            ds_info = (
                builder.dataset_info if compute_ds_info else self._preset_dataset_info
            )
            if self._builder_config.use_formation_energies:
                self.datasets[key], ds_info = (
                    self._remove_atomic_energies_and_update_ds_info(
                        self.datasets[key], ds_info
                    )
                )
            self._dataset_info[key] = ds_info

        if not compute_ds_info:
            self._dataset_info = list(self._dataset_info.values())[0]

    def _build_train_first_splits(
        self,
        readers: FlatReadersDict,
        systems_preprocessing: list[SystemsPreprocessingFunction] | None,
        graph_postprocessing: list[GraphPostProcessingFunction] | None,
        strict_unseen_atoms: bool = False,
    ) -> tuple[dict[str, GraphDataset], DatasetInfo]:
        """Build splits where the train split is processed first to obtain
        dataset info, and remaining splits filter out unseen elements.

        Args:
            readers: A flat readers dict that must contain a `"train"` key.
            systems_preprocessing: Optional preprocessing functions.
            graph_postprocessing: Optional batch-level post-processing functions.
            strict_unseen_atoms: If `True`, raise a `ValueError` when
                non-train splits contain systems with unseen atoms instead
                of silently filtering them out.

        Returns:
            A tuple of (datasets, dataset_info) where datasets is keyed by
            split name and dataset_info will be the `DatasetInfo` object
            computed for the 'train' split.
        """
        datasets: dict[str, GraphDataset] = {}

        # 1. Process the training split first to obtain dataset info.
        train_key = TRAIN_SPLIT_KEY
        train_builder = SingleGraphDatasetBuilder(
            readers=readers[train_key],
            builder_config=self._single_builder_config,
            dataset_info=True,
            shuffle=True,
        )
        datasets[train_key] = train_builder.get_dataset(
            prefetch=False,
            systems_preprocessing=systems_preprocessing,
            graph_postprocessing=graph_postprocessing,
        )

        # 2. Build remaining splits.
        allowed_zs = train_builder.dataset_info.allowed_atomic_numbers
        allowed_charges = train_builder.dataset_info.available_total_charges
        if strict_unseen_atoms:
            # In strict mode, don't filter — build as-is, then validate.
            other_preprocessing = systems_preprocessing
        else:
            unseen_elements_filter = filter_systems_by_elements(list(allowed_zs))
            unseen_filter = [unseen_elements_filter]
            if self._builder_config.ensure_no_unseen_total_charges:
                unseen_charge_filter = filter_systems_by_charges(list(allowed_charges))
                unseen_filter.append(unseen_charge_filter)
            other_preprocessing = list(systems_preprocessing or []) + unseen_filter

        for key, reader in readers.items():
            if key == train_key:
                continue
            builder = SingleGraphDatasetBuilder(
                readers=reader,
                builder_config=self._single_builder_config,
                dataset_info=False,
            )
            datasets[key] = builder.get_dataset(
                prefetch=False,
                systems_preprocessing=other_preprocessing,
                graph_postprocessing=graph_postprocessing,
            )

        if strict_unseen_atoms:
            self._validate_no_unseen_atoms(datasets, set(allowed_zs))
            if self._builder_config.ensure_no_unseen_total_charges:
                self._validate_no_unseen_total_charges(datasets, set(allowed_charges))

        ds_info = train_builder.dataset_info
        if self._builder_config.use_formation_energies:
            for key, _dataset in datasets.items():
                datasets[key], ds_info = (
                    self._remove_atomic_energies_and_update_ds_info(_dataset, ds_info)
                )

        return datasets, ds_info

    def _handle_training_build_mode(
        self,
        systems_preprocessing: list[SystemsPreprocessingFunction] | None = None,
        graph_postprocessing: list[GraphPostProcessingFunction] | None = None,
    ) -> None:
        self.datasets, self._dataset_info = self._build_train_first_splits(
            readers=self._readers,
            systems_preprocessing=systems_preprocessing,
            graph_postprocessing=graph_postprocessing,
        )

    def _handle_multi_build_mode(
        self,
        systems_preprocessing: list[SystemsPreprocessingFunction] | None = None,
        graph_postprocessing: list[GraphPostProcessingFunction] | None = None,
    ) -> None:
        """Build datasets in `MULTI` mode.

        Iterates over a nested readers dict, building each sub-split
        independently. Computes dataset info from each training split and
        combines them. Replay datasets reuse the preset dataset info.
        Finally, merges datasets sharing the same subsplit name and assigns
        dataset indices. Additionally, populates the `DatasetInfo` with the
        ordered dataset names.

        Args:
            systems_preprocessing: Optional preprocessing functions.
            graph_postprocessing: Optional batch-level post-processing functions.
        """
        readers: NestedReadersDict = self._readers  # type: ignore[assignment]
        dataset_infos: dict[str, DatasetInfo] = {}

        datasets: dict[str, dict[str, GraphDataset]] = defaultdict(dict)

        replay_key = REPLAY_DATASET_KEY
        train_key = TRAIN_SPLIT_KEY

        finetuning = replay_key in readers

        # Process replay first so its info is available for e0s backfill.
        if finetuning:
            preset_ds_info = self._preset_dataset_info
            allowed_zs = set(preset_ds_info.allowed_atomic_numbers)
            allowed_charges = set(preset_ds_info.available_total_charges)

            for split_name, subsplit_reader in readers[replay_key].items():
                is_training_split: bool = split_name == train_key
                builder = SingleGraphDatasetBuilder(
                    readers=subsplit_reader,
                    builder_config=self._single_builder_config,
                    dataset_info=False,
                    shuffle=is_training_split,
                )
                datasets[replay_key][split_name] = builder.get_dataset(
                    prefetch=False,
                    systems_preprocessing=systems_preprocessing,
                    graph_postprocessing=graph_postprocessing,
                )
                if self._builder_config.use_formation_energies:
                    datasets[replay_key][split_name], preset_ds_info = (
                        self._remove_atomic_energies_and_update_ds_info(
                            datasets[replay_key][split_name],
                            preset_ds_info,
                        )
                    )

            self._validate_no_unseen_atoms(datasets[replay_key], allowed_zs)
            if self._builder_config.ensure_no_unseen_total_charges:
                self._validate_no_unseen_total_charges(
                    self.datasets[replay_key], allowed_charges
                )
            if preset_ds_info.dataset_name is None:
                preset_ds_info = preset_ds_info.model_copy(
                    update={"dataset_name": replay_key}
                )
            dataset_infos[replay_key] = preset_ds_info

        # Process remaining (non-replay) datasets.
        for dataset_name, dataset_splits in readers.items():
            if dataset_name == replay_key:
                continue

            ds_datasets, ds_info = self._build_train_first_splits(
                readers=dataset_splits,
                systems_preprocessing=systems_preprocessing,
                graph_postprocessing=graph_postprocessing,
                strict_unseen_atoms=True,
            )
            datasets[dataset_name] = ds_datasets

            # Fill in missing atomic numbers from replay.
            update = {"dataset_name": dataset_name}
            if finetuning:
                update["atomic_energies_map"] = (
                    dataset_infos[replay_key].atomic_energies_map
                    | ds_info.atomic_energies_map
                )
            dataset_infos[dataset_name] = ds_info.model_copy(update=update)

        # Order dataset infos to match the replay-first ordering used by
        # _combine_datasets_and_set_indices (replay=0, others in insertion order).
        ds_keys = list(dataset_infos.keys())
        if replay_key in ds_keys:
            ds_keys.remove(replay_key)
            ds_keys = [replay_key] + ds_keys
        ordered_infos = [dataset_infos[k] for k in ds_keys]

        # Merge dataset infos into one.
        if finetuning:
            self._dataset_info = self._combine_dataset_infos_if_finetuning(
                ordered_infos
            )
        else:
            self._dataset_info = self._combine_dataset_infos_in_multi_case(
                ordered_infos
            )

        # Combine datasets with the same subsplit keys and set dataset indices.
        self.datasets = self._combine_datasets_and_set_indices(
            datasets,
            graph_postprocessing,
            homogenize=self._builder_config.homogenize,
        )

    @staticmethod
    def _combine_datasets_and_set_indices(
        datasets: dict[str, dict[str, GraphDataset]],
        graph_postprocessing: list[GraphPostProcessingFunction] | None = None,
        homogenize: bool = False,
    ) -> dict[str, GraphDataset]:
        """Combine datasets sharing the same subsplit key into one, adding
        dataset indices to each graph.

        Expects a nested dict keyed by dataset name then subsplit name,
        e.g. `{"replay": {"train": ds, "val": ds}, "oc20": {"train": ds}}`.
        The replay dataset always gets `dataset_idx=0`; other datasets
        are numbered starting from 1 in insertion order. The mapping from
        dataset name to index is implicit in the ordering of `dataset_name`
        in `DatasetInfo`.

        Returns:
            Merged datasets keyed by subsplit name (e.g. `"train"`, `"val"`).
        """
        # Establish ordering: replay first (dataset_idx=0), then the rest.
        replay_key = REPLAY_DATASET_KEY
        ds_keys = list(datasets.keys())
        if replay_key in ds_keys:
            ds_keys.remove(replay_key)
            ds_keys = [replay_key] + ds_keys
        ds_key_to_idx = {k: i for i, k in enumerate(ds_keys)}

        # Group graphs per subsplit, tagging each with the dataset index.
        graphs_per_subsplit: dict[str, list[Graph]] = {}
        for key, subsplits in datasets.items():
            dataset_idx = ds_key_to_idx[key]
            for subsplit_name, subsplit in subsplits.items():
                tagged_graphs = [
                    graph.replace_globals(
                        dataset_idx=jnp.full_like(
                            graph.globals.weight, dataset_idx, dtype=jnp.int32
                        )
                    )
                    for graph in subsplit.graphs
                ]
                graphs_per_subsplit.setdefault(subsplit_name, []).extend(tagged_graphs)

        # Build a merged GraphDataset for each subsplit.
        merged: dict[str, GraphDataset] = {}
        for subsplit_name, graphs in graphs_per_subsplit.items():
            # Use the max batch dimensions across the individual datasets.
            matching = [
                subdatasets[subsplit_name]
                for subdatasets in datasets.values()
                if subsplit_name in subdatasets
            ]
            filtered_n_edge_long_range = [
                ds.n_edge_long_range
                for ds in matching
                if ds.n_edge_long_range is not None
            ]
            if len(filtered_n_edge_long_range) == 0:
                merged_max_n_edge_long_range = None
            else:
                merged_max_n_edge_long_range = max(filtered_n_edge_long_range)

            merged[subsplit_name] = GraphDataset(
                graphs=graphs,
                max_n_node=max(ds.max_n_node for ds in matching),
                max_n_edge=max(ds.max_n_edge for ds in matching),
                max_n_edge_long_range=merged_max_n_edge_long_range,
                batch_size=matching[0].batch_size,
                shuffle=matching[0].shuffle,
                graph_postprocessing=graph_postprocessing,
                homogenize=homogenize,
            )

        return merged

    @staticmethod
    def _validate_no_unseen_atoms(
        datasets: dict[str, GraphDataset],
        allowed_zs: set[int],
    ) -> None:
        """Raise if any graph contains atoms not in the allowed set."""
        for split_name, dataset in datasets.items():
            for graph in dataset.graphs:
                graph_zs = set(graph.nodes.atomic_numbers.tolist())
                unseen = graph_zs - allowed_zs
                if unseen:
                    raise ValueError(
                        f"Split '{split_name}' contains unseen atomic "
                        f"numbers {sorted(unseen)} not present in the "
                        f"allowed set (allowed: {sorted(allowed_zs)})."
                    )

    @staticmethod
    def _validate_no_unseen_total_charges(
        datasets: dict[str, GraphDataset],
        available_total_charges: set[int],
    ) -> None:
        """Raise if any graph contains charge values not in the allowed set."""
        for split_name, dataset in datasets.items():
            for graph in dataset.graphs:
                if graph.globals.charge is not None:
                    graph_total_charges = set(
                        np.asarray(graph.globals.charge).astype(int).tolist()
                    )
                    unseen = graph_total_charges - available_total_charges
                    if unseen:
                        raise ValueError(
                            f"Split '{split_name}' contains unseen total "
                            f"charges {sorted(unseen)} not present in the "
                            f"allowed set (allowed: {sorted(available_total_charges)})."
                        )

    @staticmethod
    def _sum_num_graphs(dataset_infos: list[DatasetInfo]) -> int | None:
        """Sum `num_graphs` across dataset infos, skipping entries whose
        `num_graphs` is `None` (common in unit-test fixtures and in
        datasets built without a full scan). Returns `None` if no entry
        has `num_graphs` set."""
        counts = [ds.num_graphs for ds in dataset_infos if ds.num_graphs is not None]
        return sum(counts) if counts else None

    @staticmethod
    def _weighted_average_over_graphs(
        dataset_infos: list[DatasetInfo],
        statistic_name: str,
    ) -> float | None:
        """Weighted average of `ds.<statistic_name>` across dataset infos,
        weighted by each dataset's `num_graphs`. Datasets whose statistic
        is `None` are skipped (optional fields such as
        `avg_r_min_angstrom`). If any contributing dataset has
        `num_graphs=None` we fall back to equal weights rather than
        crash. Returns `None` if nothing remains to average.
        """
        contributions = [
            (ds.num_graphs, getattr(ds, statistic_name))
            for ds in dataset_infos
            if getattr(ds, statistic_name) is not None
        ]
        if not contributions:
            return None
        if any(count is None for count, _ in contributions):
            return sum(stat for _, stat in contributions) / len(contributions)
        total = sum(count for count, _ in contributions)
        return sum(count * stat for count, stat in contributions) / total

    @staticmethod
    def _merge_long_range_cutoffs(
        dataset_infos: list[DatasetInfo],
    ) -> float | None:
        """Return the combined `long_range_cutoff_angstrom` for a merged
        :class:`DatasetInfo`.

        Any dataset without a long-range cutoff (`None`) disables long-range
        interactions in the merged info — otherwise we'd silently claim a
        cutoff that isn't honoured by every subset. If all datasets set a
        cutoff we take the maximum so the merged radius covers every subset's
        neighbourhood.
        """
        cutoffs = [ds.long_range_cutoff_angstrom for ds in dataset_infos]
        if any(c is None for c in cutoffs):
            return None
        return max(cutoffs)

    @staticmethod
    def _merge_total_charge_sets(
        dataset_infos: list[DatasetInfo],
    ) -> set[int] | None:
        """Union `total_charge_set` across datasets, ignoring `None`
        entries. Returns `None` if every dataset has `None`.
        """
        charge_sets = [
            ds.total_charge_set
            for ds in dataset_infos
            if ds.total_charge_set is not None
        ]
        if not charge_sets:
            return None
        return set().union(*charge_sets)

    @classmethod
    def _combine_dataset_infos_in_multi_case(
        cls,
        dataset_infos: list[DatasetInfo],
    ) -> DatasetInfo:
        """Merge multiple :class:`DatasetInfo` instances into one
        with list-valued fields."""
        avg_num_neighbors = cls._weighted_average_over_graphs(
            dataset_infos, "avg_num_neighbors"
        )
        avg_r_min_angstrom = cls._weighted_average_over_graphs(
            dataset_infos, "avg_r_min_angstrom"
        )
        return DatasetInfo(
            dataset_name=[ds.dataset_name for ds in dataset_infos],
            num_graphs=cls._sum_num_graphs(dataset_infos),
            atomic_energies_map=[ds.atomic_energies_map for ds in dataset_infos],
            total_charge_set=cls._merge_total_charge_sets(dataset_infos),
            graph_cutoff_angstrom=dataset_infos[0].graph_cutoff_angstrom,
            long_range_cutoff_angstrom=cls._merge_long_range_cutoffs(dataset_infos),
            # `avg_num_neighbors` is non-optional on `DatasetInfo` (default
            # 1.0); fall back to that if the helper had nothing to work with.
            avg_num_neighbors=avg_num_neighbors
            if avg_num_neighbors is not None
            else 1.0,
            avg_r_min_angstrom=avg_r_min_angstrom,
            scaling_mean=dataset_infos[0].scaling_mean,
            scaling_stdev=dataset_infos[0].scaling_stdev,
            atomic_energies_removed=dataset_infos[0].atomic_energies_removed,
        )

    @classmethod
    def _combine_dataset_infos_if_finetuning(
        cls,
        dataset_infos: list[DatasetInfo],
    ) -> DatasetInfo:
        """Merge multiple :class:`DatasetInfo` instances into one,
        assuming the first corresponds to that of the pretraining
        dataset."""
        return dataset_infos[0].model_copy(
            update={
                "dataset_name": [ds.dataset_name for ds in dataset_infos],
                # `num_graphs` is a scalar on `DatasetInfo`, so sum across
                # datasets rather than emitting a list (which pydantic would
                # reject on reload).
                "num_graphs": cls._sum_num_graphs(dataset_infos),
                "atomic_energies_map": [ds.atomic_energies_map for ds in dataset_infos],
            }
        )

    def _wrap_datasets_with_prefetch(
        self,
        datasets: dict[str, GraphDataset],
        mesh: jax.sharding.Mesh,
    ) -> dict[str, PrefetchIterator]:
        """Wrap each GraphDataset in a PrefetchIterator."""
        wrapped = {}
        for dataset_name, dataset in datasets.items():
            wrapped[dataset_name] = wrap_dataset_with_prefetch(
                dataset=dataset,
                mesh=mesh,
                num_batch_prefetch_host=self._builder_config.num_batch_prefetch_host,
                num_batch_prefetch_device=self._builder_config.num_batch_prefetch_device,
            )
        return wrapped

    @staticmethod
    def _remove_atomic_energies_and_update_ds_info(
        dataset: GraphDataset, ds_info: DatasetInfo
    ) -> tuple[GraphDataset, DatasetInfo]:
        """Removes the atomic energies from a dataset and sets the
        `atomic_energies_removed` field of the dataset info to true.
        Then returns the updated dataset and dataset info again.
        """
        dataset.graphs = remove_e0s_from_graphs(
            dataset.graphs, ds_info.atomic_energies_map
        )
        ds_info = ds_info.model_copy(update={"atomic_energies_removed": True})
        return dataset, ds_info
