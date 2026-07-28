# Copyright 2025 Zhongguancun Academy
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
import os

import grain
import numpy as np
import pydantic
from tqdm_loggable.auto import tqdm

from mlip.data.chemical_systems_readers.chemical_systems_reader import (
    ChemicalSystemsReader,
)
from mlip.data.dataset_info import DatasetInfo
from mlip.data.helpers.streaming_scan_utils import (
    count_dynamic_batches,
    CachingHelper,
    Histogram,
    IncrementalE0s,
)
from mlip.data.helpers.type_aliases import SystemsPreprocessingFunction
from mlip.graph import Graph

logger = logging.getLogger("mlip")


class BatchingInfo(pydantic.BaseModel):
    """Batch-dimension limits, counts, excluded indices, and dataset stats."""

    max_n_node: int
    max_n_edge: int
    max_n_edge_long_range: int | None = None
    num_nodes: int
    num_batches: int
    exclude_ids: list[int] = pydantic.Field(default_factory=list)
    dataset_info: DatasetInfo


def scan_chemical_map_dataset(
    chemical_dataset: grain.MapDataset,
    readers: list[ChemicalSystemsReader] | ChemicalSystemsReader,
    *,
    graph_cutoff_angstrom: float,
    long_range_cutoff_angstrom: float | None,
    preprocessing_fns: list[SystemsPreprocessingFunction] | None,
    batch_size: int,
    max_n_node: int | None,
    max_n_edge: int | None,
    max_n_edge_long_range: int | None,
    skip_last_batch: bool = False,
    cache_dir: str | os.PathLike | None = None,
    num_workers: int = 0,
) -> BatchingInfo:
    """Scan a Grain ``MapDataset`` of chemical systems once.

    Always compute DatasetInfo and caches real ``avg_num_neighbors`` /
    ``avg_r_min_angstrom`` inside ``BatchingInfo.dataset_info`` even if
    ``dataset_info`` is not True.
    """
    caching_helper = CachingHelper(
        cache_dir,
        readers,
        batch_size,
        graph_cutoff_angstrom,
        long_range_cutoff_angstrom,
        max_n_node,
        max_n_edge,
        max_n_edge_long_range,
    )

    cached = caching_helper.load(BatchingInfo)
    if cached is not None:
        return cached

    iter_ds = chemical_dataset.to_iter_dataset(
        grain.ReadOptions(num_threads=0, prefetch_buffer_size=1)
    )
    if num_workers > 0:
        iter_ds = iter_ds.mp_prefetch(
            grain.MultiprocessingOptions(
                num_workers=num_workers, per_worker_buffer_size=128
            )
        )

    preprocessing_fns = list(preprocessing_fns or [])
    n_node_cap = batch_size * max_n_node if max_n_node is not None else None
    n_edge_cap = batch_size * max_n_edge * 2 if max_n_edge is not None else None
    n_edge_lr_cap = (
        batch_size * max_n_edge_long_range * 2
        if max_n_edge_long_range is not None
        else None
    )

    # Per-graph scalars for autofill of unset dims (O(num_graphs)).
    n_nodes: list[int] = []
    n_edges: list[int] = []
    n_edges_lr: list[int] = []

    # Online statistics for other graph properties.
    neighbors_hist = Histogram()
    e0s_acc = IncrementalE0s()
    min_dist_sum = 0.0
    min_dist_count = 0
    total_charge_set: set[int] = set()
    exclude_ids: list[int] = []
    num_nodes_kept = 0
    discarded = 0

    for idx, system in tqdm(
        enumerate(iter_ds), desc="streaming dataset scan", total=len(chemical_dataset)
    ):
        systems = [system]
        for preprocess in preprocessing_fns:
            systems = preprocess(systems)
            if not systems:
                break
        if not systems:
            exclude_ids.append(idx)
            discarded += 1
            continue

        graph = Graph.from_chemical_system(
            chemical_system=systems[0],
            graph_cutoff_angstrom=graph_cutoff_angstrom,
            long_range_cutoff_angstrom=long_range_cutoff_angstrom,
        )
        n_node = int(np.sum(graph.n_node))
        n_edge = int(len(graph.senders))
        n_lr = (
            int(len(graph.senders_long_range))
            if long_range_cutoff_angstrom is not None
            else 0
        )

        oversized = (
            (n_node_cap is not None and n_node > n_node_cap)
            or (n_edge_cap is not None and n_edge > n_edge_cap)
            or (n_edge_lr_cap is not None and n_lr > n_edge_lr_cap)
        )
        if oversized:
            exclude_ids.append(idx)
            discarded += 1
            continue

        n_nodes.append(n_node)
        n_edges.append(n_edge)
        n_edges_lr.append(n_lr)

        num_nodes_kept += n_node

        _, counts = np.unique(np.asarray(graph.receivers), return_counts=True)
        neighbors_hist.add(counts)

        if n_edge > 0:
            vectors = graph.edge_vectors(use_np=True)
            min_dist_sum += float(np.linalg.norm(vectors, axis=-1).min())
            min_dist_count += 1

        if graph.globals.charge is not None:
            total_charge_set.update(np.asarray(graph.globals.charge).astype(int))

        e0s_acc.add(
            np.asarray(graph.nodes.atomic_numbers),
            float(np.asarray(graph.globals.energy).item()),
        )

    if not n_nodes:
        if discarded:
            raise ValueError(
                "All graphs were discarded due to preprocessing filters "
                "or size constraints."
            )
        raise ValueError("Streaming scan found no valid chemical systems.")

    if discarded:
        logger.warning(
            "Discarded %s graphs due to preprocessing filters or size constraints.",
            discarded,
        )

    resolved_max_n_node = max_n_node
    if resolved_max_n_node is None:
        resolved_max_n_node = int(np.ceil(np.median(n_nodes)))
        if batch_size * resolved_max_n_node < max(n_nodes):
            resolved_max_n_node = int(np.ceil(max(n_nodes) / batch_size))
        logger.info(
            "The batching parameter max_n_node has been computed to be %s.",
            resolved_max_n_node,
        )

    resolved_max_n_edge = max_n_edge
    if resolved_max_n_edge is None:
        median_nei = neighbors_hist.median()
        resolved_max_n_edge = int(np.ceil(median_nei * resolved_max_n_node / 2))
        if resolved_max_n_edge * batch_size * 2 < max(n_edges):
            resolved_max_n_edge = int(np.ceil(max(n_edges) / (2 * batch_size)))
        logger.info(
            "The batching parameter max_n_edge has been computed to be %s.",
            resolved_max_n_edge,
        )

    resolved_max_n_edge_lr = max_n_edge_long_range
    if long_range_cutoff_angstrom is not None:
        if resolved_max_n_edge_lr is None:
            max_lr = max(n_edges_lr) if n_edges_lr else 0
            resolved_max_n_edge_lr = max(1, int(np.ceil(max_lr / (2 * batch_size))))
            logger.info(
                "The batching parameter max_n_edge_long_range has been computed "
                "to be %s.",
                resolved_max_n_edge_lr,
            )
    else:
        resolved_max_n_edge_lr = None

    n_node_cap = batch_size * resolved_max_n_node
    n_edge_cap = batch_size * resolved_max_n_edge * 2
    n_graph_cap = batch_size + 1
    n_edge_lr_cap = (
        None
        if resolved_max_n_edge_lr is None
        else batch_size * resolved_max_n_edge_lr * 2
    )

    num_batches = count_dynamic_batches(
        n_nodes,
        n_edges,
        n_edges_lr,
        n_node=n_node_cap + 1,
        n_edge=n_edge_cap,
        n_graph=n_graph_cap,
        n_edge_long_range=n_edge_lr_cap,
        skip_last_batch=skip_last_batch,
    )

    measured_avg_num_neighbors = (
        neighbors_hist.mean() if neighbors_hist.total_count else 1.0
    )
    measured_avg_r_min = (
        float(min_dist_sum / min_dist_count) if min_dist_count > 0 else 0.0
    )

    exclude_ids = sorted(set(exclude_ids))
    batching_info = BatchingInfo(
        max_n_node=resolved_max_n_node,
        max_n_edge=resolved_max_n_edge,
        max_n_edge_long_range=resolved_max_n_edge_lr,
        num_nodes=num_nodes_kept,
        num_batches=num_batches,
        exclude_ids=exclude_ids,
        dataset_info=DatasetInfo(
            num_graphs=len(n_nodes),
            atomic_energies_map=e0s_acc.solve(),
            graph_cutoff_angstrom=graph_cutoff_angstrom,
            avg_num_neighbors=measured_avg_num_neighbors,
            avg_r_min_angstrom=measured_avg_r_min,
            scaling_mean=0.0,
            scaling_stdev=1.0,
            long_range_cutoff_angstrom=long_range_cutoff_angstrom,
            total_charge_set=total_charge_set or None,
        )
    )
    caching_helper.save(batching_info)

    return batching_info
