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

import jax.numpy as jnp
import numpy as np
from jax import Array

from mlip.data import ChemicalSystem
from mlip.data.graph_dataset_builder import (
    GraphPostProcessingFunction,
    SystemsPreprocessingFunction,
)
from mlip.graph import Graph

# Preprocessing of Hessian labels (ChemicalSystem stage, prior to batching).


def _pad_hessian_matrices(
    chemical_system: ChemicalSystem, max_system_size: int
) -> ChemicalSystem:
    """Pad full Hessian matrices on trailing axes so they can be batched.

    Args:
        chemical_system: A system with Hessian labels of shape `(n,3,n,3)`.
        max_system_size: Largest graph size.

    Returns:
        A chemical system with Hessian labels of shape `(n,3,N,3)`,
        where N is the largest system's size.
    """
    # padding all elements to max_n_node to enable batching
    atomic_numbers = chemical_system.atomic_numbers
    atomic_numbers = np.pad(
        atomic_numbers, ((0, max_system_size - len(atomic_numbers)))
    )  # [max_system_size]

    # teacher metrics are None in the dummy graph used in model initialization
    # or when training on DFT Hessians
    if (
        chemical_system.extras is not None
        and chemical_system.extras["teacher_error"] is None
    ):
        chemical_system["extras"] = np.array([0.0, 0.0])

    hessian = chemical_system.hessian
    padding = max_system_size - hessian.shape[2]
    chemical_system.hessian = np.pad(
        hessian, ((0, 0), (0, 0), (0, padding), (0, 0))
    )  # [n_node, 3, max_n_node, 3]

    return chemical_system


def pad_systems_hessians(systems: list[ChemicalSystem]) -> list[ChemicalSystem]:
    """Pad the Hessian of each system in the given systems list to `(n,3,N,3)`,
    where `n` is the number of atoms in the system, and `N` is the number of atoms
    in the largest system in the list.
    """
    max_system_size = (
        max(system.positions.shape[0] for system in systems) if systems else 0
    )
    return [_pad_hessian_matrices(system, max_system_size) for system in systems]


# Batch-processing of Hessian labels (Graph stage, on the fly).


def _sample_hessian_rows(
    hessian: Array,
    num_hessian_rows: int,
    n_node: Array,
    batch_size: int,
) -> tuple[Array, Array]:
    """Subsample the full Hessian matrix to extract Hessian rows.

    Args:
        hessian: batch of Hessian matrices, of shape `(n*3, N, 3)`,
            where `n` is the total number of graph nodes in the batch,
            obtained by `sum(graph.n_node)`,
            and `N` is the size of the largest graph in the dataset,
            given by `dataset_info.largest_system_size`.
        num_hessian_rows: number of rows in the Hessian sub-sample.
        n_node: list of graph sizes.

    Returns:
        Tuple containing subsampled Hessian of size `(G,R,N,3)`,
        where `G` is the total number of graphs `len(n_node)`.
        in the batch, and `R` is the number of Hessian rows per graph,
        alongside graph ranges and sampled row indices.
    """
    # force terms will be flattened into a force vector,
    # (n_nodes, 3) -> (n_nodes*3), graph ranges have to be
    # adapted accordingly .
    graph_ranges = jnp.append(0, jnp.cumsum(n_node) * 3)

    # [n_real_graph, num_hessian_rows]
    sampled_rows = np.random.randint(
        low=graph_ranges[:-1],
        high=graph_ranges[1:],
        size=(num_hessian_rows, len(n_node)),
        dtype=int,
    ).transpose()

    # Pad to guarantee `sample_rows` has constant shape (batch_size, R).
    sampled_rows = jnp.pad(sampled_rows, ((0, batch_size - len(n_node)), (0, 0)))

    # retrieve sample rows only
    # [n_graph, num_hessian_rows, max_n_node, 3]
    hessian = jnp.asarray(hessian[sampled_rows])

    return hessian, sampled_rows


def _concat_hessian_rows(
    hessian_samples: Array,
    n_node: Array,
    target_len: int,
) -> np.ndarray:
    """Removes padding terms from the batch of Hessian subsamples,
    then adds them again to ensure padding terms are all at the end.

    Args:
        hessian_samples: batch of Hessian samples of size `(G,R,N,3)`,
            where `G` is the number of graphs, `R` is the number of Hessian rows,
            and `N` is the maximum system size.
        n_node: list of graph sizes.
        target_len: used to pad the concatenated Hessian rows,
            calculated as `sum(graph.n_node)`,
            to ensure the Hessian label is always of the same shape,
            as required for jitting.

    Returns:
        A processed batch Hessian label of shape `(R, n, 3)`,
        where `n` is the total number of graph nodes.
    """
    batch_size = len(n_node)
    n_real_node = sum(n_node)

    # flip n_graph and n_rows dims
    # [num_hessian_rows, n_graph, max_system_size, 3]
    hessian_samples = hessian_samples.transpose(1, 0, 2, 3)

    # re-order to have all padding terms at the end
    # to match the order of the predicted Hessian.
    # then merge `n_graph` and `max_n_node` dims with
    # `np.concatenate` to match predicted Hessian shape:
    # [num_hessian_rows, n_graph*max_n_node, 3]
    joined_rows = [
        np.concatenate([row[i, : n_node[i]] for i in range(batch_size)])
        for row in hessian_samples
    ]
    hessian_label = np.stack(joined_rows)
    # max(0, diff) to avoid having negative values in pad()
    hessian_label = np.pad(
        hessian_label, ((0, 0), (0, max(0, target_len - n_real_node)), (0, 0))
    )

    return hessian_label


def process_graph_hessian(
    batched_graph: Graph,
    num_rows: int,
) -> Graph:
    """Process Hessian labels to match the format of Hessian predictions.

    First, rows are subsampled from graph-wise Hessians into a batch of shape
    `(R,G,N,3)` where `R = num_rows`, `G = n_graph` and `N = max_system_size`.
    Hessians are processed, cropped then permuted into shape `(n, R, 3)`,
    where `n` is maximum possible number of graph nodes in a batch, ensuring
    the processed Hessian label shape is static.

    Args:
        batched_graph: batch of graphs with full, padded Hessian matrices.
        num_rows: number of Hessian rows to be subsampled.

    Returns:
        A batched graph with processed subsampled Hessian labels.
    """
    padding_mask = batched_graph.graph_mask()
    batch_size = batched_graph.num_graphs
    n_node = batched_graph.n_node[padding_mask]
    n_real_node = sum(n_node)

    max_system_size = batched_graph.nodes.hessian.shape[-2]
    # crop padding terms and reshape from (n_node, 3, max_system_size, 3)
    # to (n_node * 3, max_system_size, 3)
    full_hessian = batched_graph.nodes.hessian[:n_real_node].reshape(
        n_real_node * 3, max_system_size, 3
    )
    # take sub-sample of size num_hessian_rows
    # [n_graph, num_hessian_rows, max_system_size, 3]
    hessian_samples, sampled_rows = _sample_hessian_rows(
        full_hessian, num_rows, n_node, batch_size
    )
    # concatenate Hessian rows along the position axis
    # and pad to match predicted Hessian shape [r, G*N, 3]
    # where `r` is the total number of Hessian rows, `G` number of graphs
    # and `N` is the `max_n_node` set in the dataset confing or automatically
    # computed to the median graph size.
    target_len, _ = batched_graph.nodes.positions.shape  # [G*N, 3]
    hessian_labels = _concat_hessian_rows(hessian_samples, n_node, target_len)

    # transpose to `(n, R, 3)` as the Hessian label is a node feature, where `n = G*N`
    hessian_labels = hessian_labels.transpose(1, 0, -1)

    batched_graph = batched_graph.replace_nodes(hessian=hessian_labels)
    batched_graph = batched_graph.replace_globals(sample_hessian_rows=sampled_rows)

    return batched_graph


def get_hessian_processing_functions() -> tuple[
    SystemsPreprocessingFunction, GraphPostProcessingFunction
]:
    """Return preprocessing and postprocessing functions for Hessian labels.

    The first function :func:`~mlip.data.helpers.hessian_utils.pad_systems_hessians`
    operates on chemical systems prior to graph construction (e.g., padding Hessians),
    while the second function,
    :func:`~mlip.data.helpers.hessian_utils.process_graph_hessian` processes Hessian
    labels after graph objects have been created.

    Returns:
        A tuple containing a systems-level preprocessing function (applied before
        graph creation) and a graph-level postprocessing function (applied after graph
        creation).
    """
    return pad_systems_hessians, process_graph_hessian


def request_all_hessian_rows_batched(batched_graph: Graph) -> Graph:
    """Set `sample_hessian_rows` to request the full Hessian for all graphs in a batch.

    Rather than jacrev operating on `total_padded_nodes*3` independent outputs, this
    mimics the subsampled-rows path used in `HessianPredictor`, such that jacrev
    differentiates `max_n_atoms*3` summed outputs instead, requiring less memory.

    R = max(n_atoms_per_graph) * 3. For graphs with fewer atoms, the extra row slots
    are redirected to a padding graph, so they contribute nothing to the Jacobian sum.

    Hessians computed with this setting of `sample_hessian_rows` will be of shape
    (total_nodes, R, 3).
    """
    padding_mask = batched_graph.graph_mask()
    batch_size = batched_graph.num_graphs
    n_node = batched_graph.n_node[padding_mask]

    R = int(n_node.max()) * 3
    graph_starts_3 = np.concatenate([[0], np.cumsum(n_node[:-1])]) * 3
    # Rows beyond the real atom count are redirected to padding graph
    padding_force_idx = int(n_node.sum()) * 3

    row_offsets = np.arange(R)[None, :]  # (1, R)
    valid = row_offsets < (n_node * 3)[:, None]  # (n_real_graphs, R)
    sampled_rows = np.where(
        valid, graph_starts_3[:, None] + row_offsets, padding_force_idx
    )
    # Pad to (batch_size, R) to include a slot for the padding graph
    sampled_rows = np.pad(sampled_rows, ((0, batch_size - len(n_node)), (0, 0)))

    return batched_graph.replace_globals(sample_hessian_rows=jnp.array(sampled_rows))


def single_graph_hessian_from_subsampled_batch(
    batch_hessian: Array, system_start: int, system_end: int
) -> Array:
    """Retrieve a single graph's Hessian from the subsampled-rows batch output.

    After `request_all_hessian_rows_batched`, `nodes.hessian` has shape
    (total_nodes, R, 3). We extract the per-system (n_atoms, 3, n_atoms, 3) Hessian.
    """
    n_atoms = system_end - system_start

    # (n_atoms, n_atoms*3, 3): local[col_i, r, col_j] = H_g[r//3, r%3, col_i, col_j]
    # Rows beyond n_atoms*3 are zero (padding forces) and are excluded with :n_atoms*3
    local = batch_hessian[system_start:system_end, : n_atoms * 3]

    # Split R=n_atoms*3 -> (n_atoms, 3): axes become (col_i, row_i, row_j, col_j)
    local = local.reshape(n_atoms, n_atoms, 3, 3)

    # Transpose to (row_i, row_j, col_i, col_j)
    return local.transpose(1, 2, 0, 3)
