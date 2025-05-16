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
from typing import Callable, Optional

import ase
import jax
import jraph
import numpy as np

from mlip.data import ChemicalSystem
from mlip.data.helpers import (
    AtomicNumberTable,
    GraphDataset,
    create_graph_from_chemical_system,
)
from mlip.models import ForceField
from mlip.typing import Prediction

logger = logging.getLogger("mlip")


def _run_inference_on_a_single_batch(
    jitted_force_field_fun: Callable[[jraph.GraphsTuple], Prediction],
    batch: jraph.GraphsTuple,
) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
    """Runs inference on a single batch with a given already-jitted force field."""
    batch_energies = []
    batch_forces = []
    batch_stress = []

    output = jitted_force_field_fun(batch)
    mask = jraph.get_graph_padding_mask(batch)

    node_idx = 0
    for i in range(output.energy.shape[0]):
        if mask[i]:
            batch_energies.append(float(output.energy[i]))
            graph_forces = output.forces[node_idx : node_idx + batch.n_node[i]]
            node_idx += batch.n_node[i]
            batch_forces.append(graph_forces)
            if output.stress is not None:
                batch_stress.append(output.stress[i])

    return batch_energies, batch_forces, batch_stress


def _get_optimal_max_n_node(graphs: list[jraph.GraphsTuple]) -> int:
    """Finds optimal max. number of nodes setting for given graphs."""
    num_atoms = [graph.nodes.positions.shape[0] for graph in graphs]
    max_n_node = int(np.ceil(np.median(num_atoms)))
    logger.debug("Setting max_n_node to %s.", max_n_node)
    return max_n_node


def _get_optimal_max_n_edge(graphs: list[jraph.GraphsTuple], max_n_node: int) -> int:
    """Finds optimal max. number of edges setting for given graphs."""
    num_neighbors = []
    for graph in graphs:
        _, counts = np.unique(graph.receivers, return_counts=True)
        num_neighbors.append(counts)
    median = int(np.ceil(np.median(np.concatenate(num_neighbors)).item()))
    max_n_edge = (median * max_n_node) // 2
    logger.debug("Setting max_n_edge to %s.", max_n_edge)
    return max_n_edge


def _prepare_graphs(
    structures: list[ase.Atoms],
    allowed_atomic_numbers: set[int],
    cutoff_distance: float,
) -> list[jraph.GraphsTuple]:
    """Prepares graphs from list of `ase.Atoms` objects."""
    z_table = AtomicNumberTable(sorted(allowed_atomic_numbers))

    chemical_systems = [
        ChemicalSystem(
            atomic_numbers=atoms.numbers,
            atomic_species=np.asarray([z_table.z_to_index(z) for z in atoms.numbers]),
            positions=atoms.get_positions(),
        )
        for atoms in structures
    ]

    return [
        create_graph_from_chemical_system(system, cutoff_distance)
        for system in chemical_systems
    ]


def _prepare_graph_dataset(
    graphs: list[jraph.GraphsTuple],
    batch_size: int,
    max_n_node: Optional[int],
    max_n_edge: Optional[int],
) -> GraphDataset:
    """Initializes the graph dataset object from jraph graphs."""
    if max_n_node is None:
        max_n_node = _get_optimal_max_n_node(graphs)
    if max_n_edge is None:
        max_n_edge = _get_optimal_max_n_edge(graphs, max_n_node)

    return GraphDataset(
        graphs=graphs,
        batch_size=batch_size,
        max_n_node=max_n_node,
        max_n_edge=max_n_edge,
        should_shuffle=False,
        skip_last_batch=False,
        raise_exc_if_graphs_discarded=True,
    )


def run_batched_inference(
    structures: list[ase.Atoms],
    force_field: ForceField,
    batch_size: int = 16,
    max_n_node: Optional[int] = None,
    max_n_edge: Optional[int] = None,
) -> list[Prediction]:
    """Runs a batched inference on given structures.

    Computes energies, forces, and if available with the given force field,
    stress tensors. Result will be returned as a list of `Prediction` objects, one
    for each input structure.

    Args:
        structures: The structures to batch and then compute predictions for.
        force_field: The force field object to compute the predictions with.
        batch_size: The batch size. Default is 16.
        max_n_node: This value will be multiplied with the batch size to determine the
                    maximum number of nodes we allow in a batch.
                    Note that a batch will always contain max_n_node * batch_size
                    nodes, as the remaining ones are filled up with dummy nodes.
                    The default is `None` which means an optimal number is automatically
                    computed for the dataset.
        max_n_edge: This value will be multiplied with the batch size to determine the
                    maximum number of edges we allow in a batch.
                    Note that a batch will always contain max_n_edge * batch_size
                    edges, as the remaining ones are filled up with dummy edges.
                    The default is `None` which means an optimal number is automatically
                    computed for the dataset.
    Returns:
        A list of predictions for each structure. These dataclasses will hold a float
        for energy, a numpy array for forces of shape `(num_atoms, 3)`, and optionally
        one for stress of shape `(3, 3)`.
    """
    graphs = _prepare_graphs(
        structures, force_field.allowed_atomic_numbers, force_field.cutoff_distance
    )
    graph_dataset = _prepare_graph_dataset(graphs, batch_size, max_n_node, max_n_edge)

    logger.info(
        "Graphs preparation done. Now running inference "
        "on %s structure(s) in %s batches...",
        len(graph_dataset.graphs),
        len(graph_dataset),
    )

    jitted_force_field_fun = jax.jit(force_field)

    energies = []
    forces = []
    stress = []

    for batch_idx, batch in enumerate(graph_dataset):
        energies_batch, forces_batch, stress_batch = _run_inference_on_a_single_batch(
            jitted_force_field_fun, batch
        )
        energies.extend(energies_batch)
        forces.extend(forces_batch)
        stress.extend(stress_batch)

        logger.info("Batch %s completed.", batch_idx + 1)

    if len(stress) == 0:
        stress = [None] * len(energies)

    return [
        Prediction(energy=e, forces=f, stress=s)
        for e, f, s in zip(energies, forces, stress)
    ]
