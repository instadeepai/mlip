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
from typing import Callable

import ase
import jax
from jax import Array

from mlip.data import (
    ASEAtomsReader,
    GraphDatasetBuilderConfig,
    SingleGraphDatasetBuilder,
)
from mlip.data.helpers.hessian_utils import (
    request_all_hessian_rows_batched,
    single_graph_hessian_from_subsampled_batch,
)
from mlip.data.helpers.type_aliases import GraphDatasetLike
from mlip.graph import Graph
from mlip.models import ForceField
from mlip.typing import Prediction

logger = logging.getLogger("mlip")


def check_for_single_atom_systems(structures: list[ase.Atoms] | GraphDatasetLike):
    """Raise an error if a single-atom system is found within structures."""

    if isinstance(structures, list) and isinstance(structures[0], ase.Atoms):
        if any(len(struct) == 1 for struct in structures):
            raise ValueError("Single atom systems are not supported yet.")

    elif isinstance(structures, GraphDatasetLike):
        struct_lengths = []
        for batch in structures:
            mask = batch.graph_mask()
            struct_lengths.extend(batch.n_node[mask])
        if min(struct_lengths) <= 1:
            raise ValueError("Single atom systems are not supported yet.")


def run_inference_on_a_single_batch(
    jitted_force_field_fun: Callable[[Graph], Prediction],
    batch: Graph,
) -> tuple[list[float], list[Array], list[Array], list[Array], list[Array]]:
    """Runs inference on a single batch with a given already-jitted force field."""
    batch_energies = []
    batch_forces = []
    batch_stress = []
    batch_hessians = []
    batch_partial_charges = []

    # Use the row-summing trick: jacrev over n_atoms*3 summed outputs instead of
    # total_padded_nodes*3 independent outputs, saving ~batch_size backward passes.
    batch = request_all_hessian_rows_batched(batch)
    output = jitted_force_field_fun(batch)
    mask = batch.graph_mask()

    node_idx = 0
    for i in range(output.energy.shape[0]):
        graph_start = node_idx
        graph_end = node_idx + batch.n_node[i]
        if mask[i]:
            batch_energies.append(float(output.energy[i]))

            if output.forces is not None:
                graph_forces = output.forces[graph_start:graph_end]
                batch_forces.append(graph_forces)

            if output.hessian is not None:
                graph_hessian = single_graph_hessian_from_subsampled_batch(
                    output.hessian, graph_start, graph_end
                )

                batch_hessians.append(graph_hessian)

            if output.stress is not None:
                batch_stress.append(output.stress[i])

            if output.partial_charges is not None:
                graph_partial_charges = output.partial_charges[graph_start:graph_end]
                batch_partial_charges.append(graph_partial_charges)

            node_idx += batch.n_node[i]

    return (
        batch_energies,
        batch_forces,
        batch_stress,
        batch_hessians,
        batch_partial_charges,
    )


def run_batched_inference(
    structures: list[ase.Atoms] | GraphDatasetLike,
    force_field: ForceField,
    batch_size: int = 16,
    max_n_node: int | None = None,
    max_n_edge: int | None = None,
    set_none_charges_to_zero: bool = True,
) -> list[Prediction]:
    """Runs a batched inference on given structures.

    Computes energies, forces, and if available with the given force field,
    stress tensors. Result will be returned as a list
    of `Prediction` objects, one for each input structure.

    Note: When using `batch_size=1`, we recommend to set `max_n_node` and
    `max_n_edge` explicitly to avoid edge cases in the automated computation of these
    parameters that may cause errors.

    Args:
        structures: The list of `ase.Atoms` to iterate over and then compute
                    predictions for. Optionally, an already processed `GraphDataset`
                    or `PrefetchIterator` object may be passed.
        force_field: The force field object to compute the predictions with.
        batch_size: The batch size. Default is 16. Ignored if structures are passed
                    as a `GraphDataset` or `PrefetchIterator`.
        max_n_node: This value will be multiplied with the batch size to determine the
                    maximum number of nodes we allow in a batch.
                    Note that a batch will always contain max_n_node * batch_size
                    nodes, as the remaining ones are filled up with dummy nodes.
                    The default is `None` which means an optimal number is automatically
                    computed for the dataset. Ignored if structures are passed as
                    a `GraphDataset` or `PrefetchIterator`.
        max_n_edge: This value will be multiplied with the batch size to determine the
                    maximum number of edges we allow in a batch.
                    Note that a batch will always contain max_n_edge * batch_size
                    edges, as the remaining ones are filled up with dummy edges.
                    The default is `None` which means an optimal number is automatically
                    computed for the dataset. Ignored if structures are passed as
                    a `GraphDataset` or `PrefetchIterator`.
        set_none_charges_to_zero: Whether to set None total charges to zero during
                    preprocessing. Default is `True`. Ignored if structures are
                    passed as a `GraphDataset` or `PrefetchIterator`.

    Returns:
        A list of predictions for each structure. These dataclasses will hold
        a float for energy, a numpy array for forces of shape `(num_atoms, 3)`.
        Optionally, will also contain a stress array of shape `(3, 3)` and a
        partial charge array of shape `(num_atoms,)`.

    Raises:
        ValueError: if any of the input systems has only one atom.
    """
    check_for_single_atom_systems(structures)

    if isinstance(structures, list) and isinstance(structures[0], ase.Atoms):
        if batch_size is None:
            raise ValueError("Batch size must be set when passing ase.Atoms")

        if set_none_charges_to_zero and getattr(
            force_field.config, "use_total_charge_embedding", False
        ):
            logger.warning(
                "`set_none_charges_to_zero=True` and the model uses total charge "
                "embedding. Structures without an explicit charge will be assigned "
                "charge 0, which may affect prediction quality. Consider setting "
                "charges explicitly for each structure as `atoms.info['charge']`."
            )

        builder_config = GraphDatasetBuilderConfig(
            graph_cutoff_angstrom=force_field.cutoff_distance,
            long_range_cutoff_angstrom=force_field.long_range_cutoff_distance,
            max_n_node=max_n_node,
            max_n_edge=max_n_edge,
            batch_size=batch_size,
            set_none_charges_to_zero=set_none_charges_to_zero,
        )

        reader = ASEAtomsReader(structures)
        builder = SingleGraphDatasetBuilder(
            reader, builder_config, dataset_info=force_field.dataset_info
        )

        graph_dataset = builder.get_dataset(prefetch=False)

    else:
        graph_dataset = structures

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
    hessians = []
    partial_charges = []

    for batch_idx, batch in enumerate(graph_dataset):
        start_time = time.perf_counter()
        (
            energies_batch,
            forces_batch,
            stress_batch,
            hessians_batch,
            partial_charges_batch,
        ) = run_inference_on_a_single_batch(jitted_force_field_fun, batch)
        energies.extend(energies_batch)
        forces.extend(forces_batch)
        stress.extend(stress_batch)
        hessians.extend(hessians_batch)
        partial_charges.extend(partial_charges_batch)

        end_time = time.perf_counter()
        logger.info(
            "Batch %s completed. Took %.3f seconds.",
            batch_idx + 1,
            end_time - start_time,
        )

    if len(forces) == 0:
        forces = [None] * len(energies)

    if len(stress) == 0:
        stress = [None] * len(energies)

    if len(hessians) == 0:
        hessians = [None] * len(energies)

    if len(partial_charges) == 0:
        partial_charges = [None] * len(energies)

    return [
        Prediction(energy=e, forces=f, stress=s, hessian=h, partial_charges=pc)
        for e, f, s, h, pc in zip(energies, forces, stress, hessians, partial_charges)
    ]
