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

import jax
import jraph
import numpy as np

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.data.helpers.neighborhood import get_neighborhood
from mlip.typing.graph_definition import GraphEdges, GraphGlobals, GraphNodes


def create_graph_from_chemical_system(
    chemical_system: ChemicalSystem,
    distance_cutoff_angstrom: float,
    batch_it_with_minimal_dummy: bool = False,
) -> jraph.GraphsTuple:
    """Creates a jraph.GraphsTuple object from a chemical system object.

    This includes computing the senders/receivers/shifts for the system and otherwise
    just transferring data 1-to-1 to the graph.

    Args:
        chemical_system: The chemical system object.
        distance_cutoff_angstrom: The graph distance cutoff in Angstrom.
        batch_it_with_minimal_dummy: Batch the dummy together with a minimal dummy
                                     graph of size 1 node and 1 edge. Needed if you
                                     want to run a model inference on just this single
                                     graph. Default is False.

    Returns:
        The ``jraph.GraphsTuple`` object for the given chemical system.
    """
    senders, receivers, shift_vectors = get_neighborhood(
        positions=chemical_system.positions,
        cutoff=distance_cutoff_angstrom,
        pbc=chemical_system.pbc,
        cell=chemical_system.cell,
    )

    cell = np.zeros((3, 3)) if chemical_system.cell is None else chemical_system.cell
    energy = np.array(0.0 if chemical_system.energy is None else chemical_system.energy)

    graph = jraph.GraphsTuple(
        nodes=GraphNodes(
            positions=chemical_system.positions,
            forces=chemical_system.forces,
            species=chemical_system.atomic_species,
        ),
        edges=GraphEdges(shifts=shift_vectors, displ_fun=None),
        globals=jax.tree.map(
            lambda x: x[None, ...],
            GraphGlobals(
                cell=cell,
                energy=energy,
                stress=chemical_system.stress,
                weight=np.asarray(chemical_system.weight),
            ),
        ),
        receivers=receivers,
        senders=senders,
        n_edge=np.array([senders.shape[0]]),
        n_node=np.array([chemical_system.positions.shape[0]]),
    )

    if batch_it_with_minimal_dummy:
        return next(
            dynamically_batch(
                [graph],
                n_node=graph.nodes.positions.shape[0] + 1,
                n_edge=graph.senders.shape[0] + 1,
                n_graph=2,
            )
        )

    return graph
