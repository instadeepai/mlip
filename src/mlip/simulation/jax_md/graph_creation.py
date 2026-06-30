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

from typing import Callable

import ase
import jax
import jax.numpy as jnp

from mlip.data.chemical_system import ChemicalSystem
from mlip.graph import Graph, GraphEdges, GraphGlobals, GraphNodes


def create_graph_from_atoms_and_edges(
    atoms: ase.Atoms,
    senders: jax.Array,
    receivers: jax.Array,
    displacement_fun: Callable[[jax.Array, jax.Array], jax.Array],
    cell_to_box_fun: Callable[[jax.Array], jax.Array] | None = None,
    senders_long_range: jax.Array | None = None,
    receivers_long_range: jax.Array | None = None,
) -> Graph:
    """Creates a graph from an `ase.Atoms` object and a list of edges.

    This is the graph creation function used in the JAX-MD simulation engine.

    This function will leave the shifts of the graph empty and will populate the
    displacement function in the graph object instead. The total `charge` is
    read from `atoms.info` via :meth:`ChemicalSystem.from_ase_atoms` so that
    charge-aware models (e.g. with a Coulomb energy head) can apply the
    partial-charge correction.

    Args:
        atoms: The `ase.Atoms` object of the system.
        senders: The sender indexes of the edges for the graph.
        receivers: The receiver indexes of the edges for the graph.
        displacement_fun: Function that takes in two position vectors and
                          returns the displacement vector between them.
        cell_to_box_fun: A function that takes in a cell and outputs a minimum
            representation for use in the displ_fun.
        senders_long_range: Optional sender indexes of the long-range edges.
                            If provided together with `receivers_long_range`,
                            the resulting graph will carry a long-range neighbor
                            list using `displacement_fun` for vector computation.
        receivers_long_range: Optional receiver indexes of the long-range edges.

    Returns:
        The graph representing the system.
    """
    chem_system = ChemicalSystem.from_ase_atoms(atoms, get_property_fields=False)

    def _displ_fun(
        receivers: jax.Array, senders: jax.Array, cell: jax.Array | None
    ) -> jax.Array:
        box = cell_to_box_fun(cell)
        return displacement_fun(receivers, senders, box=box)

    vmapped_displ_fun = jax.tree_util.Partial(
        jax.vmap(_displ_fun, in_axes=(0, 0, None)),
    )

    has_long_range = senders_long_range is not None and receivers_long_range is not None
    if has_long_range:
        n_edge_long_range = jnp.array([len(senders_long_range)])
        edges_long_range = GraphEdges(shifts=None, displ_fun=vmapped_displ_fun)
    else:
        n_edge_long_range = None
        edges_long_range = None

    return Graph(
        nodes=GraphNodes(
            positions=atoms.get_positions(),
            forces=None,
            atomic_numbers=jnp.asarray(atoms.numbers),
        ),
        edges=GraphEdges(shifts=None, displ_fun=vmapped_displ_fun),
        globals=jax.tree.map(
            lambda x: x[None, ...],
            GraphGlobals(
                cell=jnp.identity(3, dtype=float),
                energy=jnp.array(0.0),
                stress=None,
                weight=jnp.asarray(1.0),
                charge=jnp.asarray(chem_system.charge)
                if chem_system.charge is not None
                else None,
            ),
        ),
        receivers=receivers,
        senders=senders,
        n_edge=jnp.array([len(senders)]),
        n_node=jnp.array([len(atoms.numbers)]),
        senders_long_range=senders_long_range if has_long_range else None,
        receivers_long_range=receivers_long_range if has_long_range else None,
        n_edge_long_range=n_edge_long_range,
        edges_long_range=edges_long_range,
    )
