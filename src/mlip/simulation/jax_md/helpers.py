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

from typing import Any, Callable, TypeAlias

import ase
import jax
import jax.numpy as jnp
import jax_md
import jraph
import numpy as np
from ase.units import Ang, J, eV, fs, kB, kcal, kg, m, mol, s
from jax.tree_util import tree_map

from mlip.simulation.configs.jax_md_config import JaxMDSimulationConfig
from mlip.simulation.enums import SimulationType
from mlip.simulation.jax_md.jaxmd_utils import batched_nvt_langevin
from mlip.simulation.jax_md.states import EpisodeLog, SystemState
from mlip.utils.jax_utils import TupleLeaf

NeighborList: TypeAlias = jax_md.partition.NeighborList
NeighborListFns: TypeAlias = jax_md.partition.NeighborListFns

DUMMY_ARRAY = np.array([[0.0, 0.0, 0.0]])
DUMMY_CELL = np.array([[[0.0, 0.0, 0.0]] * 3])

TIMESTEP_CONVERSION_FACTOR = np.sqrt(kg * (kcal / mol) / J) * (m / Ang) * (fs / s)
TEMPERATURE_CONVERSION_FACTOR = kB / (kcal / mol)
KCAL_PER_MOL_PER_ELECTRON_VOLT = eV / (kcal / mol)
VELOCITY_CONVERSION_FACTOR = fs / TIMESTEP_CONVERSION_FACTOR

MINIMIZATION_PARAMETER_TIMESTEP_MAX_RATIO = 4
MINIMIZATION_PARAMETER_N_MIN = 5
MINIMIZATION_PARAMETER_F_INC = 1.1
MINIMIZATION_PARAMETER_F_DEC = 0.5
MINIMIZATION_PARAMETER_ALPHA_START = 0.1
MINIMIZATION_PARAMETER_F_ALPHA = 0.99


def is_neighbor_list(obj: Any) -> bool:
    """Whether an object is a JAX-MD neighbor list object."""
    return isinstance(obj, jax_md.partition.NeighborList)


def is_neighbor_fun(obj: Any) -> bool:
    """Whether an object is a JAX-MD neighbor function object."""
    return isinstance(obj, jax_md.partition.NeighborListFns)


def is_system_state(obj: Any) -> bool:
    """Whether an object is a `SystemState` object."""
    return isinstance(obj, SystemState)


def is_episode_log(obj: Any) -> bool:
    """Whether an object is a `EpisodeLog` object."""
    return isinstance(obj, EpisodeLog)


def _get_dummy_edge_mask_for_batched_graph(
    receivers: list[jax.Array], subgraph_sizes: jax.Array
) -> jax.Array:
    """The logic of this function is as follows:

    Each receiver edge array has values of length of that subgraph in those places
    where we have a dummy edge. That is how JAX-MD does the padding. This is safe,
    because that index does not even exist in the system of course. This would work
    for the senders array in the same way, but getting the mask from one of the two
    is enough.
    """
    sizes_without_dummy = jnp.delete(subgraph_sizes, -1)
    sizes_as_list = list(jnp.split(sizes_without_dummy, sizes_without_dummy.shape[0]))
    dummy_edge_mask = tree_map(lambda r, n: r == n, receivers, sizes_as_list)
    dummy_edge_mask = jax.lax.concatenate(
        dummy_edge_mask + [jnp.array([True])], dimension=0
    )
    return dummy_edge_mask


def update_graph_in_simulation_step(
    system_state: SystemState | list[SystemState],
    positions: np.ndarray | list[np.ndarray],
    graph: jraph.GraphsTuple,
    is_batched: bool,
) -> jraph.GraphsTuple:
    """Updates a graph in a simulation step.

    1) Creates a batch of graphs out of a graph by adding one simple dummy graph. The
       dummy graph has just one node and one edge.

    2) The positions of the input graph are updated with the given positions
       and the edges are also updated given the edges contained in the given
       system state.

    Args:
        system_state: The system state during the simulation.
                      Might be a list if batched simulation.
        positions: The current positions of the system.
                   Might be a list if batched simulation.
        graph: The graph of the system.
        is_batched: Whether this function already receives multiple systems, because
                    the simulation is a batched one.

    Returns:
        The updated and batched graph.
    """
    neighbors = tree_map(
        lambda state, pos: state.neighbors.update(pos),
        system_state,
        positions,
        is_leaf=is_system_state,
    )
    senders = tree_map(lambda n: n.idx[1, :], neighbors, is_leaf=is_neighbor_list)
    receivers = tree_map(lambda n: n.idx[0, :], neighbors, is_leaf=is_neighbor_list)

    # If we have batched simulations, we don't need to do the batching part of this
    # function, which simplifies some of the replace operations
    if is_batched:
        new_positions = jax.lax.concatenate(positions + [DUMMY_ARRAY], dimension=0)
        node_cumsum = jax.lax.concatenate(
            [np.array([0]), jnp.delete(jnp.cumsum(graph.n_node), -1)], dimension=0
        )
        offset = jnp.repeat(node_cumsum, graph.n_edge)
        new_receivers = (
            jax.lax.concatenate(receivers + [np.array([0])], dimension=0) + offset
        )
        new_senders = (
            jax.lax.concatenate(senders + [np.array([0])], dimension=0) + offset
        )
        # Some models need all dummy edges to point to dummy nodes.
        # The following lines make sure this is the case.
        dummy_edge_mask = _get_dummy_edge_mask_for_batched_graph(
            receivers, graph.n_node
        )
        dummy_edge_value = new_receivers[-1]
        new_receivers = jnp.where(dummy_edge_mask, dummy_edge_value, new_receivers)
        new_senders = jnp.where(dummy_edge_mask, dummy_edge_value, new_senders)
        # These conversions needed for integer types:
        new_species = jnp.asarray(graph.nodes.species)
        new_n_node = jnp.asarray(graph.n_node)
        new_n_edge = jnp.asarray(graph.n_edge)
        return graph._replace(
            senders=new_senders,
            receivers=new_receivers,
            n_node=new_n_node,
            n_edge=new_n_edge,
            nodes=graph.nodes._replace(positions=new_positions, species=new_species),
        )

    new_positions = jax.lax.concatenate([positions, DUMMY_ARRAY], dimension=0)
    new_species = jax.lax.concatenate([graph.nodes.species, np.array([0])], dimension=0)

    num_nodes = int(graph.n_node[0])
    new_receivers = jax.lax.concatenate([receivers, np.array([num_nodes])], dimension=0)
    new_senders = jax.lax.concatenate([senders, np.array([num_nodes])], dimension=0)

    new_n_node = jax.lax.concatenate([graph.n_node, np.array([1])], dimension=0)
    new_n_edge = jax.lax.concatenate([graph.n_edge, np.array([1])], dimension=0)
    new_energy = jax.lax.concatenate(
        [graph.globals.energy, np.array([0.0])], dimension=0
    )
    new_weight = jax.lax.concatenate(
        [graph.globals.weight, np.array([0.0])], dimension=0
    )
    new_cell = jax.lax.concatenate([graph.globals.cell, DUMMY_CELL], dimension=0)

    return graph._replace(
        senders=new_senders,
        receivers=new_receivers,
        n_node=new_n_node,
        n_edge=new_n_edge,
        nodes=graph.nodes._replace(positions=new_positions, species=new_species),
        globals=graph.globals._replace(
            cell=new_cell,
            energy=new_energy,
            weight=new_weight,
        ),
    )


def init_simulation_algorithm(
    model_calculate_fun: Callable,
    shift_fun: Callable,
    sim_config: JaxMDSimulationConfig,
) -> tuple[Callable, Callable]:
    """Initializes the minimizer or MD integrator object of JAX-MD.

    Currently, for MD, the NVT-Langevin integrator is returned, and for energy
    minimization, the FIRE descent algorithm is used as the only options.

    Args:
        model_calculate_fun: The model calculate function outputting
                             either forces or energies.
        shift_fun: The shift function.
        sim_config: The pydantic config object for the JAX-MD simulation engine.

    Returns:
        A simulation init function and a simulation apply function used later to run
        the simulation.
    """
    if sim_config.simulation_type == SimulationType.MD:
        return batched_nvt_langevin(
            model_calculate_fun,
            shift_fun,
            kT=sim_config.temperature_kelvin * TEMPERATURE_CONVERSION_FACTOR,
            dt=sim_config.timestep_fs * TIMESTEP_CONVERSION_FACTOR,
        )

    start_timestep_fs = sim_config.timestep_fs * TIMESTEP_CONVERSION_FACTOR
    return jax_md.minimize.fire_descent(
        model_calculate_fun,
        shift_fun,
        dt_start=start_timestep_fs,
        dt_max=start_timestep_fs * MINIMIZATION_PARAMETER_TIMESTEP_MAX_RATIO,
        n_min=MINIMIZATION_PARAMETER_N_MIN,
        f_inc=MINIMIZATION_PARAMETER_F_INC,
        f_dec=MINIMIZATION_PARAMETER_F_DEC,
        alpha_start=MINIMIZATION_PARAMETER_ALPHA_START,
        f_alpha=MINIMIZATION_PARAMETER_F_ALPHA,
    )


def init_neighbor_lists(
    displacement_fun: Callable,
    positions: np.ndarray | list[np.ndarray],
    cutoff_distance_angstrom: float,
    edge_capacity_multiplier: float,
) -> tuple[NeighborList | list[NeighborList], NeighborListFns | list[NeighborListFns]]:
    """Initialize the neighbor lists objects for JAX-MD.

    Uses tree_map so that it works for single system or multiple systems for
    batched simulations.

    Args:
        displacement_fun: The displacement function.
        positions: The positions of the system.
        cutoff_distance_angstrom: The graph cutoff distance in Angstrom.
        edge_capacity_multiplier: The edge capacity multiplier to decide how much
                                  padding is added to the neighbor lists.

    Returns:
        A tuple of the neighbor list object and the neighbor lists function object
        that JAX-MD needs for a simulation.
    """

    def _init_impl(pos):
        _neighbor_fun = jax_md.partition.neighbor_list(
            displacement_or_metric=displacement_fun,
            box=jnp.nan,
            r_cutoff=cutoff_distance_angstrom,
            disable_cell_list=False,
            format=jax_md.partition.NeighborListFormat.Sparse,
            capacity_multiplier=edge_capacity_multiplier,
        )
        _neighbors = _neighbor_fun.allocate(pos)
        return TupleLeaf([_neighbors, _neighbor_fun])

    init_result = tree_map(_init_impl, positions)
    neighbors = tree_map(lambda x: x[0], init_result)
    neighbor_fun = tree_map(lambda x: x[1], init_result)

    return neighbors, neighbor_fun


def get_masses(atoms: ase.Atoms) -> np.ndarray:
    """Returns the masses for a given set of atoms.

    Important note: this is currently just implemented as the ase.Atoms.get_masses()
    function which returns 1.008 for hydrogen instead of 1, etc. This may need to be
    adapted in the future, but for our H,C,N,O,S,P elements, the difference should be
    small.

    Args:
        atoms: An ase.Atoms object representing the molecule/system

    Returns:
        The atomic masses.
    """
    return atoms.get_masses()
