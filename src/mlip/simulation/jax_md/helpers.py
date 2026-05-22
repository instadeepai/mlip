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
import numpy as np
from ase.units import Ang, J, bar, eV, fs, kB, kcal, kg, m, mol, s

from mlip.graph import Graph
from mlip.models.force_field import ForceField
from mlip.simulation.configs.jax_md_config import JaxMDSimulationConfig
from mlip.simulation.enums import MDIntegrator, SimulationType
from mlip.simulation.jax_md.jaxmd_utils import batched_nvt_langevin
from mlip.simulation.jax_md.npt_montecarlo_langevin import npt_montecarlo_langevin
from mlip.simulation.jax_md.states import EpisodeLog, SystemState
from mlip.simulation.montecarlo_barostat import create_high_precision_force_field
from mlip.utils.jax_utils import TupleLeaf

NeighborList: TypeAlias = jax_md.partition.NeighborList
NeighborListFns: TypeAlias = jax_md.partition.NeighborListFns

DUMMY_ARRAY = np.array([[0.0, 0.0, 0.0]])
DUMMY_CELL = np.array([[[0.0, 0.0, 0.0]] * 3])

TIMESTEP_CONVERSION_FACTOR = np.sqrt(kg * (kcal / mol) / J) * (m / Ang) * (fs / s)
TEMPERATURE_CONVERSION_FACTOR = kB / (kcal / mol)
KCAL_PER_MOL_PER_ELECTRON_VOLT = eV / (kcal / mol)
VELOCITY_CONVERSION_FACTOR = fs / TIMESTEP_CONVERSION_FACTOR
PRESSURE_CONVERSION_FACTOR = bar / (kcal / mol) / (Ang**3)

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


def box_to_cell(box: jax.Array) -> jax.Array:
    """Converts a box (a minimal representation of the cell) to a (3,3) array."""
    if box.ndim == 0:
        return jnp.eye(3) * box
    elif box.ndim == 1:
        return jnp.diag(box)
    elif box.ndim == 2:
        return box
    else:
        raise ValueError(f"Box must be 0D, 1D or 2D, got {box.ndim}D.")


def init_displacement_fun(
    box: np.ndarray | None,
) -> tuple[Callable, Callable, Callable]:
    """Initialise the displacement function, shift function, and cell-to-box converter.

    Args:
        box: Box dimensions. None for free (non-periodic) space; a 1-D array of
             length 3 for an orthorhombic periodic box.

    Returns:
        (displacement_fun, shift_fun, cell_to_box_fun)
    """
    if box is None:
        displacement_fun, shift_fun = jax_md.space.free()
        cell_to_box_fun = lambda cell: cell  # noqa: E731
    else:
        displacement_fun, shift_fun = jax_md.space.periodic_general(
            box, fractional_coordinates=False, wrapped=False
        )

        def _cell_to_box(cell: jnp.ndarray) -> jnp.ndarray:
            return jnp.diag(cell[0])

        cell_to_box_fun = jax.tree_util.Partial(_cell_to_box)

    return displacement_fun, shift_fun, cell_to_box_fun


def make_batched_displ_fun(
    base_displacement_fun: Callable, n_edge_per_system: jax.Array
) -> Callable:
    """Create a displacement function for batched simulations.

    In batched simulations each system has its own box, so edges belonging to system i
    must use cell[i] when computing displacements. This function is used to override
    the displacement function from `init_displacement_fun` for batched simulations.

    Args:
        base_displacement_fun: The displacement function from `init_displacement_fun`.
        n_edge_per_system: Per-system edge counts, including the dummy graph.

    Returns:
        A `Partial`-wrapped displacement function with the expected call signature:
        `(receivers_pos, senders_pos, cell) -> displacements`.
    """
    edge_system_idx = jnp.concatenate([
        jnp.full((n,), i, dtype=jnp.int32) for i, n in enumerate(n_edge_per_system)
    ])

    def _batched_displ_fun(
        receivers_pos: jax.Array,
        senders_pos: jax.Array,
        cell: jax.Array,
    ) -> jax.Array:
        per_edge_cell = cell[edge_system_idx]  # (num_edges, 3, 3)

        def _single_edge(r: jax.Array, s: jax.Array, c: jax.Array) -> jax.Array:
            return base_displacement_fun(r, s, box=jnp.diag(c))

        return jax.vmap(_single_edge, in_axes=(0, 0, 0))(
            receivers_pos, senders_pos, per_edge_cell
        )

    return jax.tree_util.Partial(_batched_displ_fun)


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
    dummy_edge_mask = jax.tree.map(lambda r, n: r == n, receivers, sizes_as_list)
    dummy_edge_mask = jax.lax.concatenate(
        dummy_edge_mask + [jnp.array([True])], dimension=0
    )
    return dummy_edge_mask


def get_neighbor_list_senders(neighbors: NeighborList) -> jax.Array:
    """Return senders from a jax-md neighbor list (channel 1 of `idx`)."""
    return neighbors.idx[1, :]


def get_neighbor_list_receivers(neighbors: NeighborList) -> jax.Array:
    """Return receivers from a jax-md neighbor list (channel 0 of `idx`)."""
    return neighbors.idx[0, :]


def _extract_neighbor_indices(
    neighbors: NeighborList | list[NeighborList],
) -> tuple[jax.Array | list[jax.Array], jax.Array | list[jax.Array]]:
    """Returns `(senders, receivers)` extracted from a neighbor list.

    The output preserves the pytree structure of `neighbors`: a single
    `NeighborList` yields a pair of `jax.Array`, while a list of
    `NeighborList` (batched simulations) yields a pair of lists of
    `jax.Array`.
    """
    senders = jax.tree.map(
        get_neighbor_list_senders, neighbors, is_leaf=is_neighbor_list
    )
    receivers = jax.tree.map(
        get_neighbor_list_receivers, neighbors, is_leaf=is_neighbor_list
    )
    return senders, receivers


def _build_batched_offset_indices(
    senders: list[jax.Array],
    receivers: list[jax.Array],
    n_edge: jax.Array,
    n_node: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Concatenate per-system senders/receivers, offset by node-cumsum, and
    redirect dummy edges to dummy nodes.
    """
    node_cumsum = jax.lax.concatenate(
        [np.array([0]), jnp.delete(jnp.cumsum(n_node), -1)], dimension=0
    )
    offset = jnp.repeat(node_cumsum, n_edge)
    new_receivers = (
        jax.lax.concatenate(receivers + [np.array([0])], dimension=0) + offset
    )
    new_senders = jax.lax.concatenate(senders + [np.array([0])], dimension=0) + offset
    dummy_edge_mask = _get_dummy_edge_mask_for_batched_graph(receivers, n_node)
    dummy_edge_value = new_receivers[-1]
    new_receivers = jnp.where(dummy_edge_mask, dummy_edge_value, new_receivers)
    new_senders = jnp.where(dummy_edge_mask, dummy_edge_value, new_senders)
    return new_senders, new_receivers


def update_graph_in_simulation_step(
    system_state: SystemState | list[SystemState],
    positions: np.ndarray | list[np.ndarray],
    graph: Graph,
    is_batched: bool,
    box: jax.Array | list[jax.Array] | None,
) -> Graph:
    """Updates a graph in a simulation step.

    1) Creates a batch of graphs out of a graph by adding one simple dummy graph. The
       dummy graph has just one node and one edge.

    2) The positions of the input graph are updated with the given positions
       and the edges are also updated given the edges contained in the given
       system state.

    3) When the system state carries a long-range neighbor list, the long-range
       senders/receivers/n_edge_long_range fields on the graph are updated as well.

    Args:
        system_state: The system state during the simulation.
                      Might be a list if batched simulation.
        positions: The current positions of the system.
                   Might be a list if batched simulation.
        graph: The graph of the system.
        is_batched: Whether this function already receives multiple systems, because
                    the simulation is a batched one.
        box: The current box of the system. Can have shape (,), (3,) or (3,3). For
            batched simulations, can be a list of per-system boxes. Is ignored if None.

    Returns:
        The updated and batched graph.
    """
    if is_batched and not isinstance(box, list):
        box = [box] * len(positions)

    neighbors = jax.tree.map(
        lambda state, pos, b: state.neighbors.update(pos, box=b),
        system_state,
        positions,
        box,
        is_leaf=is_system_state,
    )
    senders, receivers = _extract_neighbor_indices(neighbors)

    has_long_range = graph.n_edge_long_range is not None
    if has_long_range:
        long_range_neighbors = jax.tree.map(
            lambda state, pos, b: state.long_range_neighbors.update(pos, box=b),
            system_state,
            positions,
            box,
            is_leaf=is_system_state,
        )
        senders_long_range, receivers_long_range = _extract_neighbor_indices(
            long_range_neighbors
        )
    else:
        senders_long_range = None
        receivers_long_range = None

    def _concat(
        current_arr: np.ndarray | list[np.ndarray], new_vals: list[Any] | jax.Array
    ) -> jax.Array:
        """Concatenates an array or list of arrays with a new set of values."""
        if not isinstance(current_arr, list):
            current_arr = [current_arr]
        if not isinstance(new_vals, (np.ndarray, jax.Array)):
            new_vals = jnp.array(new_vals, dtype=current_arr[0].dtype)
        return jax.lax.concatenate(current_arr + [new_vals], dimension=0)

    # If we have batched simulations, we don't need to do the batching part of this
    # function, which simplifies some of the replace operations
    if is_batched:
        new_positions = _concat(positions, DUMMY_ARRAY)
        new_senders, new_receivers = _build_batched_offset_indices(
            senders, receivers, graph.n_edge, graph.n_node
        )
        # These conversions needed for integer types:
        new_atomic_numbers = jnp.asarray(graph.nodes.atomic_numbers)
        new_n_node = jnp.asarray(graph.n_node)
        new_n_edge = jnp.asarray(graph.n_edge)

        new_globals = graph.globals
        if box[0] is not None:
            cells = [box_to_cell(b)[None, ...] for b in box]
            new_cell = _concat(cells, jnp.zeros_like(cells[0]))
            new_globals = new_globals.replace(cell=new_cell)

        replace_kwargs = dict(
            senders=new_senders,
            receivers=new_receivers,
            n_node=new_n_node,
            n_edge=new_n_edge,
            nodes=graph.nodes.replace(
                positions=new_positions, atomic_numbers=new_atomic_numbers
            ),
            globals=new_globals,
        )
        if has_long_range:
            new_lr_senders, new_lr_receivers = _build_batched_offset_indices(
                senders_long_range,
                receivers_long_range,
                graph.n_edge_long_range,
                graph.n_node,
            )
            replace_kwargs["senders_long_range"] = new_lr_senders
            replace_kwargs["receivers_long_range"] = new_lr_receivers
            replace_kwargs["n_edge_long_range"] = jnp.asarray(graph.n_edge_long_range)
        return graph.replace(**replace_kwargs)

    new_positions = _concat(positions, DUMMY_ARRAY)
    new_atomic_numbers = _concat(graph.nodes.atomic_numbers, [0])

    num_nodes = int(positions.shape[0])
    new_receivers = _concat(receivers, [num_nodes])
    new_senders = _concat(senders, [num_nodes])
    new_receivers = _concat(receivers, [num_nodes])
    new_senders = _concat(senders, [num_nodes])

    new_n_node = _concat(graph.n_node, [1])
    new_n_edge = _concat(graph.n_edge, [1])
    new_energy = _concat(graph.globals.energy, [0.0])
    new_weight = _concat(graph.globals.weight, [0.0])

    current_cell = (
        box_to_cell(box)[None, ...] if box is not None else graph.globals.cell
    )
    new_cell = _concat(current_cell, jnp.zeros_like(current_cell))

    globals_updates: dict[str, jax.Array] = {
        "cell": new_cell,
        "energy": new_energy,
        "weight": new_weight,
    }
    if graph.globals.charge is not None:
        globals_updates["charge"] = _concat(graph.globals.charge, [0.0])

    replace_kwargs = dict(
        senders=new_senders,
        receivers=new_receivers,
        n_node=new_n_node,
        n_edge=new_n_edge,
        nodes=graph.nodes.replace(
            positions=new_positions, atomic_numbers=new_atomic_numbers
        ),
        globals=graph.globals.replace(**globals_updates),
    )
    if has_long_range:
        replace_kwargs["senders_long_range"] = _concat(senders_long_range, [num_nodes])
        replace_kwargs["receivers_long_range"] = _concat(
            receivers_long_range, [num_nodes]
        )
        replace_kwargs["n_edge_long_range"] = _concat(graph.n_edge_long_range, [1])
    return graph.replace(**replace_kwargs)


def init_simulation_algorithm(
    make_model_calculate_fun: Callable,
    force_field: ForceField,
    shift_fun: Callable,
    sim_config: JaxMDSimulationConfig,
) -> tuple[Callable, Callable]:
    """Initializes the minimizer or MD integrator object of JAX-MD.

    For MD, either the `NVT_LANGEVIN` or `NPT_MC_LANGEVIN` integrator can be used
    (see :py:class:`MDIntegrator <mlip.simulation.enums.MDIntegrator>`).
    For energy minimization, the `FIRE` descent algorithm is used as the only option.

    Args:
        make_model_calculate_fun: Function to create a model calculate function
                                  to calculate forces or energies from a force field.
        force_field: The force field to use for calculating forces and energies.
        shift_fun: The shift function.
        sim_config: The pydantic config object for the JAX-MD simulation engine.

    Returns:
        A simulation init function and a simulation apply function used later to run
        the simulation.
    """
    model_calculate_fun = make_model_calculate_fun(
        force_field_model=force_field, is_energy_fun=False
    )

    if sim_config.simulation_type == SimulationType.MD:
        if sim_config.md_integrator == MDIntegrator.NVT_LANGEVIN:
            return batched_nvt_langevin(
                model_calculate_fun,
                shift_fun,
                kT=sim_config.temperature_kelvin * TEMPERATURE_CONVERSION_FACTOR,
                dt=sim_config.timestep_fs * TIMESTEP_CONVERSION_FACTOR,
            )
        elif sim_config.md_integrator == MDIntegrator.NPT_MC_LANGEVIN:
            molecule_indices = sim_config.molecule_indices
            if not isinstance(molecule_indices, list):
                raise ValueError("Molecule indices must be set for NPT simulations.")
            if isinstance(molecule_indices[0], list):
                molecule_indices = [jnp.array(m) for m in molecule_indices]
            else:
                molecule_indices = jnp.array(molecule_indices)

            barostat_force_field = create_high_precision_force_field(force_field)
            barostat_energy_fun = make_model_calculate_fun(
                force_field_model=barostat_force_field, is_energy_fun=True
            )
            return npt_montecarlo_langevin(
                model_calculate_fun,
                barostat_energy_fun,
                shift_fun,
                kT=sim_config.temperature_kelvin * TEMPERATURE_CONVERSION_FACTOR,
                dt=sim_config.timestep_fs * TIMESTEP_CONVERSION_FACTOR,
                pressure=sim_config.pressure_bar * PRESSURE_CONVERSION_FACTOR,
                barostat_interval=sim_config.barostat_update_interval,
                molecule_indices=molecule_indices,
            )
        else:
            raise ValueError(f"MD integrator {sim_config.md_integrator} not supported.")

    elif sim_config.simulation_type == SimulationType.MINIMIZATION:
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
    else:
        raise ValueError(f"Simulation type {sim_config.simulation_type} not supported.")


def init_neighbor_lists(
    displacement_fun: Callable,
    positions: np.ndarray | list[np.ndarray],
    cutoff_distance_angstrom: float,
    edge_capacity_multiplier: float,
    box: jax.Array | list[jax.Array] | None,
) -> tuple[NeighborList | list[NeighborList], NeighborListFns | list[NeighborListFns]]:
    """Initialize the neighbor lists objects for JAX-MD.

    Works for single system or multiple systems for batched simulations.
    For batched simulations with heterogeneous boxes, pass a list of per-system boxes.

    Args:
        displacement_fun: The displacement function.
        positions: The positions of the system.
        cutoff_distance_angstrom: The graph cutoff distance in Angstrom.
        edge_capacity_multiplier: The edge capacity multiplier to decide how much
                                  padding is added to the neighbor lists.
        box: The current box of the system. If None, is ignored.
             For batched simulations, can be a list of per-system boxes.

    Returns:
        A tuple of the neighbor list object and the neighbor lists function object
        that JAX-MD needs for a simulation.
    """

    boxes = box if isinstance(box, list) else [box]
    boxes = [b for b in boxes if b is not None]
    if boxes:
        _min_edge = float(min(np.min(b) for b in boxes))
        if cutoff_distance_angstrom > _min_edge / 2.0:
            raise ValueError(
                f"Cutoff ({cutoff_distance_angstrom:.3f} Å) exceeds half of the "
                f"smallest box edge ({_min_edge / 2.0:.3f} Å). This is not supported"
                "by the standard jax_md neighbor list, and alternatives are not "
                "well-tested. Please use a larger box or a smaller cutoff."
            )

    def _init_impl(pos, box_i):
        # Pass placeholder for box, as we always use the `box` kwarg in updates.
        _neighbor_fun = jax_md.partition.neighbor_list(
            displacement_or_metric=displacement_fun,
            r_cutoff=cutoff_distance_angstrom,
            format=jax_md.partition.NeighborListFormat.Sparse,
            capacity_multiplier=edge_capacity_multiplier,
            box=jnp.nan,
            disable_cell_list=True,
            fractional_coordinates=False,
        )
        _neighbors = _neighbor_fun.allocate(pos, box=box_i)
        return TupleLeaf([_neighbors, _neighbor_fun])

    is_batched = isinstance(positions, list)
    if is_batched:
        boxes = box if isinstance(box, list) else [box] * len(positions)
        init_result = jax.tree.map(_init_impl, positions, boxes)
    else:
        init_result = _init_impl(positions, box)

    neighbors = jax.tree.map(lambda x: x[0], init_result)
    neighbor_fun = jax.tree.map(lambda x: x[1], init_result)

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
