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
from typing import Callable, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from mlip.graph import Graph
from mlip.simulation.jax_md.helpers import update_graph_in_simulation_step
from mlip.simulation.metadynamics.potential_terms import (
    BiasPotential,
    RestraintPotential,
    WallPotential,
)
from mlip.simulation.metadynamics.states import (
    MetadynamicsState,
    MetadynamicsSystemState,
)

logger = logging.getLogger("mlip")

MetaDynamicsPotential: TypeAlias = BiasPotential | WallPotential | RestraintPotential


def build_metadynamics_energy_head(
    base_energy_head: Callable[[Graph], Array],
    metadynamics_potentials: list[MetaDynamicsPotential],
) -> Callable[[Graph], Array]:
    """Build an energy head combining the base MLIP energy with metadynamics terms.

    Args:
        base_energy_head: The original per-graph energy callable from the force field.
        metadynamics_potentials: Bias, wall, and/or restraint potentials to add.

    Returns:
        A callable that takes a `Graph` and returns per-graph energies (eV) with
        all metadynamics terms summed into the first graph's energy.
    """
    if len(metadynamics_potentials) == 0:
        raise ValueError("At least one metadynamics potential must be provided.")

    def metadynamics_energy_head(graph):
        base_graph_energy = base_energy_head(graph)
        metad_energy = sum(potential(graph) for potential in metadynamics_potentials)
        metad_per_graph = jnp.zeros(graph.num_graphs).at[0].set(metad_energy)
        return base_graph_energy + metad_per_graph

    return metadynamics_energy_head


def update_graph_in_metadynamics_simulation_step(
    system_state: MetadynamicsSystemState | list[MetadynamicsSystemState],
    positions: np.ndarray | list[np.ndarray],
    graph: Graph,
    is_batched: bool,
    box: jax.Array | list[jax.Array] | None,
) -> Graph:
    """Update the graph for a metadynamics simulation step.

    Wraps `update_graph_in_simulation_step` and additionally injects the
    current Gaussian hill state so that bias potentials can access them.
    """
    updated_graph = update_graph_in_simulation_step(
        system_state, positions, graph, is_batched, box
    )
    metadynamics_state = getattr(system_state, "metadynamics_state", None)
    if metadynamics_state is not None:
        updated_graph = updated_graph.update_global_features(
            gaussian_centers=metadynamics_state.gaussian_centers,
            gaussian_heights=metadynamics_state.gaussian_heights,
            num_gaussians=metadynamics_state.num_gaussians,
        )
    return updated_graph


def update_metadynamics_state(
    metadynamics_state: MetadynamicsState,
    bias_cv_values: Array,
    scaled_height: float,
) -> MetadynamicsState:
    """Deposit a new Gaussian hill into the metadynamics state.

    Updates `gaussian_centers`, `gaussian_heights`, and `num_gaussians`,
    unless the hill buffer is full (`num_gaussians == max_gaussians`).

    Args:
        metadynamics_state: Current hill state.
        bias_cv_values: Current collective variable values, shape `(num_cvs,)`.
        scaled_height: Hill height after well-tempered scaling (eV).

    Returns:
        Updated metadynamics state with the new hill appended.
    """
    max_gaussians = metadynamics_state.gaussian_centers.shape[0]
    has_space = metadynamics_state.num_gaussians < max_gaussians
    safe_idx = jnp.minimum(metadynamics_state.num_gaussians, max_gaussians - 1)

    new_centers = jax.lax.cond(
        has_space,
        lambda: metadynamics_state.gaussian_centers.at[safe_idx].set(bias_cv_values),
        lambda: metadynamics_state.gaussian_centers,
    )
    new_heights = jax.lax.cond(
        has_space,
        lambda: metadynamics_state.gaussian_heights.at[safe_idx].set(scaled_height),
        lambda: metadynamics_state.gaussian_heights,
    )
    new_num_gaussians = jax.lax.cond(
        has_space,
        lambda: metadynamics_state.num_gaussians + 1,
        lambda: metadynamics_state.num_gaussians,
    )

    return MetadynamicsState(
        gaussian_centers=new_centers,
        gaussian_heights=new_heights,
        num_gaussians=new_num_gaussians,
    )
