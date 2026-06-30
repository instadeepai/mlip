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
from collections import deque
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pydantic
from ase import Atoms
from jax import Array

from mlip.graph import Graph

logger = logging.getLogger("mlip")


def _breadth_first_search(neighbors: dict, start: int) -> list[int]:
    """Return sorted indices of all nodes reachable from `start` via BFS."""
    visited, queue = set(), deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(n for n in neighbors[node] if n not in visited)
    return sorted(visited)


# --- Base RestraintPotential classes ---


class RestraintPotentialConfig(pydantic.BaseModel):
    """Abstract base for restraint potential config classes.

    `RestraintPotentialConfig`s can be passed to the `MetadynamicsConfig.restraints`
    list to add restraint potential terms to the metadynamics potential.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def resolve(self, atoms: Atoms) -> "RestraintPotentialConfig":
        """Return a resolved copy of this config, populated with system-derived fields.

        The default implementation is a no-op. Subclasses that require information
        from the initial structure should override this method.
        """
        return self

    def build_restraint(self) -> "RestraintPotential":
        """Construct the restraint potential object described by this config."""
        raise NotImplementedError


class RestraintPotential:
    """Base class for restraint potentials."""

    Config = RestraintPotentialConfig

    def __call__(self, graph: Graph) -> Array:
        """Return the restraint potential energy (eV) for the given graph."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class PositionalRestraintConfig(RestraintPotentialConfig):
    """Config for a harmonic positional restraint potential on a set of atoms.

    Adds a potential to keep a set of atoms close to their initial positions.

    Can be passed to the `MetadynamicsConfig.restraints` list to add a
    harmonic positional restraint potential term to the metadynamics potential.

    Attributes:
        kappa: Restraint force constant in eV / Å².
        atom_indices: Indices of atoms to restrain. Mutually exclusive with
            `start_atom_index`; one of the two must be provided.
        start_atom_index: If `atom_indices` is `None`, a BFS is performed from
            this atom to identify the restrained fragment.
        initial_positions: Reference positions populated by calling `resolve`.
            Do not set manually.
        type: Discriminator field; always `"positional"`.
    """

    kappa: float
    atom_indices: list[int] | None = None
    start_atom_index: int | None = None

    initial_positions: jnp.ndarray | None = None
    type: Literal["positional"] = pydantic.Field("positional", init=False)

    @pydantic.model_validator(mode="after")
    def _check_not_manually_set(self) -> "PositionalRestraintConfig":
        if self.initial_positions is not None:
            raise ValueError(
                "`initial_positions` is set by `resolve()` Do not set manually."
            )
        return self

    def resolve(self, atoms: Atoms) -> "PositionalRestraintConfig":
        """Populate `atom_indices` and `initial_positions` from `atoms`.

        If `atom_indices` is not populated, runs a breadth-first search from
        `start_atom_index` to identify the connected fragment to be restrained.
        """
        indices = self.atom_indices
        positions = atoms.get_positions()

        if indices is None and self.start_atom_index is not None:
            dist_mat = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
            neighbors = {
                i: np.where((dist_mat[i] > 0.1) & (dist_mat[i] < 1.8))[0]
                for i in range(len(positions))
            }
            indices = _breadth_first_search(neighbors, self.start_atom_index)
            logger.info(
                f"Position restraint auto-detected: {len(indices)} restrained atoms."
            )

        initial_positions = jnp.array(positions[indices])
        return self.model_copy(
            update={"atom_indices": indices, "initial_positions": initial_positions}
        )

    def build_restraint(self) -> "PositionalRestraintPotential":
        """Construct the :class:`PositionalRestraintPotential` for this config."""
        if self.atom_indices is None or self.initial_positions is None:
            raise ValueError(
                "PositionalRestraintConfig must be resolved before calling "
                "build_restraint(). Call `resolve(atoms)` first."
            )
        return PositionalRestraintPotential(self)


class PositionalRestraintPotential(RestraintPotential):
    """Positional restraint potential V = 0.5 * kappa * sum_i |r_i - r0_i|²."""

    Config = PositionalRestraintConfig

    def __init__(self, config: PositionalRestraintConfig):
        self._config = config
        self.atom_indices = jnp.array(self._config.atom_indices)

    def __call__(self, graph: Graph) -> Array:
        """Return V = 0.5 * kappa * Σ_i |r_i - r0_i|² (eV)."""
        if graph.edges.displ_fun is None:
            raise ValueError(
                "Graph edges must have a displacement function to use a "
                "`PositionalRestraintPotential`."
            )
        dr = graph.edges.displ_fun(
            graph.nodes.positions[self.atom_indices],
            self._config.initial_positions,
            graph.globals.cell,
        )
        return 0.5 * self._config.kappa * jnp.sum(dr**2)
