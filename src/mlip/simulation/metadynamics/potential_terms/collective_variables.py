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

from collections.abc import Sequence
from typing import Literal

import ase.data as ase_data
import jax.numpy as jnp
import numpy as np
import pydantic
from ase import Atoms
from jax import Array

from mlip.graph import Graph

# --- Base CollectiveVariable classes ---


def safe_sqrt(inp: Array) -> Array:
    return jnp.sqrt(inp + 1e-8)


class CollectiveVariableConfig(pydantic.BaseModel):
    """Abstract base for collective variable config classes.

    `CollectiveVariableConfig`s can be passed to the `MetadynamicsConfig.bias_cvs`
    list to add collective variables to the bias term of the metadynamics potential.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def resolve(self, atoms: Atoms) -> "CollectiveVariableConfig":
        """Return a resolved copy of this config, populated with system-derived fields.

        The default implementation is a no-op. Subclasses that require information
        from the initial structure should override this method.
        """
        return self

    def build_cv(self) -> "CollectiveVariable":
        """Construct the collective variable object described by this config."""
        raise NotImplementedError


class CollectiveVariable:
    """Abstract base class for metadynamics collective variables (CVs)."""

    Config = CollectiveVariableConfig
    periodic: bool = False

    def __call__(self, graph: Graph) -> Array:
        """Compute the scalar value of this CV for the given graph."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _compute_vector(self, graph: Graph, pos_1: Array, pos_2: Array) -> Array:
        """Compute the displacement vector from `pos_2` to `pos_1`.

        Uses `graph.edges.displ_fun` so that periodic boundary conditions are
        respected automatically.

        Args:
            graph: Current simulation graph; must have a non-None `displ_fun`.
            pos_1: First positions, shape `(3,)` or `(N, 3)`.
            pos_2: Second positions, same shape as `pos1`.

        Returns:
            Displacement vector(s) of the same shape as the inputs.
        """
        if graph.edges.displ_fun is None:
            raise ValueError(
                "Graph must have a displacement function for this CollectiveVariable."
            )
        if pos_1.shape != pos_2.shape:
            raise ValueError("Input positions must have same dimensionality.")

        pos_1 = jnp.atleast_2d(pos_1)
        pos_2 = jnp.atleast_2d(pos_2)

        return graph.edges.displ_fun(pos_1, pos_2, graph.globals.cell)


# --- Distance CV ---


class DistanceCVConfig(CollectiveVariableConfig):
    """Config for a distance collective variable (CV) between two groups of atoms.

    Distance is computed between the centroid of each group of atoms, where each group
    contains one or more atoms.

    Can be passed to the `MetadynamicsConfig.bias_cvs` list to add a
    distance CV to the bias term of the metadynamics potential.

    Args:
        atom_indices_1: Indices specifying the first group of atoms.
            For a single atom, pass a one-element sequence, e.g. `[10]`.
        atom_indices_2: Indices specifying the second group of atoms.
            For a single atom, pass a one-element sequence, e.g. `[10]`.
        type: Discriminator field; always `"distance"`.
    """

    atom_indices_1: Sequence[int]
    atom_indices_2: Sequence[int]

    type: Literal["distance"] = pydantic.Field("distance", init=False)

    def build_cv(self) -> "DistanceCV":
        """Construct the :class:`DistanceCV` for this config."""
        return DistanceCV(self)


class DistanceCV(CollectiveVariable):
    """Distance collective variable (CV) between two groups of atoms.

    Distance is computed between the centroid of each group of atoms, where each group
    contains one or more atoms.
    """

    Config: DistanceCVConfig

    def __init__(self, config: DistanceCVConfig):
        self._config = config
        self.atom_indices_i = jnp.array(config.atom_indices_1)
        self.atom_indices_j = jnp.array(config.atom_indices_2)

    def __call__(self, graph: Graph) -> Array:
        """Return the distance between the configured atom pair (Å)."""
        positions = graph.nodes.positions
        positions_i = jnp.mean(positions[self.atom_indices_i], axis=0)
        positions_j = jnp.mean(positions[self.atom_indices_j], axis=0)
        vector = self._compute_vector(graph, positions_i, positions_j)
        return safe_sqrt(jnp.sum(vector**2))


# --- Angle CV ---


class AngleCVConfig(CollectiveVariableConfig):
    """Config for a bond-angle collective variable (CV).

    Bond-angle is computed for the triplet p–q–r, where q is the vertex atom.

    Can be passed to the `MetadynamicsConfig.bias_cvs` list to add a
    bond-angle CV to the bias term of the metadynamics potential.

    Attributes:
        atom_indices: Triplet of atom indices `(p, q, r)`, where q is the vertex atom.
        type: Discriminator field; always `"angle"`.
    """

    atom_indices: tuple[int, int, int]

    type: Literal["angle"] = pydantic.Field("angle", init=False)

    def build_cv(self) -> "AngleCV":
        """Construct the `AngleCV` for this config."""
        return AngleCV(self)


class AngleCV(CollectiveVariable):
    """Bond-angle collective variable (CV) in radians.

    Bond-angle is computed for the triplet p–q–r, where q is the vertex atom.
    """

    Config = AngleCVConfig

    def __init__(self, config: AngleCVConfig):
        self._config = config

    def __call__(self, graph: Graph) -> Array:
        """Return the p–q–r bond angle in radians."""
        positions = graph.nodes.positions
        positions_p = positions[self._config.atom_indices[0]]
        positions_q = positions[self._config.atom_indices[1]]
        positions_r = positions[self._config.atom_indices[2]]
        vec_1 = self._compute_vector(graph, positions_p, positions_q)
        vec_2 = self._compute_vector(graph, positions_r, positions_q)

        cos_theta = jnp.sum(vec_1 * vec_2) / (
            safe_sqrt(jnp.sum(vec_1**2)) * safe_sqrt(jnp.sum(vec_2**2))
        )
        return jnp.arccos(jnp.clip(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6))


# --- Dihedral CV ---


class DihedralCVConfig(CollectiveVariableConfig):
    """Config for a dihedral-angle collective variable (CV).

    Dihedral is computed for the quadruplet i–j–k–l, with range (-π, π).

    Can be passed to the `MetadynamicsConfig.bias_cvs` list to add a
    dihedral-angle CV to the bias term of the metadynamics potential.

    Attributes:
        atom_indices: Quadruplet of atom indices `(i, j, k, l)`.
        type: Discriminator field; always `"dihedral"`.
    """

    atom_indices: tuple[int, int, int, int]

    type: Literal["dihedral"] = pydantic.Field("dihedral", init=False)

    def build_cv(self) -> "DihedralCV":
        """Construct the `DihedralCV` for this config."""
        return DihedralCV(self)


class DihedralCV(CollectiveVariable):
    """Dihedral angle collective variable (CV) in radians

    Dihedral angle is computed for the quadruplet i–j–k–l, with range (-π, π).
    """

    Config = DihedralCVConfig
    periodic: bool = True

    def __init__(self, config: DihedralCVConfig):
        self._config = config

    def __call__(self, graph: Graph) -> jnp.ndarray:
        positions = graph.nodes.positions
        positions_i = positions[self._config.atom_indices[0]]
        positions_j = positions[self._config.atom_indices[1]]
        positions_k = positions[self._config.atom_indices[2]]
        positions_l = positions[self._config.atom_indices[3]]

        bond_ij = positions_j - positions_i
        bond_jk = positions_k - positions_j
        bond_kl = positions_l - positions_k
        normal_ijk = jnp.cross(bond_ij, bond_jk)
        normal_jkl = jnp.cross(bond_jk, bond_kl)
        bond_jk_unit = bond_jk / safe_sqrt(jnp.sum(bond_jk**2))
        normal_ijk_unit = normal_ijk / safe_sqrt(jnp.sum(normal_ijk**2))
        normal_jkl_unit = normal_jkl / safe_sqrt(jnp.sum(normal_jkl**2))
        tangent_ijk = jnp.cross(normal_ijk_unit, bond_jk_unit)
        return jnp.arctan2(
            jnp.dot(tangent_ijk, normal_jkl_unit),
            jnp.dot(normal_ijk_unit, normal_jkl_unit),
        )


# --- Coordination Number CV ---


class CoordinationNumberCVConfig(CollectiveVariableConfig):
    """Config for a coordination number collective variable (CV).

    Computes the sum of switching-function values over a set of neighbor atoms,
    yielding a differentiable approximation to integer coordination number.
    The switching function is evaluated using Horner's method::

        s(r) = (1 + (r/r0) + ... + (r/r0)^(nn-1))
             / (1 + (r/r0) + ... + (r/r0)^(mm-1))

    which approximates `(1 - (r/r0)^nn) / (1 - (r/r0)^mm)` for `r != r0`.

    Can be passed to the `MetadynamicsConfig.bias_cvs` list to add a
    coordination number CV to the bias term of the metadynamics potential.

    Attributes:
        central_idx: Index of the central atom.
        element: Element symbol of the neighbor element type to count (e.g. `"N"`).
        r0: Reference distance (Å) at which the switching function equals 0.5.
            Default 3.15.
        nn: Numerator exponent of the switching function. Default 12.
        mm: Denominator exponent; must satisfy `mm > nn`. Default 24.
        d_max: Hard distance cutoff (Å). Default 5.0.
        neighbor_indices: Atom indices of the neighbor element type, populated by
            calling `resolve`. Do not set manually.
        type: Discriminator field; always `"coordnum"`.
    """

    central_idx: int
    element: str
    r0: float = 3.15
    nn: int = 12
    mm: int = 24
    d_max: float = 5.0
    neighbor_indices: jnp.ndarray | None = None

    type: Literal["coordnum"] = pydantic.Field("coordnum", init=False)

    @pydantic.model_validator(mode="after")
    def _check_not_manually_set(self) -> "CoordinationNumberCVConfig":
        if self.neighbor_indices is not None:
            raise ValueError(
                "`neighbor_indices` is set by `resolve()` Do not set manually."
            )
        return self

    def resolve(self, atoms: Atoms) -> "CoordinationNumberCVConfig":
        """Populate `neighbor_indices` using `atoms`."""
        target_z = ase_data.atomic_numbers[self.element]
        indices = jnp.array(np.where(atoms.numbers == target_z)[0])
        return self.model_copy(update={"neighbor_indices": indices})

    def build_cv(self) -> "CoordinationNumberCV":
        """Construct the `CoordinationNumberCV` for this config."""
        if self.neighbor_indices is None:
            raise ValueError(
                "`neighbor_indices` is None. Call `resolve` before `build_cv()`."
            )
        return CoordinationNumberCV(self)


class CoordinationNumberCV(CollectiveVariable):
    """Coordination number collective variable (CV) via a rational switching function.

    Computes the sum of switching-function values over a set of neighbor atoms,
    yielding a differentiable approximation to integer coordination number.
    The switching function is evaluated using Horner's method::

        s(r) = (1 + (r/r0) + ... + (r/r0)^(nn-1))
             / (1 + (r/r0) + ... + (r/r0)^(mm-1))

    which approximates `(1 - (r/r0)^nn) / (1 - (r/r0)^mm)` for `r != r0`.
    """

    Config = CoordinationNumberCVConfig

    def __init__(self, config: CoordinationNumberCVConfig):
        self._config = config

    def __call__(self, graph: Graph) -> Array:
        """Return the coordination number of the central atom."""
        positions = graph.nodes.positions
        pos_central = positions[self._config.central_idx]
        pos_neighbors = positions[self._config.neighbor_indices]
        pos_central_broadcast = jnp.broadcast_to(pos_central, pos_neighbors.shape)
        vectors = self._compute_vector(graph, pos_neighbors, pos_central_broadcast)

        distances = safe_sqrt(jnp.sum(vectors**2, axis=1))
        norm_distance = distances / self._config.r0
        numerator = jnp.ones_like(norm_distance)
        for _ in range(self._config.nn - 1):
            numerator = 1.0 + norm_distance * numerator
        denominator = jnp.ones_like(norm_distance)
        for _ in range(self._config.mm - 1):
            denominator = 1.0 + norm_distance * denominator
        s = numerator / denominator
        return jnp.sum(jnp.where(distances < self._config.d_max, s, 0.0))
