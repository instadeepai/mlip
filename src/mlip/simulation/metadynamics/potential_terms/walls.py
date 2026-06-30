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

import jax.numpy as jnp
import pydantic

from mlip.graph import Graph
from mlip.simulation.metadynamics.potential_terms.collective_variables import (
    AngleCVConfig,
    DistanceCVConfig,
)

# --- Base WallPotential classes ---


class WallPotentialConfig(pydantic.BaseModel):
    """Abstract base for wall potential config classes.

    `WallPotentialConfig`s can be passed to the `MetadynamicsConfig.walls`
    list to add wall potential terms to the metadynamics potential.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def build_walls(self) -> list["WallPotential"]:
        """Construct the wall potential object(s) described by this config."""
        raise NotImplementedError


class WallPotential:
    """Base class for one-sided wall potentials applied to a collective variable."""

    Config = WallPotentialConfig

    def __call__(self, graph: Graph) -> jnp.ndarray:
        """Return the wall potential energy (eV) for the given graph."""
        raise NotImplementedError("This method should be implemented by subclasses.")


# --- Distance walls ---


class DistanceWallConfig(WallPotentialConfig):
    """Config for a distance wall potential between two (groups of) atoms.

    Distance is computed between the centroid of each group of atoms, where each group
    contains one or more atoms.

    Can be used to configure a lower wall, an upper wall, or both.

    Can be passed to the `MetadynamicsConfig.walls` list to add
    pairwise distance wall term(s) to the metadynamics potential.

    Attributes:
        atom_indices_1: Indices specifying the first group of atoms.
            For a single atom, pass a one-element sequence, e.g. `[10]`.
        atom_indices_2: Indices specifying the second group of atoms.
            For a single atom, pass a one-element sequence, e.g. `[10]`.
        kappa: Wall force constant in eV / Å^exp.
        lower: Lower wall threshold in Å. Penalty when distance < lower.
            If None, no lower wall is added.
        upper: Upper wall threshold in Å. Penalty when distance > upper.
            If None, no upper wall is added.
        exp: Wall exponent. Default = 2 (harmonic wall).
        type: Discriminator field; always `"distance"`.
    """

    atom_indices_1: Sequence[int]
    atom_indices_2: Sequence[int]
    kappa: float
    lower: float | None = None
    upper: float | None = None
    exp: int = 2

    type: Literal["distance"] = pydantic.Field("distance", init=False)

    def build_walls(self) -> list[WallPotential]:
        """Construct lower and/or upper distance wall potentials."""
        walls = []
        if self.lower is not None:
            walls.append(LowerDistanceWallPotential(self))
        if self.upper is not None:
            walls.append(UpperDistanceWallPotential(self))
        return walls


class LowerDistanceWallPotential(WallPotential):
    """Lower wall on a distance CV: V = kappa * max(lower - distance, 0)^exp.

    Penalises configurations where the distance falls below `config.lower`.
    """

    Config = DistanceWallConfig

    def __init__(self, config: DistanceWallConfig):
        assert config.lower is not None
        self._config = config

        self.distance_cv = DistanceCVConfig(
            atom_indices_1=config.atom_indices_1,
            atom_indices_2=config.atom_indices_2,
        ).build_cv()

    def __call__(self, graph: Graph) -> jnp.ndarray:
        """Return the wall energy (eV); non-zero when distance is below `lower`."""
        distance = self.distance_cv(graph)
        excess = self._config.lower - distance
        safe_excess = jnp.where(excess > 0.0, excess, 0.0)
        return self._config.kappa * safe_excess**self._config.exp


class UpperDistanceWallPotential(WallPotential):
    """Upper wall on a distance CV: V = kappa * max(distance - upper, 0)^exp.

    Penalises configurations where the distance exceeds `config.upper`.
    """

    Config = DistanceWallConfig

    def __init__(self, config: DistanceWallConfig):
        assert config.upper is not None
        self._config = config

        self.distance_cv = DistanceCVConfig(
            atom_indices_1=config.atom_indices_1,
            atom_indices_2=config.atom_indices_2,
        ).build_cv()

    def __call__(self, graph: Graph) -> jnp.ndarray:
        """Return the wall energy (eV); non-zero when distance exceeds `upper`."""
        distance = self.distance_cv(graph)
        excess = distance - self._config.upper
        safe_excess = jnp.where(excess > 0.0, excess, 0.0)
        return self._config.kappa * safe_excess**self._config.exp


# --- Angle walls ---


class AngleWallConfig(WallPotentialConfig):
    """Config for a bond-angle wall potential.

    Bond-angle is computed for the triplet p–q–r, where q is the vertex atom.

    Can be used to configure a lower wall, an upper wall, or both.

    Can be passed to the `MetadynamicsConfig.walls` list to add
    bond-angle wall term(s) to the metadynamics potential.

    Attributes:
        atom_indices: Triplet of atom indices `(p, q, r)` where q is the vertex.
        kappa: Wall force constant in eV / rad^exp.
        lower_rad: Lower wall threshold in radians. Penalty when angle < lower.
            If None, no lower wall is added.
        upper_rad: Upper wall threshold in radians. Penalty when angle > upper.
            If None, no upper wall is added.
        exp: Wall exponent. Default = 2 (harmonic wall).
        type: Discriminator field; always `"angle"`.
    """

    atom_indices: tuple[int, int, int]
    kappa: float
    lower_rad: float | None = None
    upper_rad: float | None = None
    exp: int = 2

    type: Literal["angle"] = pydantic.Field("angle", init=False)

    def build_walls(self) -> list[WallPotential]:
        """Construct lower and/or upper angle wall potentials."""
        walls = []
        if self.lower_rad is not None:
            walls.append(LowerAngleWallPotential(self))
        if self.upper_rad is not None:
            walls.append(UpperAngleWallPotential(self))
        return walls


class LowerAngleWallPotential(WallPotential):
    """Lower wall on an angle CV: V = kappa * (lower - θ)^exp for θ < lower.

    Penalises configurations where the angle falls below `config.lower_rad`.
    """

    Config = AngleWallConfig

    def __init__(self, config: AngleWallConfig):
        assert config.lower_rad is not None
        self._config = config
        self.angle_cv = AngleCVConfig(atom_indices=config.atom_indices).build_cv()

    def __call__(self, graph: Graph) -> jnp.ndarray:
        """Return the wall energy (eV); non-zero when angle is below `lower_rad`."""
        theta = self.angle_cv(graph)
        excess = self._config.lower_rad - theta
        safe_excess = jnp.where(excess > 0.0, excess, 0.0)
        return self._config.kappa * safe_excess**self._config.exp


class UpperAngleWallPotential(WallPotential):
    """Upper wall on an angle CV: V = kappa * (θ - upper)^exp for θ > upper.

    Penalises configurations where the angle exceeds `config.upper_rad`.
    """

    Config = AngleWallConfig

    def __init__(self, config: AngleWallConfig):
        assert config.upper_rad is not None
        self._config = config
        self.angle_cv = AngleCVConfig(atom_indices=config.atom_indices).build_cv()

    def __call__(self, graph: Graph) -> jnp.ndarray:
        """Return the wall energy (eV); non-zero when angle exceeds `at_rad`."""
        theta = self.angle_cv(graph)
        excess = theta - self._config.upper_rad
        safe_excess = jnp.where(excess > 0.0, excess, 0.0)
        return self._config.kappa * safe_excess**self._config.exp
