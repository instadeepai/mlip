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

import ase
import pydantic
from ase.units import kB

from mlip.simulation.configs.jax_md_config import JaxMDSimulationConfig
from mlip.simulation.metadynamics.potential_terms import (
    BiasPotential,
    BiasPotential1D,
    BiasPotential2D,
    CollectiveVariableConfig,
    RestraintPotential,
    RestraintPotentialConfig,
    WallPotential,
    WallPotentialConfig,
)


class MetadynamicsConfig(pydantic.BaseModel):
    """User-facing configuration for a metadynamics simulation.

    Attributes:
        bias_cvs: One or two collective variable (CV) configurations defining the bias
            coordinates. Gaussian hills are deposited along these CVs.
        bias_sigmas: One or two sigma values to use for depositing Gaussian hills
            on each of the specified bias CVs. Units match those of the CVs.
        walls: Wall potential configurations, used to constrain coordinates.
        restraints: Positional restraint configurations, used to fix atoms near their
            initial positions.
        initial_height: Initial Gaussian hill height in eV.
        bias_factor: Well-tempered bias factor for rescaling hill heights at
            each deposition. Set to `None` for plain (untempered) metadynamics.
        deposition_interval: Number of simulation steps between Gaussian hill
            depositions.
        max_gaussians: Maximum number of Gaussian hills to store.
        thermal_energy_ev: Thermal energy (k_B * T) in eV, populated by calling
            `resolve`. Do not set manually.
    """

    bias_cvs: list[CollectiveVariableConfig]
    bias_sigmas: list[float]
    walls: list[WallPotentialConfig] = []
    restraints: list[RestraintPotentialConfig] = []

    initial_height: float
    bias_factor: float | None
    deposition_interval: int
    max_gaussians: int

    thermal_energy_ev: float | None = None

    @pydantic.model_validator(mode="after")
    def _check_not_manually_set(self) -> "MetadynamicsConfig":
        if self.thermal_energy_ev is not None:
            raise ValueError(
                "`thermal_energy_ev` is set by `resolve()` Do not set manually."
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_bias_cvs_count(self) -> "MetadynamicsConfig":
        if len(self.bias_cvs) > 2:
            raise ValueError(f"Only 1-2 bias CVs supported. Got {len(self.bias_cvs)}.")
        if len(self.bias_cvs) != len(self.bias_sigmas):
            raise ValueError("Must provide the same number of bias CVs and sigmas.")
        return self

    def resolve(
        self, atoms: ase.Atoms, temperature_kelvin: float
    ) -> "MetadynamicsConfig":
        """Return a resolved copy with all information required for metadynamics.

        Args:
            atoms: Initial structure used to resolve CVs and restraints that require
                system-derived information.
            temperature_kelvin: Simulation temperature in Kelvin, used to compute
                `thermal_energy_ev`.
        """
        return self.model_copy(
            update={
                "bias_cvs": [cv_cfg.resolve(atoms) for cv_cfg in self.bias_cvs],
                "restraints": [r_cfg.resolve(atoms) for r_cfg in self.restraints],
                "thermal_energy_ev": kB * temperature_kelvin,
            }
        )

    def build_bias_potential(self) -> BiasPotential:
        """Construct the Gaussian-hill bias potential from the configured CVs."""
        cvs = [cv_cfg.build_cv() for cv_cfg in self.bias_cvs]
        sigmas = self.bias_sigmas
        if len(cvs) == 1:
            return BiasPotential1D(cvs[0], sigmas[0])
        return BiasPotential2D(cvs[0], cvs[1], sigmas[0], sigmas[1])

    def build_wall_potentials(self) -> list[WallPotential]:
        """Construct all configured wall potentials."""
        wall_potentials = []
        for wall_cfg in self.walls:
            wall_potentials.extend(wall_cfg.build_walls())
        return wall_potentials

    def build_restraint_potentials(self) -> list[RestraintPotential]:
        """Construct all configured positional restraints."""
        return [r_cfg.build_restraint() for r_cfg in self.restraints]


class JaxMDMetadynamicsSimulationConfig(JaxMDSimulationConfig):
    """Simulation config for a JAX-MD metadynamics run.

    Extends `JaxMDSimulationConfig` with a `metadynamics_config` field that
    controls hill deposition, collective variables, walls, and restraints.

    Attributes:
        metadynamics_config: Metadynamics-specific settings including CVs,
            sigmas, bias factor, and deposition interval.
    """

    metadynamics_config: MetadynamicsConfig
