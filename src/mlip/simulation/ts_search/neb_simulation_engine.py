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

import ase
import numpy as np
from ase.mep import NEB
from ase.optimize import BFGS, FIRE

from mlip.simulation.ase.mlip_ase_calculator import MLIPForceFieldASECalculator
from mlip.simulation.configs.neb_config import NEBSimulationConfig
from mlip.simulation.enums import StructureOptimizationMethod
from mlip.simulation.simulation_engine import ForceField, SimulationEngine
from mlip.simulation.state import NEBSimulationState

logger = logging.getLogger("mlip")


class NEBSimulationEngine(SimulationEngine):
    """Simulation engine handling Nudged Elastic Band (NEB) transition state
    searches using the ASE backend.
    """

    Config = NEBSimulationConfig
    simulation_state_class = NEBSimulationState

    def __init__(
        self,
        atoms: list[ase.Atoms],
        force_field: ForceField,
        config: NEBSimulationConfig,
    ) -> None:
        """Constructor.

        Attributes:
            atoms: List of `Atoms` objects. If the list has two elements, these will be
                   considered the initial and final images of the NEB. If the list has
                   three elements, the image in the middle will be considered the
                   transition state guess. If the list has more elements, interpolation
                   will be skipped. In case of two or three provided images, the initial
                   images in between will be interpolated using the IDPP method to yield
                   the total number of images specified in the config.
            force_field: Force field model used to compute the predictions.
            config: Configuration for the NEB simulation.
        """
        super().__init__(atoms, force_field, config)

    def _initialize(
        self,
        atoms: list[ase.Atoms],
        force_field: ForceField,
        config: NEBSimulationConfig,
    ) -> None:
        """Initialize the NEB simulation."""
        self.state.potential_energy = None
        self._config = config

        positions = atoms[0].get_positions()
        self._num_atoms = positions.shape[0]
        self.state.atomic_numbers = atoms[0].numbers
        self.force_field = force_field
        self.images = atoms

        self.model_calculator = self._get_model_calculator()

        for image in self.images:
            self._init_box_neb(image)

        self.images[0].calc = self._get_model_calculator()
        self.images[-1].calc = self._get_model_calculator()

        self.neb = NEB([])

    def run(self) -> None:
        """Run the NEB simulation."""
        if self._config.continue_from_previous_run:
            if len(self.images) != self._config.num_images:
                logger.warning(
                    "Number of images in config is different from number of provided"
                    " images. Ignoring number of images in config."
                )

            for image in self.images:
                image.calc = self._get_model_calculator()

            self.neb = NEB(
                self.images,
                k=self._config.neb_k,
                climb=self._config.climb,
                method=self._config.neb_method.value,
                parallel=False,
            )
        elif len(self.images) <= 3:
            self._init_neb()
        else:
            for image in self.images:
                image.calc = self._get_model_calculator()

            self.neb = NEB(
                self.images,
                k=self._config.neb_k,
                climb=self._config.climb,
                method=self._config.neb_method.value,
                parallel=False,
            )

        if self._config.optimizer == StructureOptimizationMethod.BFGS:
            dyn = BFGS(
                self.neb,
                alpha=self._config.bfgs_alpha,
                maxstep=self._config.bfgs_maxstep,
                logfile=None,
            )
        elif self._config.optimizer == StructureOptimizationMethod.FIRE:
            dyn = FIRE(self.neb, dt=self._config.fire_timestep, logfile=None)
        else:
            logger.warning("Optimizer not implemented. Using BFGS.")
            dyn = BFGS(
                self.neb,
                alpha=self._config.bfgs_alpha,
                maxstep=self._config.bfgs_maxstep,
                logfile=None,
            )

        def log_to_console() -> None:
            """Logs info to console."""
            step = dyn.get_number_of_steps()
            compute_time = time.perf_counter() - self.self_start_interval_time
            self._log_to_console(step, compute_time)

        def set_beginning_interval_time() -> None:
            self.self_start_interval_time = time.perf_counter()

        def update_state() -> None:
            """Update the internal SimulationState object."""
            step = dyn.get_number_of_steps()
            compute_time = time.perf_counter() - self.self_start_interval_time
            self._update_state_neb(step, compute_time)

        dyn.attach(log_to_console, interval=self._config.log_interval)
        dyn.attach(self._call_loggers, interval=self._config.log_interval)
        dyn.attach(update_state, interval=self._config.snapshot_interval)
        dyn.attach(set_beginning_interval_time, interval=self._config.log_interval)
        self.self_start_interval_time = time.perf_counter()

        dyn.run(
            steps=self._config.num_steps,
            fmax=self._config.max_force_convergence_threshold,
        )

    def _init_neb(self) -> None:
        """Initialize the NEB setup. Interplolate between images using IDPP."""
        if len(self.images) not in (2, 3):
            raise ValueError("`init_neb` is only supported for 2 and 3 images.")

        num_images = max(self._config.num_images, len(self.images))

        if num_images == len(self.images):
            images = list(self.images)

        elif len(self.images) == 2:
            images = [self.images[0]]
            images.extend([self.images[0].copy() for _ in range(num_images - 2)])
            images.append(self.images[-1])

        else:
            num_images_1 = num_images // 2 + 1
            num_images_2 = num_images - num_images_1 + 1

            images_1 = [self.images[0]]
            images_1.extend([self.images[0].copy() for _ in range(num_images_1 - 2)])
            images_1.append(self.images[1])

            images_2 = [self.images[1].copy()]
            images_2.extend([self.images[-1].copy() for _ in range(num_images_2 - 2)])
            images_2.append(self.images[-1])

            for image in images_1:
                image.calc = self._get_model_calculator()
            for image in images_2:
                image.calc = self._get_model_calculator()

            neb1 = NEB(
                images_1,
                k=self._config.neb_k,
                climb=self._config.climb,
                method=self._config.neb_method.value,
                parallel=False,
            )
            neb2 = NEB(
                images_2,
                k=self._config.neb_k,
                climb=self._config.climb,
                method=self._config.neb_method.value,
                parallel=False,
            )

            neb1.interpolate(method="idpp")
            neb2.interpolate(method="idpp")

            images = neb1.images + neb2.images[1:]

        for image in images:
            image.calc = self._get_model_calculator()

        self.neb = NEB(
            images,
            k=self._config.neb_k,
            climb=self._config.climb,
            method=self._config.neb_method.value,
            parallel=False,
        )

        if len(self.images) == 2 and num_images > 2:
            self.neb.interpolate(method="idpp")

    def _init_box_neb(self, atoms: ase.Atoms) -> None:
        if isinstance(self._config.box, float):
            atoms.cell = np.eye(3) * self._config.box
            atoms.pbc = True
        elif isinstance(self._config.box, list):
            atoms.cell = np.diag(np.array(self._config.box))
            atoms.pbc = True
        else:
            atoms.cell = None
            atoms.pbc = False

    def _update_state_neb(
        self,
        step: int,
        compute_time: float,
    ) -> None:
        """Update the internal state of the simulation.
        Here, the positions for every image
        are updated and not concatenated, as for the MD simulations and energy
        minimizations.

        Args:
            step: The current step.
            compute_time: The compute time.
        """
        self.state.positions = np.zeros((
            len(self.neb.images),
            len(self.neb.images[0].positions),
            3,
        ))
        potential_energy = np.zeros(len(self.neb.images))
        forces_real = np.zeros((
            len(self.neb.images),
            len(self.neb.images[0].positions),
            3,
        ))

        for i, image in enumerate(self.neb.images):
            self.state.positions[i] = image.positions
            potential_energy[i] = image.get_potential_energy()

        if not self.state.potential_energy:
            self.state.potential_energy = [potential_energy]
        else:
            self.state.potential_energy.append(potential_energy)

        if not self.state.forces:
            self.state.forces = [self.neb.get_forces()]
        else:
            self.state.forces.append(self.neb.get_forces())

        for i, image in enumerate(self.neb.images):
            forces_real[i] = image.get_forces()

        if not self.state.forces_real:
            self.state.forces_real = [forces_real]
        else:
            self.state.forces_real.append(forces_real)

        self.state.step = step
        self.state.compute_time_seconds += compute_time

    def _get_model_calculator(self) -> MLIPForceFieldASECalculator:
        return MLIPForceFieldASECalculator(
            self.images[0], self._config.edge_capacity_multiplier, self.force_field
        )

    def _call_loggers(self) -> None:
        for _logger in self.loggers:
            _logger(self.state)

    def _log_to_console(self, step: int, compute_time: float) -> None:
        """Logs timing information to console via our logger."""
        if step == 0:
            logger.debug(
                "Initialization took %.2f seconds.",
                compute_time,
            )
        else:
            logger.info(
                "Steps %s to %s completed in %.2f seconds.",
                self.state.step,
                step,
                compute_time,
            )
