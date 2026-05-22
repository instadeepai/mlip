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
import random
import time

import ase
import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import BFGS

from mlip.simulation.ase.ase_montecarlo_barostat import ASEMonteCarloBarostat
from mlip.simulation.ase.mlip_ase_calculator import MLIPForceFieldASECalculator
from mlip.simulation.configs.ase_config import ASESimulationConfig
from mlip.simulation.enums import MDIntegrator, SimulationType
from mlip.simulation.simulation_engine import ForceField, SimulationEngine
from mlip.simulation.state import SimulationState
from mlip.simulation.temperature_scheduling import get_temperature_schedule
from mlip.simulation.utils import (
    has_simulation_exploded,
    resolve_atoms_charge_for_model,
)

SIMULATION_RANDOM_SEED = 42

logger = logging.getLogger("mlip")


class ASESimulationEngine(SimulationEngine):
    """Simulation engine handling simulations with the ASE backend.

    For MD, the NVT-Langevin algorithm is used
    (see `here <https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md.langevin>`__).
    For energy minimization, the BFGS algorithm is used
    (see `here <https://wiki.fysik.dtu.dk/ase/ase/optimize.html#ase.optimize.BFGS>`__).
    """

    Config = ASESimulationConfig

    def __init__(
        self,
        atoms: ase.Atoms,
        force_field: ForceField,
        config: ASESimulationConfig,
    ) -> None:
        super().__init__(atoms, force_field, config)

    def _initialize(
        self,
        atoms: ase.Atoms,
        force_field: ForceField,
        config: ASESimulationConfig,
    ) -> None:
        if isinstance(atoms, list):
            raise ValueError("Batched simulations not supported with ASE backend.")

        logger.debug("Initialization of simulation begins...")
        self._config = config
        self.atoms = atoms
        self.atoms.center()
        self._init_box()
        self.atoms = resolve_atoms_charge_for_model(
            self.atoms, force_field, self._config.set_none_charge_to_zero
        )

        positions = atoms.get_positions()
        self._num_atoms = positions.shape[0]
        self.state.atomic_numbers = self.atoms.numbers

        self._force_field = force_field
        self.is_md_simulation = self._config.simulation_type == SimulationType.MD
        self.is_npt_simulation = (
            self.is_md_simulation and self._config.md_integrator.ensemble == "npt"
        )
        self.model_calculator = MLIPForceFieldASECalculator(
            self.atoms,
            self._config.edge_capacity_multiplier,
            force_field,
        )

        self._temperature_schedule = get_temperature_schedule(
            self._config.temperature_schedule_config, self._config.num_steps
        )

        logger.debug("Initialization of simulation completed.")

    def _init_box(self) -> None:
        """Update the PBC parameters of the underlying `ase.Atoms`"""
        # Pass if atoms already have PBC and cell, best source of truth
        if np.any(self.atoms.cell) or np.any(self.atoms.pbc):
            logger.warning(
                "Ignoring `box` parameter as `atoms` already has PBC configured."
            )
            return
        # Support cubic periodic box from config for Jax-MD consistency.
        # To be discouraged once both engines support arbitrary lattices.
        if isinstance(self._config.box, float):
            self.atoms.cell = np.eye(3) * self._config.box
            self.atoms.pbc = True
        elif isinstance(self._config.box, list):
            self.atoms.cell = np.diag(np.array(self._config.box))
            self.atoms.pbc = True
        else:
            self.atoms.cell = None
            self.atoms.pbc = False

    def _setup_montecarlo_barostat(self, dyn: Langevin) -> ASEMonteCarloBarostat:
        """Setup the MonteCarloBarostat for `NPT_MC_LANGEVIN` simulations."""

        def _temperature_getter():
            return dyn.todict()["temperature_K"]

        return ASEMonteCarloBarostat(
            self.atoms,
            self.model_calculator,
            pressure_bar=self._config.pressure_bar,
            molecule_indices=np.array(self._config.molecule_indices),
            temperature_getter=_temperature_getter,
            random_seed=SIMULATION_RANDOM_SEED,
        )

    def run(self) -> None:
        """See documentation of abstract parent class.
        This runs the simulation using the ASE backend.

        Important: The state of the simulation is updated and the loggers are called
        during this function.
        """
        logger.info("Starting simulation...")
        self.atoms.calc = self.model_calculator
        random.seed(SIMULATION_RANDOM_SEED)
        _rng = np.random.default_rng(SIMULATION_RANDOM_SEED)

        if self.is_md_simulation:
            if self.atoms.get_velocities() is None or np.all(
                self.atoms.get_velocities() == 0.0
            ):
                # Set random velocities according to Maxwell-Boltzmann distribution
                MaxwellBoltzmannDistribution(
                    self.atoms,
                    temperature_K=self._config.temperature_kelvin,
                    rng=_rng,
                )
                Stationary(self.atoms)
                ZeroRotation(self.atoms)

            dyn = Langevin(
                self.atoms,
                timestep=self._config.timestep_fs * units.fs,
                temperature_K=self._config.temperature_kelvin,
                friction=self._config.friction,
                rng=_rng,
            )

            if self._config.md_integrator == MDIntegrator.NPT_MC_LANGEVIN:
                if self._config.molecule_indices is None:
                    raise ValueError(
                        "Molecule indices must be set for NPT simulations."
                    )
                barostat = self._setup_montecarlo_barostat(dyn)
                dyn.attach(
                    barostat.step, interval=self._config.barostat_update_interval
                )
        elif self._config.simulation_type == SimulationType.MINIMIZATION:
            dyn = BFGS(self.atoms, logfile=None)
        else:
            raise NotImplementedError(
                f"{self._config.simulation_type=} not implemented for ASE backend"
            )

        def update_temporary_state() -> None:
            """Update the internal temporary SimulationState object."""
            self._update_temporary_state()

        def update_state() -> None:
            """Update the internal SimulationState object using the temporary state."""
            step = dyn.get_number_of_steps()
            compute_time = time.perf_counter() - self.self_start_interval_time
            self._update_state(step, compute_time)

        def log_to_console() -> None:
            """Logs info to console."""
            step = dyn.get_number_of_steps()
            compute_time = time.perf_counter() - self.self_start_interval_time
            self._log_to_console(step, compute_time)

        def update_temperature() -> None:
            """Update the temperature if a temperature schedule is given."""
            cur_step = dyn.get_number_of_steps()
            temperature_kelvin = self._temperature_schedule(cur_step)
            dyn.set_temperature(temperature_K=temperature_kelvin)

        def begin_new_log_interval() -> None:
            """Setup variables required at each log_interval steps."""
            self.self_start_interval_step = dyn.get_number_of_steps()
            self.self_start_interval_time = time.perf_counter()
            self.temporary_state = SimulationState()

        dyn.attach(update_temporary_state, interval=self._config.snapshot_interval)
        dyn.attach(update_state, interval=self._config.log_interval)
        dyn.attach(log_to_console, interval=self._config.log_interval)
        dyn.attach(self._call_loggers, interval=self._config.log_interval)
        # Every self._config.log_interval steps, we log. At the end of this logging, we
        # set the beginning of this new interval in order to calculate total compute
        # time

        if self.is_md_simulation:
            dyn.attach(update_temperature)

        dyn.attach(begin_new_log_interval, interval=self._config.log_interval)
        # Begin the first log interval
        begin_new_log_interval()

        if self.is_md_simulation:
            run_args = {"steps": self._config.num_steps}
        elif self._config.simulation_type == SimulationType.MINIMIZATION:
            run_args = {
                "steps": self._config.num_steps,
                "fmax": self._config.max_force_convergence_threshold,
            }

        for _ in dyn.irun(**run_args):
            if self._has_simulation_exploded():
                logger.warning("Simulation exploded. Stopping simulation early.")
                # Update state and call loggers before exiting
                update_state()
                log_to_console()
                self._call_loggers()
                break

        logger.info("Simulation completed.")

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
                self.self_start_interval_step,
                step,
                compute_time,
            )

    def _update_temporary_list(self, name: str, new: np.ndarray) -> None:
        """Update one field of the temporary state as a list."""
        new = new.astype(np.float32)  # Convert to float32 for lower memory usage.
        current_value = getattr(self.temporary_state, name)
        if current_value is None:
            setattr(self.temporary_state, name, [new])
        else:
            current_value.append(new)

    def _update_temporary_state(self) -> None:
        """Update the internal temporary state of the simulation."""
        log_outputs = self._config.log_outputs

        if log_outputs.positions:
            self._update_temporary_list("positions", self.atoms.get_positions())
        if log_outputs.forces:
            self._update_temporary_list("forces", self.atoms.get_forces())

        if self.is_md_simulation:
            if log_outputs.temperature:
                self._update_temporary_list("temperature", self.atoms.get_temperature())
            if log_outputs.kinetic_energy:
                self._update_temporary_list(
                    "kinetic_energy", self.atoms.get_kinetic_energy()
                )
            if log_outputs.velocities:
                self._update_temporary_list("velocities", self.atoms.get_velocities())

            if log_outputs.cell and self.is_npt_simulation:
                self._update_temporary_list("cell", self.atoms.get_cell().array)

    def _update_state_array(self, name: str) -> None:
        """Update one field of the state as a numpy array."""
        state_value = getattr(self.state, name)
        temp_value = getattr(self.temporary_state, name)
        if temp_value is None:  # No new values to add
            return

        new = np.stack(temp_value, axis=0)
        if state_value is None:
            setattr(self.state, name, new)
        else:
            setattr(self.state, name, np.concatenate([state_value, new], axis=0))

    def _update_state(self, step: int, compute_time: float) -> None:
        """Update the internal state of the simulation, using the temporary state.

        Args:
            step: The current step of the simulation
            compute_time: The time spent in the last interval
        """
        log_outputs = self._config.log_outputs

        if log_outputs.positions:
            self._update_state_array("positions")
        if log_outputs.forces:
            self._update_state_array("forces")
        self.state.step = step
        self.state.compute_time_seconds += compute_time

        if self.is_md_simulation:
            if log_outputs.temperature:
                self._update_state_array("temperature")
            if log_outputs.kinetic_energy:
                self._update_state_array("kinetic_energy")
            if log_outputs.velocities:
                self._update_state_array("velocities")

            if log_outputs.cell and self.is_npt_simulation:
                self._update_state_array("cell")

    def _has_simulation_exploded(self) -> bool:
        """Check if the simulation has exploded."""
        if not self.is_md_simulation:
            return False
        return has_simulation_exploded(self.atoms.get_temperature())
