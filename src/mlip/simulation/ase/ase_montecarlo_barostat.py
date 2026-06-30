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
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator as ASECalculator
from ase.calculators.calculator import all_changes

from mlip.simulation.ase.mlip_ase_calculator import MLIPForceFieldASECalculator
from mlip.simulation.montecarlo_barostat import (
    INITIAL_MAX_DELTA_VOLUME_FRACTION,
    MonteCarloBarostatState,
    accept_volume_change,
    create_high_precision_force_field,
    propose_volume_change,
    sanitize_molecule_indices,
    tune_barostat,
)

logger = logging.getLogger("mlip")


class ASEMonteCarloBarostat:
    def __init__(
        self,
        atoms: Atoms,
        base_calculator: MLIPForceFieldASECalculator | ASECalculator,
        temperature_getter: Callable[[], float],
        pressure_bar: float,
        molecule_indices: np.ndarray,
        random_seed: int,
    ):
        """MonteCarloBarostat for ASE that reproduces the OpenMM MonteCarloBarostat.

        The `step` method of this class should be attached to an ASE dynamics object,
        with the `interval` parameter set to the number of steps between barostat
        updates. For example:
        ```
        dyn = ase.md.Langevin(atoms, ...)
        barostat = ASEMonteCarloBarostat(atoms, ...)
        dyn.attach(barostat.step, interval=25)
        ```

        The `step` method updates the positions and cell of the shared `atoms` object
        in place, by proposing a volume change, and accepting or rejecting based on
        a Metropolis criterion:
            acceptance_probability = exp(-Delta_W / kB * T)
            where Delta_W = Delta_E + P * Delta_V - N_mol * kB * T * ln(V_new / V_old)

        See the OpenMM documentation for more details:
        docs.openmm.org/latest/userguide/theory/02_standard_forces.html#montecarlobarostat

        Args:
            atoms: The ase.Atoms object.
            base_calculator: The base calculator to use for the energy calculation.
                             Either an mlip `MLIPForceFieldASECalculator` or any
                             external ASE calculator. Note that external ASE calculators
                             are used as-is; unlike the mlip path they do not
                             receive the high-precision/deterministic energy
                             guarantees `create_high_precision_force_field` provides.
            temperature_getter: Function to get the external temperature in Kelvin.
            pressure_bar: Target (external) pressure in bar.
            molecule_indices: Topology definition. Example for 2 water molecules:
                              [0, 0, 0, 1, 1, 1].
            random_seed: Random seed for the Monte Carlo Barostat.
        """
        self.atoms = atoms
        self.temperature_getter = temperature_getter
        self._calculator = self._setup_calculator(base_calculator)

        molecule_indices = sanitize_molecule_indices(molecule_indices)
        self.num_molecules = molecule_indices.max() + 1
        molecule_counts = np.bincount(molecule_indices)

        self.barostat_state = MonteCarloBarostatState(
            target_pressure=pressure_bar * units.bar,
            max_delta_volume=atoms.get_volume() * INITIAL_MAX_DELTA_VOLUME_FRACTION,
            num_attempted=0,
            num_accepted=0,
            num_attempted_since_tune=0,
            num_accepted_since_tune=0,
            mol_counts=jnp.array(molecule_counts),
            mol_indices=jnp.array(molecule_indices),
            rng=jax.random.PRNGKey(random_seed),
        )

        self._propose_step = jax.jit(propose_volume_change)
        self._accept_step = jax.jit(accept_volume_change)
        self._tune_barostat_step = jax.jit(tune_barostat)

    def _setup_calculator(
        self, base_calculator: MLIPForceFieldASECalculator | ASECalculator
    ) -> MLIPForceFieldASECalculator | ASECalculator:
        """Setup the calculator to use for the energy calculation in the barostat step.

        The Metropolis criterion requires high-precision and deterministic energy
        predictions, so we adapt the force field accordingly.
        """
        if not isinstance(base_calculator, MLIPForceFieldASECalculator):
            return base_calculator

        barostat_force_field = create_high_precision_force_field(
            base_calculator.force_field
        )
        return MLIPForceFieldASECalculator(
            self.atoms,
            base_calculator.edge_capacity_multiplier,
            barostat_force_field,
            allow_nodes_to_change=base_calculator.allow_nodes_to_change,
            node_capacity_multiplier=base_calculator.node_capacity_multiplier,
        )

    def step(self):
        """Performs the update step of the Monte Carlo Barostat."""
        # Compute energy of the current state of the system
        self._calculator.calculate(
            self.atoms, properties=["energy"], system_changes=all_changes
        )
        energy_old = self._calculator.results["energy"]

        # Store old state info in case we reject the move
        pos_old = jnp.array(self.atoms.get_positions())
        box_old = jnp.array(self.atoms.get_cell().array)
        results_old = self.atoms.calc.results.copy()

        # Propose a volume change and compute the new energy
        barostat_state_new, volume_old, volume_new, box_new, pos_new = (
            self._propose_step(self.barostat_state, box_old, pos_old)
        )

        # Calculate energy after the volume change
        self.atoms.set_cell(np.array(box_new))
        self.atoms.set_positions(np.array(pos_new))
        self._calculator.calculate(
            self.atoms, properties=["energy"], system_changes=all_changes
        )
        energy_new = self._calculator.results["energy"]

        # Accept or reject the volume change based on the Metropolis criterion
        kT = self.temperature_getter() * units.kB
        barostat_state_new, accepted = self._accept_step(
            barostat_state_new,
            energy_old,
            energy_new,
            volume_old,
            volume_new,
            kT,
            self.num_molecules,
        )

        # Tune the barostat state based on recent acceptance rate
        self.barostat_state = self._tune_barostat_step(barostat_state_new, volume_old)

        # Update self.atoms based on the acceptance
        if bool(accepted):
            self.atoms.calc.results["energy"] = energy_new
        else:
            # Revert the changes if the volume change is rejected
            self.atoms.set_cell(np.array(box_old))
            self.atoms.set_positions(np.array(pos_old))
            self.atoms.calc.results = results_old
