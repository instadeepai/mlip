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

import functools
import inspect

import jax.numpy as jnp
from jax import lax, random
from jax_md import dataclasses
from jax_md.dataclasses import dataclass as jax_compatible_dataclass

from mlip.models.force_field import ForceField
from mlip.models_v1.mlip_network_v1 import MLIPNetworkV1
from mlip.typing.properties import Properties
from mlip.utils.jax_utils import segment_sum

Array = jnp.ndarray
VolumeArray = Array
EnergyArray = Array

INITIAL_MAX_DELTA_VOLUME_FRACTION = 0.01
TUNE_FREQUENCY = 10
TUNING_FACTOR = 1.1
TUNE_UP_THRESHOLD = 0.75
TUNE_DOWN_THRESHOLD = 0.25
TUNE_MAX_DELTA_VOLUME_FRACTION = 0.3


@jax_compatible_dataclass
class MonteCarloBarostatState:
    """Shared state for the Monte Carlo Barostat."""

    target_pressure: float

    num_attempted: int
    num_accepted: int
    num_attempted_since_tune: int
    num_accepted_since_tune: int
    max_delta_volume: float
    rng: Array

    # Topology
    mol_counts: Array  # Number of atoms per molecule
    mol_indices: Array  # Molecule index for each atom


def _box_to_volume(box: Array, dim: int) -> VolumeArray:
    """Converts a box to a volume."""
    if box.ndim == 0:
        return box**dim
    elif box.ndim == 1:
        return jnp.prod(box)
    elif box.ndim == 2:
        return jnp.abs(jnp.linalg.det(box))
    else:
        raise ValueError(f"Box must be 0D, 1D or 2D, got {box.ndim}D.")


def _scale_molecule_centroids(
    positions: Array, molecule_indices: Array, mol_counts: Array, scale_factor: float
) -> Array:
    """Scales the center of mass of molecules, preserving internal geometry.

    IMPORTANT: Assumes that coordinates are raw Cartesian coordinates,
    not fractional or wrapped coordinates.

    Args:
        positions: The current positions of the atoms in raw cartesian coordinates.
        molecule_indices: The indices of the molecules each atom belongs to.
        mol_counts: The number of atoms in each molecule.
        scale_factor: The scale factor to apply to the molecule centroids.

    Returns:
        The new positions of the atoms in raw cartesian coordinates.
    """
    num_molecules = mol_counts.shape[0]

    # Compute the centroids of each molecule (num_molecules, 3)
    molecule_positions_summed = segment_sum(
        positions,
        molecule_indices,
        num_segments=num_molecules,
        deterministic=True,
    )
    counts_reshaped = mol_counts[:, None]
    molecule_centroids = molecule_positions_summed / counts_reshaped
    new_molecule_centroids = molecule_centroids * scale_factor

    # Broacast back to (num_atoms, 3), the molecule centroids for each atom
    atom_centroids_old = molecule_centroids[molecule_indices]
    atom_centroids_new = new_molecule_centroids[molecule_indices]
    return positions - atom_centroids_old + atom_centroids_new


def create_high_precision_force_field(force_field: ForceField) -> ForceField:
    """Create a copy of a ForceField for high-precision energy calculations.

    The MonteCarloBarostat step requires high-precision and deterministic energy
    outputs. This is achieved by zeroing atomic energies (to avoid large constant
    offsets that hurt numerical precision) and enabling deterministic scatter ops.
    The created force field outputs energy only, as no other properties are required.

    Args:
        force_field: The original ForceField instance to create a copy from.

    Returns:
        A new ForceField instance with deterministic scatter ops enabled, without
        formation energies added, and with a deterministic energy head.
    """
    if isinstance(force_field.predictor.mlip_network, MLIPNetworkV1):
        new_config = force_field.config.model_copy(
            update={"atomic_energies": "zero", "deterministic_scatter_ops": True}
        )
    else:
        new_config = force_field.config.model_copy(
            update={"add_atomic_energies": False, "deterministic_scatter_ops": True}
        )

    model_class = type(force_field.predictor.mlip_network)
    new_mlip_network = model_class(
        config=new_config,
        dataset_info=force_field.dataset_info,
    )

    energy_head = force_field.predictor._energy_head
    if "deterministic" not in inspect.signature(energy_head).parameters:
        raise ValueError(
            f"Energy head '{energy_head}' does not support the 'deterministic' kwarg. "
            "This is required for setting up a high-precision force field for the "
            "MonteCarloBarostat. Please add this as a kwarg to the energy head."
        )
    deterministic_energy_head = functools.partial(energy_head, deterministic=True)

    predictor_class = type(force_field.predictor)
    new_predictor = predictor_class(
        mlip_network=new_mlip_network,
        required_properties=Properties(energy=True, forces=False),
        energy_head=deterministic_energy_head,
    )

    return ForceField(predictor=new_predictor, params=force_field.params)


def sanitize_molecule_indices(molecule_indices: Array) -> Array:
    """Sanitize the molecule indices to be a contiguous array of integers.

    Args:
        molecule_indices: The molecule indices to sanitize.

    Returns:
        The sanitized molecule indices.
    """
    _, molecule_indices_sanitized = jnp.unique(
        jnp.array(molecule_indices), return_inverse=True
    )
    return molecule_indices_sanitized


def propose_volume_change(
    barostat_state: MonteCarloBarostatState,
    box: Array,
    positions: Array,
) -> tuple[MonteCarloBarostatState, VolumeArray, VolumeArray, Array, Array]:
    """Proposes a volume change and scales system geometry.

    Args:
        barostat_state: The current barostat state.
        box: The current box.
        positions: The current positions.

    Returns:
        (updated barostat state, current volume, new volume, new box, new positions)
    """
    rng, step_rng = random.split(barostat_state.rng)

    dim = positions.shape[1]
    volume_old = _box_to_volume(box, dim)

    # Propose delta volume
    delta_volume = (random.uniform(step_rng) * 2 - 1) * barostat_state.max_delta_volume
    volume_new = volume_old + delta_volume

    # Calculate length scale (prevent negative volume issues)
    length_scale = (jnp.maximum(volume_new, 1e-6) / volume_old) ** (1.0 / dim)

    box_new = box * length_scale
    positions_new = _scale_molecule_centroids(
        positions, barostat_state.mol_indices, barostat_state.mol_counts, length_scale
    )

    barostat_state_new = dataclasses.replace(barostat_state, rng=rng)

    return barostat_state_new, volume_old, volume_new, box_new, positions_new


def accept_volume_change(
    barostat_state: MonteCarloBarostatState,
    energy_old: EnergyArray,
    energy_new: EnergyArray,
    volume_old: VolumeArray,
    volume_new: VolumeArray,
    kT: float,
    num_molecules: int,
) -> tuple[MonteCarloBarostatState, bool]:
    """Accepts or rejects the volume change based on the Metropolis criterion.

    Metropolis criterion:
        acceptance_probability = exp(-Delta_W / kB * T)
        where Delta_W = Delta_E + P * Delta_V - N_mol * kB * T * ln(V_new / V_old)

    Always accepts if Delta_W <= 0.
    Always rejects if volume_new <= 0.

    See the OpenMM documentation for more details:
    docs.openmm.org/latest/userguide/theory/02_standard_forces.html#montecarlobarostat

    Args:
        barostat_state: The current barostat state.
        energy_old: The old energy.
        energy_new: The new energy.
        volume_old: The old volume.
        volume_new: The new volume.
        kT: The temperature in energy units (kB * T).
        num_molecules: The number of molecules.

    Returns:
        The updated barostat state, and the accepted flag.
    """
    delta_e = energy_new - energy_old
    p_term = barostat_state.target_pressure * (volume_new - volume_old)
    entropy_term = num_molecules * kT * jnp.log(volume_new / volume_old)
    delta_w = delta_e + p_term - entropy_term
    acceptance_probability = jnp.exp(-delta_w / kT)

    # Accept if (Delta_W <= 0 OR rand < prob) AND volume_new > 0
    rng_new, accept_rng = random.split(barostat_state.rng)
    sampled_prob = random.uniform(accept_rng)
    condition = (delta_w <= 0) | (sampled_prob < acceptance_probability)
    accepted = condition & (volume_new > 0)

    new_accepted_count = barostat_state.num_accepted + jnp.where(accepted, 1, 0)
    new_attempted_count = barostat_state.num_attempted + 1
    new_accepted_tune = barostat_state.num_accepted_since_tune + jnp.where(
        accepted, 1, 0
    )
    new_attempted_tune = barostat_state.num_attempted_since_tune + 1

    barostat_state_new = dataclasses.replace(
        barostat_state,
        num_accepted=new_accepted_count,
        num_attempted=new_attempted_count,
        num_accepted_since_tune=new_accepted_tune,
        num_attempted_since_tune=new_attempted_tune,
        rng=rng_new,
    )

    return barostat_state_new, accepted


def tune_barostat(
    barostat_state: MonteCarloBarostatState,
    volume_old: VolumeArray,
) -> MonteCarloBarostatState:
    """Tunes the barostat state based on the acceptance rate.

    If not enough barostat steps have been taken since last tune, the barostat state
    is returned unchanged. Otherwise, the acceptance rate is computed since the last
    tune. If the acceptance rate is too low, the maximum volume change is decreased.
    If the acceptance rate is too high, the maximum volume change is increased.

    Args:
        barostat_state: The current barostat state.
        volume_old: The old volume.

    Returns:
        The updated barostat state.
    """

    def tune_maximum_volume_change(bs):
        """Tunes the maximum volume change based on the acceptance rate."""
        rate = bs.num_accepted_since_tune / bs.num_attempted_since_tune

        new_max_v = bs.max_delta_volume
        # Tune the maximum volume change based on the acceptance rate
        new_max_v = lax.cond(
            rate < TUNE_DOWN_THRESHOLD,
            lambda v: v / TUNING_FACTOR,
            lambda v: lax.cond(
                rate > TUNE_UP_THRESHOLD, lambda _: v * TUNING_FACTOR, lambda _: v, None
            ),
            new_max_v,
        )
        # Cap at fraction of total volume
        new_max_v = jnp.minimum(new_max_v, volume_old * TUNE_MAX_DELTA_VOLUME_FRACTION)

        return dataclasses.replace(
            bs,
            max_delta_volume=new_max_v,
            num_accepted_since_tune=0,
            num_attempted_since_tune=0,
        )

    # Only tune if there have been enough barostat update steps since last tune
    barostat_state_new = lax.cond(
        barostat_state.num_attempted_since_tune >= TUNE_FREQUENCY,
        tune_maximum_volume_change,
        lambda x: x,
        barostat_state,
    )

    return barostat_state_new
