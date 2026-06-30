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

import jax.numpy as jnp
from jax_md.dataclasses import dataclass as jax_compatible_dataclass

from mlip.graph.graph import Energy, PartialCharges
from mlip.typing import Prediction


@jax_compatible_dataclass
class AuxiliaryProperties:
    """Auxiliary properties that can be returned from a JAX-MD force function
    in addition to the forces.

    Attributes:
        energy: The energy prediction of the model.
        partial_charges: Optionally, the partial charges prediction of the model.
    """

    energy: Energy | list[Energy]
    partial_charges: PartialCharges | list[PartialCharges] | None = None


def create_auxiliary_properties(
    force_field_output: Prediction, split_indices: list[int], is_batched_sim: bool
) -> AuxiliaryProperties:
    """Creates auxiliary properties during a simulation from the force field output.

    Always adds energy. Optionally, if available in force field output, other
    properties are added.

    Args:
        force_field_output: The prediction of the force field.
        split_indices: The atom indices that define where to split the individual graphs
                       in case of batched simulations.
        is_batched_sim: Whether we are running a batched simulation currently.

    Returns:
        The created auxiliary properties object.
    """

    energies = force_field_output.energy[:-1]
    energies = list(energies) if is_batched_sim else energies[0]

    partial_charges = None
    if force_field_output.partial_charges is not None:
        partial_charges = jnp.delete(force_field_output.partial_charges, -1, axis=0)
        if is_batched_sim:
            partial_charges = jnp.split(partial_charges, split_indices, axis=0)

    return AuxiliaryProperties(
        energy=energies,
        partial_charges=partial_charges,
    )
