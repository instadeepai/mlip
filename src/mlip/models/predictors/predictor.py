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

import abc
from typing import Callable, TypeAlias

import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from mlip.graph import Graph
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.predictors.energy_heads import standard_energy_computation_head
from mlip.typing.properties import Properties

Energy: TypeAlias = Array
Forces: TypeAlias = Array


class ForceFieldPredictor(nn.Module, abc.ABC):
    """Base class for all force field predictors.

    Attributes:
        mlip_network: The MLIP network to use as a backbone for property prediction.
        required_properties: The properties that the predictor is required to compute.
        energy_head: The energy head to use for aggregating MLIP outputs into energies,
            and/or other outputs. If None, we use a default standard energy head that
            sums node energies into a total energy per graph.
    """

    mlip_network: MLIPNetwork
    required_properties: Properties
    energy_head: Callable[[Graph], Array] | None = None

    @abc.abstractmethod
    def __call__(self, graph: Graph) -> Graph:
        """Compute required properties of the graph."""
        pass

    @property
    def _default_energy_head(self):
        """Return the default energy head to use if none is provided."""
        return standard_energy_computation_head

    @property
    def _energy_head(self):
        """Return the energy head to use for energy computation.

        This property should be used for all usages of the energy head.
        Uses `_default_energy_head` if no energy head was provided at initialization.
        """
        if self.energy_head is not None:
            return self.energy_head
        return self._default_energy_head

    def compute_energy(
        self, positions: Array, strains: Array, graph: Graph
    ) -> tuple[Energy, Graph]:
        """Compute the total energy of the input graph.

        Requires `positions` and `strains` as explicit arguments to enable first-order
        differentiation of the energy with respect to these quantities.

        Args:
            positions: The positions of the nodes.
            strains: The strains to apply to the cell for energy computation.
            graph: The input graph.

        Returns:
            The total energy, and an updated Graph containing all computed properties.
        """
        positions, cell = self._apply_strains(positions, strains, graph)
        graph = graph.replace_nodes(positions=positions)
        graph = graph.replace_globals(cell=cell)

        # Calculate node energies using mlip_network
        graph = self.mlip_network(graph)

        # Apply energy head to compute graph energies and total energy
        graph_energies = self._energy_head(graph)
        graph = graph.replace_globals(energy=graph_energies)
        total_energy = jnp.sum(graph_energies)

        # Extract any other required properties from the graph features
        if self.required_properties.partial_charges:
            partial_charges = graph.nodes.features["partial_charges"]
            graph = graph.replace_nodes(partial_charges=partial_charges)
            graph = graph.replace_globals(dipole_moment=graph.compute_dipole_moment())
            graph = graph.replace_globals(
                non_corrected_charge=graph.globals.features["non_corrected_charge"]
            )

        return total_energy, graph

    def _apply_strains(
        self, positions: Array, strains: Array, graph: Graph
    ) -> tuple[Array, Array]:
        """Applies `strains` to the `positions` and `cell` for the energy derivative.

        Cell is updated by:
            new_cell = cell @ (I + strains)
        Positions are updated based on the cell-coordinate relationship:
            new_positions = positions + positions @ strains

        Args:
            positions: The positions of the nodes.
            strains: The strains to apply.
            graph: The input graph.

        Returns:
            A tuple of the updated positions and cell.
        """
        # Stress should be symmetric up to numerical error, but we ensure this.
        # See [https://github.com/mir-group/nequip/blob/main/nequip/nn/grad_output.py].
        symm_strains = 0.5 * (strains + strains.transpose(0, 2, 1))

        strains_repeated = jnp.repeat(
            symm_strains,
            graph.n_node,
            axis=0,
            total_repeat_length=positions.shape[0],
        )
        positions += jnp.einsum("ni,nij->nj", positions, strains_repeated)

        cell = graph.globals.cell + graph.globals.cell @ symm_strains
        return positions, cell

    @abc.abstractmethod
    def compute_forces_and_stress(
        self, positions: Array, strains: Array, graph: Graph
    ) -> tuple[Forces, Graph]:
        """Compute forces and stress of the input graph.

        Force and stress prediction may differ between predictors. For example,
        some may compute forces as negative gradients of the energy, while others
        may use direct force prediction heads.

        We require `positions` and `strains` as explicit arguments to enable
        differentiation of the forces with respect to these quantities if desired.

        If the predictor is not required to compute either forces or stress, these
        can be set to zeros in the output.
        """
        pass

    @staticmethod
    def pressure_from_stress(stress: Array) -> Array:
        """Compute pressure from the (3, 3) stress tensor."""
        return -1 / 3 * jnp.trace(stress, axis1=1, axis2=2)
