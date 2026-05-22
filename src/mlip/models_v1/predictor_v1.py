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

from typing import Callable, TypeAlias

import jax.numpy as jnp
from jax import Array

from mlip.graph import Graph
from mlip.models.predictors import ConservativePredictor
from mlip.models_v1.mlip_network_v1 import MLIPNetworkV1
from mlip.typing.properties import Properties

Energy: TypeAlias = Array
Forces: TypeAlias = Array

Senders: TypeAlias = Array
Receivers: TypeAlias = Array
NodeEnergies: TypeAlias = Array


class ForceFieldPredictorV1(ConservativePredictor):
    """Base class for all force field predictors.

    Attributes:
        mlip_network: The MLIP network to use as a backbone for property prediction.
        required_properties: The properties that the predictor is required to compute.
        energy_head: The energy head to use for aggregating MLIP outputs into energies,
            and/or other outputs. If None, we use a default standard energy head that
            sums node energies into a total energy per graph.
    """

    mlip_network: MLIPNetworkV1
    required_properties: Properties
    energy_head: Callable[[Graph], Array] | None = None

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

        # Calculate node energies using mlip_network_v1
        # NOTE: do not change this for V1 compatibility
        graph = self.mlip_network.calculate(graph)

        # Apply energy head to compute graph energies and total energy
        graph_energies = self._energy_head(graph)
        graph = graph.replace_globals(energy=graph_energies)
        total_energy = jnp.sum(graph_energies)

        return total_energy, graph
