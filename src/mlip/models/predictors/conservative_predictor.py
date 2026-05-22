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

import jax
import jax.numpy as jnp
from jax import Array

from mlip.graph import Graph
from mlip.models.predictors.predictor import ForceFieldPredictor, Forces

logger = logging.getLogger("mlip")


class ConservativePredictor(ForceFieldPredictor):
    """Implementation of the conservative force field predictor.

    The conservative predictor computes forces and stress as first-order derivatives
    of the energy with respect to atomic positions and strains, respectively.
    """

    def __call__(self, graph: Graph) -> Graph:
        """Evaluates the predictor on a given graph.

        Computes the required properties, and updates the input graph with these
        quantities. If only energy is required, auto-differentiation is skipped
        to minimize computational cost.

        Args:
            graph: The input graph.

        Returns:
            An updated graph containing all predicted properties.
        """
        # Note: strains are invariant vector fields tangent to cell
        strains = jnp.zeros_like(graph.globals.cell)

        if self._only_energy_required():
            _, graph = self.compute_energy(graph.nodes.positions, strains, graph)
            return graph

        _, graph = self.compute_forces_and_stress(graph.nodes.positions, strains, graph)
        return graph

    def compute_forces_and_stress(
        self, positions: Array, strains: Array, graph: Graph
    ) -> tuple[Forces, Graph]:
        """Computes forces and stress of the input graph.

        This method takes in atomic positions, strains, and the graph, computes the
        gradients of the energy (forces), and, if required, also computes the stress
        tensor and pressure.

        Args:
            positions: The positions of the nodes.
            strains: The strains to apply.
            graph: The input graph.

        Returns:
            Computed forces and the updated graph containing the predicted properties.
        """
        (gradients, pseudo_stress), graph = jax.grad(
            self.compute_energy, (0, 1), has_aux=True
        )(positions, strains, graph)

        forces = -gradients

        graph = graph.replace_nodes(forces=forces)

        if self.required_properties.stress:
            stress = self._stress_from_pseudo_stress(graph, pseudo_stress)
            pressure = self.pressure_from_stress(stress)
            graph = graph.replace_globals(stress=stress, pressure=pressure)

        return forces, graph

    def _only_energy_required(self) -> bool:
        """Determines if only the energy is required for this prediction.

        Returns:
            Boolean indicating if only energy is required.
        """
        return self.required_properties.true_fields() == ["energy"]

    @staticmethod
    def _stress_from_pseudo_stress(graph: Graph, pseudo_stress: Array) -> Array:
        """Converts pseudo-stress to the stress tensor using the cell determinants.

        Uses the definition from the CHGNet paper and normalizes by the determinant
        of the cell matrix.

        Args:
            graph: The graph containing cell information and edge shifts.
            pseudo_stress: The pseudo-stress tensor computed from gradients.

        Returns:
            The computed stress tensor.
        """
        if graph.edges.shifts is None:
            logger.warning(
                "`stress` in `required_properties`, but graph does not contain "
                "`shifts` so a real `stress` cannot be computed. Returning dummy values"
                " of zeros for predicted stress."
            )
            return jnp.zeros_like(pseudo_stress)

        det = jnp.linalg.det(graph.globals.cell)[:, None, None]  # [n_graphs, 1, 1]
        det = jnp.where(det > 0.0, det, 1.0)  # Note: Dummy graphs have det = 0.

        # Stress as defined in the CHGNet paper and MPtrj dataset.
        # See [https://arxiv.org/pdf/2302.14231, eq. (6)]
        stress = 1 / det * pseudo_stress
        return stress
