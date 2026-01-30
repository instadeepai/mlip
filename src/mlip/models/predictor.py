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

from dataclasses import replace
from typing import TypeAlias

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from mlip.data.helpers.edge_vectors import get_edge_relative_vectors
from mlip.typing import Prediction

RelativeEdgeVectors: TypeAlias = np.ndarray
AtomicSpecies: TypeAlias = np.ndarray
Senders: TypeAlias = np.ndarray
Receivers: TypeAlias = np.ndarray
NodeEnergies: TypeAlias = np.ndarray


class ForceFieldPredictor(nn.Module):
    """Flax module for a force field predictor.

    The apply function of this predictor returns the force field function used basically
    everywhere in the rest of the code base. This module is initialized from an
    already constructed MLIP model network module and a boolean whether to predict
    stress properties.

    Attributes:
        mlip_network: The MLIP network.
        predict_stress: Whether to predict stress properties. If false, only energies
                        and forces are computed.
    """

    mlip_network: nn.Module
    predict_stress: bool

    def __call__(self, graph: jraph.GraphsTuple, training: bool = False) -> Prediction:
        """Returns a `Prediction` dataclass of properties based on an input graph.

        Args:
            graph: The input graph.
            training: Whether the model is in training mode or not. If true, rngs should be
                      passed for stochastic modules.

        Returns:
            The properties as a `Prediction` object including "energy" and "forces".
            It may also include "stress" and "pressure" when `predict_stress=True`.
        """
        if graph.n_node.shape[0] == 1:
            raise ValueError(
                "Graph must be batched with at least one dummy graph. "
                "See models tutorial in documentation for details."
            )

        prediction, minus_forces, pseudo_stress = self._compute_gradients(graph, training)
        prediction = prediction.replace(forces=-minus_forces)

        if not self.predict_stress:
            return prediction

        stress_results = self._compute_stress_results(graph, pseudo_stress)
        return replace(
            prediction,
            stress=stress_results.stress,
            pressure=stress_results.pressure,
        )

    def _compute_gradients(
        self, graph: jraph.GraphsTuple, training: bool
    ) -> tuple[Prediction, np.ndarray, np.ndarray | None]:
        """Return a `(prediction, gradients, pseudo_stress)` triple.

        The `prediction` holds graph energies, and eventual optional fields.
        Dynamical forces (forces and stress) are populated later after
        the raw gradients have been processed.
        """
        # Note: strains are invariant vector fields tangent to cell
        strains = jnp.zeros_like(graph.globals.cell)

        # NOTE: When direct_force is enabled or predict_stress is disabled,
        # there is no need to compute corresponding gradients.
        direct_force = getattr(self.mlip_network.config, "direct_force", False)
        if direct_force:
            argnums = 1 if self.predict_stress else None
        else:
            argnums = (0, 1) if self.predict_stress else 0

        # Gradient is not needed
        if argnums is None:
            _, prediction = self._compute_energy(
                graph.nodes.positions, strains, graph, training
            )
            return prediction, -prediction.forces, None # pylint: disable=E1130

        # Differentiate wrt positions and strains (not cell)
        grads, prediction = jax.grad(
            self._compute_energy, argnums, has_aux=True
        )(graph.nodes.positions, strains, graph, training)

        if argnums == (0, 1):
            minus_forces, pseudo_stress = grads
        elif argnums == 0:
            minus_forces = grads
            pseudo_stress = None
        else:
            minus_forces = -prediction.forces
            pseudo_stress = grads

        return prediction, minus_forces, pseudo_stress

    @staticmethod
    def _compute_stress_results(
        graph: jraph.GraphsTuple,
        pseudo_stress: np.ndarray,
    ) -> Prediction:
        assert (
            graph.edges.shifts is not None
        ), "without shifts, the computed pseudo_stress is incorrect"

        det = jnp.linalg.det(graph.globals.cell)[:, None, None]  # [n_graphs, 1, 1]
        det = jnp.where(det > 0.0, det, 1.0)  # dummy graphs have det = 0

        # Stress as defined in the CHGNet paper and MPtrj dataset.
        # See [https://arxiv.org/pdf/2302.14231, eq. (6)]
        stress = 1 / det * pseudo_stress

        # Potential energy pressure contribution.
        potential_pressure = -1 / 3 * jnp.trace(stress, axis1=1, axis2=2)

        return Prediction(
            stress=stress,  # [n_graphs, 3, 3] stress tensor [eV / A^3]
            pressure=potential_pressure,  # [n_graphs,] pressure [eV / A^3]
        )

    def _compute_energy(
        self,
        positions: np.ndarray,
        strains: np.ndarray,
        graph: jraph.GraphsTuple,
        training: bool,
    ) -> tuple[np.ndarray, Prediction]:
        """Return total energy and a `Prediction` object holding graph energies.

        Total energy is expressed as a function of (positions, strains) for automatic
        differentiation. The `Prediction` object holds graph-wise energies at this
        stage, and may be further populated by downstream methods.
        """
        node_features = self._compute_node_features(positions, strains, graph, training)

        assert node_features.shape in [(len(positions),), (len(positions), 4)], (
            f"model output needs to be an array of shape "
            f"(n_nodes, ) or (n_nodes, 4), but got {node_features.shape}"
        )

        # When `node_energies` is a concatenation of the node-wise energies and forces.
        forces = None
        if getattr(self.mlip_network.config, "direct_force", False):
            node_energies, forces = jnp.split(node_features, [1], axis=-1)
            node_energies = node_energies.squeeze(-1)
        else:
            node_energies = node_features

        total_energy = jnp.sum(node_energies)

        graph_energies = e3nn.scatter_sum(
            node_energies, nel=graph.n_node
        )  # [ n_graphs,]

        prediction = Prediction(
            energy=graph_energies,
            forces=forces,
        )

        return total_energy, prediction

    def _compute_node_features(
        self,
        positions: np.ndarray,
        strains: np.ndarray,
        graph: jraph.GraphsTuple,
        training: bool,
    ) -> np.ndarray:
        """Evaluate node-wise outputs of `.mlip_network` on graph data.

        The graph is explicitly embedded from spatial features passed as
        first arguments for automatic differentiation.
        """
        if graph.edges.shifts is None:
            # Note: strains are not used in this case.
            assert graph.edges.displ_fun is not None
            vectors = graph.edges.displ_fun(
                positions[graph.receivers], positions[graph.senders]
            )
        else:
            # Note: `vectors` captures all dependencies wrt positions and strains.
            positions, cell = self._apply_strains(positions, strains, graph)
            vectors = get_edge_relative_vectors(
                positions=positions,
                senders=graph.senders,
                receivers=graph.receivers,
                shifts=graph.edges.shifts,
                cell=cell,
                n_edge=graph.n_edge,
            )

        # [n_nodes,] or [n_nodes, N] with additional features
        node_features = self.mlip_network(
            vectors,
            graph.nodes.species,
            graph.senders,
            graph.receivers,
            n_node=graph.n_node, # kwargs
            training=training, # kwargs
        )
        padding_mask = jraph.get_node_padding_mask(graph)
        padding_mask = jnp.expand_dims(
            padding_mask, range(padding_mask.ndim, node_features.ndim)
        )
        return node_features * padding_mask

    def _apply_strains(
        self, positions: np.ndarray, strains: np.ndarray, graph: jraph.GraphsTuple
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Virtually applies strains to the positions and cell for the energy derivative.

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
