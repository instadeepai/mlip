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

import math

import jax
import jax.numpy as jnp
from jax import Array

from mlip.graph.batching_helpers import batch_graphs
from mlip.models.predictors.conservative_predictor import ConservativePredictor
from mlip.models_v1.mlip_network_v1 import MLIPNetworkV1
from mlip.models_v1.predictor_v1 import ForceFieldPredictorV1
from mlip.simulation.fep.alchemical_graph import AlchemicalGraph
from mlip.simulation.fep.alchemical_graph.masking import (
    get_alchemical_edge_mask,
    prune_edges_with_mask,
)
from mlip.simulation.fep.enums import FEPStage
from mlip.simulation.fep.models.repulsive_potentials import (
    SoftcoreLennardJonesPotential,
    SoftcoreRepulsivePotential,
)


class AlchemicalPredictor(ConservativePredictor):
    """Alchemical predictor for MLIPs.

    Can be used directly with an "Alchemical MLIP" - one that directly applies the
    alchemical lambda weighting to its update equations.

    Input `alchemical_lambda` is a vector of two lambda values:
        * lambda_1: The edge weight, 1.0 at state A to 0.0 at state B.
        * lambda_2: Controls the repulsion weight.

    The total potential is:
        U(lambda_1, lambda_2) = U_model(lambda_1) + U_R(lambda_2)

    Attributes:
        mlip_network: The MLIP network.
        required_properties: The properties that the predictor is required to compute.
        repulsive_potential: Softcore repulsion potential applied to alchemical edges.
            Defaults to SoftcoreLennardJonesPotential().
    """

    repulsive_potential: SoftcoreRepulsivePotential = SoftcoreLennardJonesPotential()

    @staticmethod
    def _get_repulsion_graph(graph: AlchemicalGraph) -> AlchemicalGraph:
        """Graph containing only edges that cross the alchemical boundary.

        Masks short-range edges, as the repulsive potential only applies to these.
        """
        sr_mask, _ = get_alchemical_edge_mask(graph)
        return prune_edges_with_mask(graph, sr_mask)

    def _compute_mixing_energy(
        self, positions: Array, strains: Array, graph: AlchemicalGraph
    ) -> tuple[Array, AlchemicalGraph]:
        """Compute the lambda_1-dependent energy term, excluding repulsion."""
        return super().compute_energy(positions, strains, graph)

    def compute_energy(
        self, positions: Array, strains: Array, graph_a: AlchemicalGraph
    ) -> tuple[Array, AlchemicalGraph]:
        """Compute the alchemical energy."""
        energy, graph_out = self._compute_mixing_energy(positions, strains, graph_a)

        graph_rep = self._get_repulsion_graph(graph_a)
        repulsion_energy, graph_rep = self._compute_repulsion_energy(
            positions, strains, graph_rep
        )

        total_energy = energy + repulsion_energy
        graph_out = graph_out.replace_globals(
            energy=graph_out.globals.energy + graph_rep.globals.energy
        )
        return total_energy, graph_out

    def _compute_repulsion_energy(
        self, positions: Array, strains: Array, graph_rep: AlchemicalGraph
    ) -> tuple[Array, AlchemicalGraph]:
        """Compute the energy of the repulsion potential.

        Args:
            positions: The positions of the atoms.
            strains: The strains of the lattice.
            graph_rep: The repulsion graph, containing only alchemical edges.

        Returns:
            Tuple of (total_repulsion_energy, updated graph_rep).
        """
        positions, cell = self._apply_strains(positions, strains, graph_rep)
        graph_rep = graph_rep.replace_nodes(positions=positions).replace_globals(
            cell=cell
        )

        per_subgraph_energy = self.repulsive_potential.compute_energy(graph_rep)
        repulsion_energy = jnp.sum(per_subgraph_energy)
        graph_rep = graph_rep.replace_globals(energy=per_subgraph_energy)
        return repulsion_energy, graph_rep

    def _compute_repulsion_potential_values(
        self, graph_a: AlchemicalGraph, repulsion_weights: Array
    ) -> Array:
        """Compute the repulsion potential energies at each reference repulsion weight.

        Args:
            graph_a: The graph for the current state.
            repulsion_weights: Repulsion weight values to evaluate the potential at.

        Returns:
            repulsion_weight_energies: Repulsion energies for all weights.
        """
        graph_rep = self._get_repulsion_graph(graph_a)
        return self.repulsive_potential.compute_multi_potential_values(
            graph_rep, repulsion_weights
        )

    def compute_alchemical_values(
        self,
        graph_a: AlchemicalGraph,
        reference_lambdas: Array,
        num_unique_edge_weights: int,
        batch_size: int | None,
    ) -> Array:
        """Compute the total energy at each reference lambda for the current state.

        The per-lambda energy is required for computing MBAR after an FEP run.

        Args:
            graph_a: The graph for the current state, with the edge weight consumed
                directly by `mlip_network`.
            reference_lambdas: Shape (n_reference, 2). Each row is (edge_weight,
                repulsion_weight) at which to evaluate the total alchemical energy.
            num_unique_edge_weights: Static upper bound on the number of distinct
                edge weights in `reference_lambdas`, to reduce redundant computation.
            batch_size: Number of unique edge weights to evaluate per vmapped batch, to
                reduce peak memory usage. If None, computes all in a single batch.

        Returns:
            per_lambda_energies: Shape (n_reference,)
        """
        positions = graph_a.nodes.positions
        strains = jnp.zeros_like(graph_a.globals.cell)

        def _mixing_energy_at(edge_weight: Array) -> Array:
            lam = jnp.asarray(graph_a.globals.alchemical_lambda)
            lam = lam.at[:, 0].set(edge_weight)
            graph = graph_a.replace_globals(alchemical_lambda=lam)
            energy, _ = self._compute_mixing_energy(positions, strains, graph)
            return energy

        edge_weights = reference_lambdas[:, 0]
        repulsion_weights = reference_lambdas[:, 1]

        size = (
            edge_weights.shape[0]
            if num_unique_edge_weights is None
            else num_unique_edge_weights
        )
        unique_edge_weights, inverse = jnp.unique(
            edge_weights, size=size, fill_value=1.0, return_inverse=True
        )

        if batch_size is None:
            unique_mixing_energies = jax.vmap(_mixing_energy_at)(unique_edge_weights)
        else:
            # Pad batches to same size, so only a single vmap is created internally.
            num_batches = math.ceil(size / batch_size)
            padding = jnp.full(num_batches * batch_size - size, unique_edge_weights[0])
            padded_edge_weights = jnp.concatenate([unique_edge_weights, padding])

            unique_mixing_energies = jax.lax.map(
                _mixing_energy_at, padded_edge_weights, batch_size=batch_size
            )[:size]

        mixing_energies = unique_mixing_energies[inverse]

        repulsion_energies = self._compute_repulsion_potential_values(
            graph_a, repulsion_weights
        )
        per_lambda_energies = mixing_energies + repulsion_energies
        return per_lambda_energies.squeeze()


class LinearAlchemicalPredictor(AlchemicalPredictor):
    """Alchemical predictor for FEP calculations, generic over any MLIP.

    Predicts an alchemical energy by duplicating the graph, predicting on endstates
    A and B independently, then linearly combining these energies. Use this with any
    `MLIPNetwork` that has not been adapted to consume the mixing weight directly.

    Input `alchemical_lambda` is a vector of two lambda values:
        * lambda_1: the edge weight, 1.0 at state A to 0.0 at state B
        * lambda_2: controls the repulsion weight

    The total potential is:
        U(lambda_1, lambda_2) = lambda_1 * U_A + (1 - lambda_1) * U_B + U_R(lambda_2)

    The `fep_stage` attribute is used to determine which of (U_A, U_B) are required.

    Attributes:
        fep_stage: The FEP stage for which this predictor is used.
    """

    fep_stage: FEPStage = FEPStage.A

    @staticmethod
    def _get_disjoint_graph(graph: AlchemicalGraph) -> AlchemicalGraph:
        """Graph containing only edges that do not cross the alchemical boundary."""
        sr_mask, lr_mask = get_alchemical_edge_mask(graph)
        lr_keep = ~lr_mask if lr_mask is not None else None
        return prune_edges_with_mask(graph, ~sr_mask, long_range_mask=lr_keep)

    def _compute_mixing_energy(
        self, positions: Array, strains: Array, graph_a: AlchemicalGraph
    ) -> tuple[Array, AlchemicalGraph]:
        """Compute lambda_1 * U_A + (1 - lambda_1) * U_B, without the repulsion term.

        Dispatches on `fep_stage` to avoid unnecessary computation.
        """
        if self.fep_stage == FEPStage.A:  # Compute only A potential
            return super()._compute_mixing_energy(positions, strains, graph_a)

        elif self.fep_stage == FEPStage.AR:  # Compute mixed A and B potentials
            graph_b = self._get_disjoint_graph(graph_a)
            return self._compute_mixed_energy(positions, strains, graph_a, graph_b)

        elif self.fep_stage == FEPStage.RB:  # Compute only B potential
            graph_b = self._get_disjoint_graph(graph_a)
            return super()._compute_mixing_energy(positions, strains, graph_b)

        else:
            raise ValueError(f"Invalid FEP stage: {self.fep_stage}")

    def _compute_mixed_energy(
        self,
        positions: Array,
        strains: Array,
        graph_a: AlchemicalGraph,
        graph_b: AlchemicalGraph,
    ) -> tuple[Array, AlchemicalGraph]:
        """Compute the energy given by the mixing of A and B potentials.

        Outputs a linear combination of the energies of state A and state B:
            edge_weight * U_A + (1 - edge_weight) * U_B

        Args:
            positions: The positions of the atoms from graph_a.
            strains: The strains of the lattice from graph_a.
            graph_a: The graph for state A, containing all edges.
            graph_b: The graph for state B, containing no alchemical edges.
        """

        edge_weight = graph_a.globals.alchemical_lambda[0, 0]

        def _mix_energies(e_a: Array, e_b: Array) -> Array:
            return edge_weight * e_a + (1 - edge_weight) * e_b

        # Batch A and B versions of the graph, and share positions and strains arrays
        batched_graph = batch_graphs([graph_a, graph_b], jittable=True)
        stacked_positions = jax.lax.concatenate([positions, positions], dimension=0)
        stacked_strains = jax.lax.concatenate([strains, strains], dimension=0)

        _, batched_graph = super()._compute_mixing_energy(
            stacked_positions, stacked_strains, batched_graph
        )
        # Extract per-graph energies and update graph_a as output graph
        graph_energies = batched_graph.globals.energy
        energy_a, energy_b = graph_energies[0], graph_energies[2]
        energy_mixed = _mix_energies(energy_a, energy_b)

        graph_a = graph_a.replace_globals(
            energy=jnp.stack([energy_mixed, jnp.zeros_like(energy_mixed)]),
        )
        graph_a = graph_a.update_global_features(energy_a=energy_a, energy_b=energy_b)
        return energy_mixed, graph_a

    def compute_alchemical_values(
        self,
        graph_a: AlchemicalGraph,
        reference_lambdas: Array,
        num_unique_edge_weights: int,
        batch_size: int | None,
    ) -> Array:
        """Compute the total energy at each reference lambda for the current state.

        The per-lambda energy is required for computing MBAR after an FEP run.

        Args:
            graph_a: The graph for state A.
            reference_lambdas: Shape (n_reference, 2). Each row is (edge_weight,
                repulsion_weight) at which to evaluate the total alchemical energy.
            num_unique_edge_weights: Unused. Required by parent `AlchemicalPredictor`.
            batch_size: Unused. Required by parent `AlchemicalPredictor`.

        Returns:
            per_lambda_energies: Shape (n_reference,)
        """
        positions = graph_a.nodes.positions
        strains = jnp.zeros_like(graph_a.globals.cell)
        graph_b = self._get_disjoint_graph(graph_a)

        _, graph = self._compute_mixed_energy(positions, strains, graph_a, graph_b)
        energy_a = graph.globals.features["energy_a"]
        energy_b = graph.globals.features["energy_b"]

        edge_weights = reference_lambdas[:, 0]
        repulsion_weights = reference_lambdas[:, 1]

        repulsion_energies = self._compute_repulsion_potential_values(
            graph_a, repulsion_weights
        )
        per_lambda_energies = (
            edge_weights * energy_a + (1 - edge_weights) * energy_b + repulsion_energies
        )
        return per_lambda_energies.squeeze()


class LinearAlchemicalPredictorV1(LinearAlchemicalPredictor, ForceFieldPredictorV1):
    """`LinearAlchemicalPredictor` compatible with legacy v1 MLIP networks."""

    mlip_network: MLIPNetworkV1
