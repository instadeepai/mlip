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
import logging
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from ase.units import eV, kJ, mol
from jax import Array

from mlip.simulation.fep.alchemical_graph import AlchemicalGraph
from mlip.utils.jax_utils import scatter_sum
from mlip.utils.safe_norm import safe_norm

logger = logging.getLogger("mlip")

# Max sigma values for each element from the GAFF XML
SCLJ_SIGMA = {
    1: 2.625,  # H (ha)
    3: 3.0,  # Li
    5: 3.0,  # B
    6: 3.479,  # C (c1/cg/ch)
    7: 3.650,  # N (n8)
    8: 3.243,  # O (oh)
    9: 3.034,  # F (f)
    11: 3.0,  # Na
    12: 3.0,  # Mg
    14: 3.5,  # Si
    15: 3.694,  # P (pX)
    16: 3.532,  # S (sX)
    17: 3.466,  # Cl (cl)
    19: 3.5,  # K
    20: 3.5,  # Ca
    35: 3.613,  # Br (br)
    53: 3.841,  # I (i)
}
# Epsilon values in kJ/mol (Original GAFF values)
SCLJ_EPSILON = {
    1: 0.0673,  # H (ha)
    3: 0.5000,  # Li
    5: 0.5000,  # B
    6: 0.6678,  # C (c1/cg/ch)
    7: 0.1351,  # N (n8)
    8: 0.3891,  # O (oh)
    9: 0.3481,  # F (f)
    11: 0.5000,  # Na
    12: 0.5000,  # Mg
    14: 0.5000,  # Si
    15: 0.9602,  # P (pX)
    16: 1.1816,  # S (sX)
    17: 1.1037,  # Cl (cl)
    19: 0.5000,  # K
    20: 0.5000,  # Ca
    35: 1.6451,  # Br (br)
    53: 2.0732,  # I (i)
}

KJ_PER_MOL_PER_ELECTRON_VOLT = eV / (kJ / mol)
SCLJ_ALPHA = 0.5


def _get_sigma_epsilon_tables(
    sigma_map: dict[int, float], epsilon_map: dict[int, float]
) -> tuple[Array, Array]:
    """Build atomic number - sigma/epsilon lookup tables.

    Args:
        sigma_map: Sigma value (Angstrom) to use for each atomic number.
        epsilon_map: Epsilon value (kJ/mol) to use for each atomic number.

    Returns:
        sigma_table: Sigma value indexed by atomic number.
        epsilon_table: Epsilon value (eV) indexed by atomic number.
    """
    max_atomic_number = max(sigma_map)
    sigma_table = jnp.zeros(max_atomic_number + 1)
    epsilon_table = jnp.zeros(max_atomic_number + 1)
    for atomic_number, sigma in sigma_map.items():
        sigma_table = sigma_table.at[atomic_number].set(sigma)
    for atomic_number, epsilon in epsilon_map.items():
        # Convert kJ/mol to eV:
        epsilon_table = epsilon_table.at[atomic_number].set(
            epsilon / KJ_PER_MOL_PER_ELECTRON_VOLT
        )
    return sigma_table, epsilon_table


def _get_edges_sigma_epsilon(
    graph_rep: AlchemicalGraph,
    sigma_map: dict[int, float],
    epsilon_map: dict[int, float],
) -> tuple[Array, Array]:
    """Get the pairwise LJ parameters (sigma, epsilon) for each edge in the graph.

    Uses Lorentz-Berthelot mixing rules for the sigma and epsilon values.

    Args:
        graph_rep: The graph to compute values for. Uses senders/receivers species.
        sigma_map: Sigma value (Angstrom) to use for each atomic number.
        epsilon_map: Epsilon value (kJ/mol) to use for each atomic number.

    Returns:
        The pairwise LJ parameters (sigma, epsilon) for each edge.
    """
    sigma_table, epsilon_table = _get_sigma_epsilon_tables(sigma_map, epsilon_map)
    atomic_numbers = graph_rep.nodes.atomic_numbers
    atom_sigma = sigma_table[atomic_numbers]
    atom_epsilon = epsilon_table[atomic_numbers]

    senders, receivers = graph_rep.senders, graph_rep.receivers
    edges_sigma = 0.5 * (atom_sigma[senders] + atom_sigma[receivers])
    edges_epsilon = jnp.sqrt(atom_epsilon[senders] * atom_epsilon[receivers])
    return edges_sigma, edges_epsilon


def _sclj_denominator(
    repulsion_weight: Array, edge_vectors: Array, edge_sigmas: Array
) -> Array:
    """Compute term used in the denominator of the softcore Lennard-Jones potential.

    denominator = alpha * (1 - repulsion_weight) + (r / sigma)^6
    """
    edge_distances = safe_norm(edge_vectors, axis=1)
    denominator = (
        SCLJ_ALPHA * (1 - repulsion_weight) + (edge_distances / edge_sigmas) ** 6
    )
    # jnp.clip blocks gradients through clipped values, preventing NaNs.
    return jnp.clip(denominator, min=1e-8)


@dataclass(frozen=True, eq=False)
class SoftcoreRepulsivePotential(abc.ABC):
    """Base class for softcore repulsion potentials on alchemical edges.

    Subclasses implement `compute_pairwise_energies`; this class provides the
    graph-level machinery (masking, per-subgraph scatter-sum, vmap over weights).

    Attributes:
        deterministic_scatter_ops: Whether to use deterministic scatter operations.
    """

    deterministic_scatter_ops: bool = False

    @abc.abstractmethod
    def compute_pairwise_energies(
        self,
        graph_rep: AlchemicalGraph,
        repulsion_weight: Array,
    ) -> Array:
        """Compute per-edge repulsion energies.

        Args:
            graph_rep: Repulsion graph (alchemical edges only).
            repulsion_weight: Scalar or per-edge repulsion weight.

        Returns:
            Per-edge energies in eV, shape (E,).
        """

    def _per_subgraph_energies(
        self, graph_rep: AlchemicalGraph, repulsion_weight: Array
    ) -> Array:
        """Call compute_pairwise_energies, mask padding edges, scatter-sum per subgraph.

        Args:
            graph_rep: Repulsion graph (alchemical edges only).
            repulsion_weight: Scalar broadcast to all edges, or a per-edge array.

        Returns:
            Per-subgraph energies, shape (M+1,) including the dummy sub-graph.
        """
        keep_mask = jnp.arange(len(graph_rep.senders)) < jnp.sum(graph_rep.n_edge[:-1])
        edge_energies = self.compute_pairwise_energies(graph_rep, repulsion_weight)
        edge_energies = jnp.where(keep_mask, edge_energies, 0.0)
        return scatter_sum(
            edge_energies,
            graph_rep.n_edge,
            deterministic=self.deterministic_scatter_ops,
        )

    def compute_energy(self, graph_rep: AlchemicalGraph) -> Array:
        """Compute repulsion energy for each sub-graph.

        Assumes that all real subgraphs are of the same system (containing the same
        number of atoms). This is difficult to check inside a jitted method, so for
        now we constrain to only be used with a single real subgraph; i.e. not
        suitable for batched simulations.

        Args:
            graph_rep: Repulsion graph (alchemical edges only) with strains already
                applied to positions and cell.

        Returns:
            Per-subgraph repulsion energies.
        """
        if graph_rep.n_node.shape[0] > 2:
            raise ValueError(
                "SoftcoreRepulsivePotential.compute_energy assumes a single "
                f"real subgraph; got n_node.shape={graph_rep.n_node.shape}."
            )

        # Map each system's repulsion weight to its edges.
        num_atoms_per_system = graph_rep.n_node[0]
        repulsion_weights = graph_rep.globals.alchemical_lambda[:, 1]
        rep_weights_padded = jnp.concatenate([repulsion_weights, jnp.zeros(1)])
        rep_weight_per_edge = rep_weights_padded[
            jnp.clip(
                graph_rep.senders // num_atoms_per_system, 0, len(repulsion_weights)
            )
        ]

        per_subgraph_energy = self._per_subgraph_energies(
            graph_rep, rep_weight_per_edge
        )
        return per_subgraph_energy

    def compute_multi_potential_values(
        self, graph_rep: AlchemicalGraph, repulsion_weights: Array
    ) -> Array:
        """Compute per-subgraph repulsion energies at each reference repulsion weight.

        Args:
            graph_rep: Repulsion graph (alchemical edges only).
            repulsion_weights: Repulsion weight values to evaluate the potential at.

        Returns:
            Per-system energies for all repulsion_weights (num_systems, num_weights).
        """

        def _energies_at_rep_weight(rep_weight: Array) -> Array:
            return self._per_subgraph_energies(graph_rep, rep_weight)[:-1]

        return jax.vmap(_energies_at_rep_weight)(repulsion_weights).T


@dataclass(frozen=True, eq=False)
class SoftcoreLennardJonesPotential(SoftcoreRepulsivePotential):
    r"""Softcore Lennard-Jones (SCLJ) repulsive potential.

    The pairwise SCLJ potential is:

    .. math::

        U_{\mathrm{SCLJ}}(r, \lambda) = 4 \epsilon \lambda
        \left(D(r, \lambda)^{-2} - D(r, \lambda)^{-1}\right)

    where :math:`\lambda` is the ``repulsion_weight`` and:

    .. math::

        D(r, \lambda) = 0.5 (1 - \lambda) + (r / \sigma)^6

    The strength of repulsion increases with :math:`\lambda` from 0 to 1,
    such that :math:`U_{\mathrm{SCLJ}}(r, 0) = 0` and
    :math:`U_{\mathrm{SCLJ}}(r, 1)` is the full Lennard-Jones potential.

    Attributes:
        deterministic_scatter_ops: Whether to use deterministic scatter operations.
            If True, uses a deterministic but slower alternative to scatter operations.
            Does not need to be set by the user; will be automatically set to True when
            needed inside a simulation.
        sigma_map: Dict specifying a sigma value (Angstrom) for each atomic number.
            The per-edge value is inferred using Lorentz-Berthelot mixing rules.
            Defaults to the maximum per-element value used by the GAFF-2.1 FF.
        epsilon_map: Dict specifying an epsilon value (kJ/mol) for each atomic number.
            The per-edge value is inferred using Lorentz-Berthelot mixing rules.
            Defaults to a representative per-element value used by the GAFF-2.1 FF.
    """

    sigma_map: dict[int, float] = field(default_factory=lambda: dict(SCLJ_SIGMA))
    epsilon_map: dict[int, float] = field(default_factory=lambda: dict(SCLJ_EPSILON))

    def compute_pairwise_energies(
        self, graph_rep: AlchemicalGraph, repulsion_weight: Array
    ) -> Array:
        vectors = graph_rep.edge_vectors()
        edges_sigma, edges_epsilon = _get_edges_sigma_epsilon(
            graph_rep, self.sigma_map, self.epsilon_map
        )

        multiplier = 4 * edges_epsilon * repulsion_weight
        denominator = _sclj_denominator(repulsion_weight, vectors, edges_sigma)
        return multiplier * ((1 / denominator) ** 2 - (1 / denominator))


@dataclass(frozen=True, eq=False)
class SoftcoreWCAPotential(SoftcoreRepulsivePotential):
    r"""Softcore Weeks-Chandler-Andersen (SCWCA) repulsive potential.

    Repulsion-only equivalent of the softcore Lennard-Jones (SCLJ) potential;
    shifted so that :math:`U = 0` at :math:`\sigma` and truncated to remove
    the attractive region (:math:`D(r, \lambda) > 2`).

    The pairwise SCLJ potential is:

    .. math::

        U_{\mathrm{SCLJ}}(r, \lambda) = 4 \epsilon \lambda
        \left(D(r, \lambda)^{-2} - D(r, \lambda)^{-1}\right)

    where :math:`\lambda` is the ``repulsion_weight`` and:

    .. math::

        D(r, \lambda) = 0.5 (1 - \lambda) + (r / \sigma)^6

    Then the SCWCA potential is:

    .. math::

        U_{\mathrm{SCWCA}}(r, \lambda) =
        \begin{cases}
            U_{\mathrm{SCLJ}}(r, \lambda) + \epsilon \lambda,
                & D(r, \lambda) \leq 2 \\
            0, & \text{otherwise}
        \end{cases}

    The strength of repulsion increases with :math:`\lambda` from 0 to 1,
    such that :math:`U_{\mathrm{SCWCA}}(r, 0) = 0` and
    :math:`U_{\mathrm{SCWCA}}(r, 1)` is the full WCA potential.

    Attributes:
        deterministic_scatter_ops: Whether to use deterministic scatter operations.
            If True, uses a deterministic but slower alternative to scatter operations.
            Does not need to be set by the user; will be automatically set to True when
            needed inside a simulation.
        sigma_map: Dict specifying a sigma value (Angstrom) for each atomic number.
            The per-edge value is inferred using Lorentz-Berthelot mixing rules.
            Defaults to the maximum per-element value used by the GAFF-2.1 FF.
        epsilon_map: Dict specifying an epsilon value (kJ/mol) for each atomic number.
            The per-edge value is inferred using Lorentz-Berthelot mixing rules.
            Defaults to a representative per-element value used by the GAFF-2.1 FF.
    """

    sigma_map: dict[int, float] = field(default_factory=lambda: dict(SCLJ_SIGMA))
    epsilon_map: dict[int, float] = field(default_factory=lambda: dict(SCLJ_EPSILON))

    def compute_pairwise_energies(
        self, graph_rep: AlchemicalGraph, repulsion_weight: Array
    ) -> Array:
        vectors = graph_rep.edge_vectors()
        edges_sigma, edges_epsilon = _get_edges_sigma_epsilon(
            graph_rep, self.sigma_map, self.epsilon_map
        )

        denominator = _sclj_denominator(repulsion_weight, vectors, edges_sigma)
        multiplier = 4 * edges_epsilon * repulsion_weight
        sclj = multiplier * ((1 / denominator) ** 2 - (1 / denominator))
        shifted = sclj + edges_epsilon * repulsion_weight
        return jnp.where(denominator > 2.0, 0.0, shifted)
