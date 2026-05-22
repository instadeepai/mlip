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

import jax
import jax.numpy as jnp

from mlip.graph import Graph
from mlip.utils.safe_norm import safe_divide, safe_norm

# We want energies in eV and distances in Angstrom.
# Thus we compute epsilon_0 in e^2.eV^-1.Angstrom^-1
EPSILON_0 = 5.5263494e-3
# This cutoff value is used to smooth the long range interaction envelope function
# around 0 Angstrom. It is not linked to the long range cutoff distance.
# The action of this parameter on the long range interactions is that beyond half
# of this value in Angstrom, the Coulomb interaction is not affected by the close
# to zero smoothing.
PHYSNET_ENVELOPE_CUTOFF = 10.0


def correct_partial_charge_feature(graph: Graph) -> Graph:
    """Correct the partial charge feature of the graph.

    Partial charges stored in node.features are corrected to have a sum matching that
    of the total charge of the graph.

    Args:
        graph: The graph to correct the partial charge feature of.

    Returns:
        The corrected graph.
    """
    if graph.nodes.features.get("partial_charges") is None:
        raise ValueError("Partial charges are not present in the graph node features.")

    ref_charge = graph.globals.charge
    charge = graph.aggregate_per_graph(graph.nodes.features["partial_charges"])
    correction = safe_divide(charge - ref_charge, graph.n_node)
    expanded_correction = jnp.repeat(
        correction,
        graph.n_node,
        total_repeat_length=graph.nodes.positions.shape[0],
    )
    corrected_partial_charges = (
        graph.nodes.features["partial_charges"] - expanded_correction
    )
    graph = graph.update_node_features(partial_charges=corrected_partial_charges)
    return graph


def compute_long_range_interactions(graph: Graph) -> jax.Array:
    """Computes the electrostatic interactions between pairs of atoms in eV.

    Calculates the Coulomb interaction energy between pairs of atoms using:

    Q_ij = k_e * q_i * q_j / r_ij

    where q_i, q_j are the partial charges on atoms i and j, and r_ij is their
    relative distance.

    To avoid the singularity at r_ij << 1, we use the smooth cutoff
    function `physnet_envelope_function()`.

    Args:
        graph: The graph to compute the long range interactions of.

    Returns:
        Array of long range interactions, one for each graph in the batch.
    """
    partial_charges = graph.nodes.features["partial_charges"]
    senders_long_range = graph.senders_long_range
    receivers_long_range = graph.receivers_long_range
    distances = safe_norm(
        graph.long_range_edge_vectors(),
        axis=-1,
    )  # [n_edge_long_range, ]

    k_e = 1 / (4 * jnp.pi * EPSILON_0)

    charge_interactions = (
        partial_charges[senders_long_range] * partial_charges[receivers_long_range]
    )

    return k_e * charge_interactions * physnet_envelope_function(distances)


def physnet_envelope_function(r: jax.Array) -> jax.Array:
    """Computes the long range interaction envelope function.

    This function is extracted from the PhysNet paper and prevents instability when
    computing the Coulomb interaction energy for radius r_ij << 1.

    Args:
        r: Array of distances between atom pairs

    Returns:
        Array of smoothed distance values that avoid singularities at small r.
    """

    def phi(x: float) -> float:
        output = 1 - 6 * x**5 + 15 * x**4 - 10 * x**3
        return output * (x < 1)

    phi_values = phi(2 * r / PHYSNET_ENVELOPE_CUTOFF)
    short_smooth_cutoff = phi_values / (jnp.sqrt(jnp.square(r) + 1))
    inverse_padded_distance = jnp.where(jnp.isclose(r, 0), jnp.zeros_like(r), 1 / r)
    long_smooth_cutoff = (1 - phi_values) * inverse_padded_distance

    return short_smooth_cutoff + long_smooth_cutoff
