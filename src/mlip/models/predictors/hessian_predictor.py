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

from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax import Array

from mlip.graph import Graph
from mlip.models.predictors import ConservativePredictor

Hessian: TypeAlias = Array


class HessianPredictor(ConservativePredictor):
    """Subclass of `ConservativePredictor` used to predict either
    the full energy Hessian matrix of a system, or a subsample
    of Hessian rows, depending on the `graph.sample_hessian_rows`
    attribute which is one of:

    * an array of indices of shape `(G, R)`, used to subsample the full
      Hessian matrix. An array of shape `(n, R, 3)` is then returned.

    * `array(True)`. In this case, the full Hessian matrix of shape
      `(N+1, 3, N+1, 3)` is returned.

    * `None`, in which case no Hessian is returned. This is useful to skip
      the additional AD pass, e.g. in mixed labels training.

    Where `N` is the number of total graph nodes including padding nodes,
    `n` number of real graph nodes, and `R` number of Hessian rows`.
    """

    def __call__(self, graph: Graph) -> Graph:
        """Evaluates the Hessian predictor on a given graph.

        Computes the required properties including the energy Hessian, and updates
        the input graph with these quantities. If Hessian is not required,
        falls back to evaluating the parent conservative predictor.

        Args:
            graph: The input graph.

        Returns:
            An updated graph containing all predicted properties.
        """
        if not self._hessian_required(graph):
            # then the parent class checks if only energy required
            return super().__call__(graph)

        get_hessian_terms = jax.jacrev(
            self.compute_sum_forces_subsample, 0, has_aux=True
        )
        hessian_terms, graph = get_hessian_terms(graph.nodes.positions, graph)
        if graph.globals.sample_hessian_rows.ndim != 0:
            # predicting a sample of the hessian
            hessian_terms = hessian_terms.transpose(1, 0, -1)

        # returned Hessian terms are of shape:
        # `(N+1, 3, N+1, 3)` in case of full Hessian
        # `(n, R, 3)` in case of subsampling.
        # `N = max_n_node`, `n = total graph nodes`,
        # `R = number of Hessian rows`.
        graph = graph.replace_nodes(hessian=hessian_terms)

        return graph

    def compute_sum_forces_subsample(
        self, positions: jnp.ndarray, graph: Graph
    ) -> tuple[Hessian, Graph]:
        """Return `(sum(F[sample_rows]), graph)` pair for downstream auto diff.
        The auxiliary `Graph` object can be forwarded by downstream methods, while
        the caller may differentiate through the subsampled force components
        `F[sample_rows]` to compute Hessian rows.
        """
        # Note: strains are invariant vector fields tangent to cell
        strains = jnp.zeros_like(graph.globals.cell)

        forces, graph = self.compute_forces_and_stress(positions, strains, graph)

        if graph.globals.sample_hessian_rows.ndim == 0:  # JITTABLE
            # return all force terms
            sum_gradients_subsample = -forces

        else:
            # sample and return sum of sampled force terms
            force_vector = forces.flatten()
            sample_hessian_rows = graph.globals.sample_hessian_rows

            # discard hessian rows corresponding to padding graphs
            mask = graph.graph_mask()
            mask_ndim = jnp.stack([
                mask for _ in range(sample_hessian_rows.shape[-1])
            ]).T

            sampled_force_terms = jnp.where(
                mask_ndim,
                force_vector[sample_hessian_rows],
                0,
            )
            sum_gradients_subsample = jnp.sum(-sampled_force_terms, axis=0)
        return sum_gradients_subsample, graph

    def _hessian_required(self, graph: Graph) -> bool:
        """Checks whether the Hessian is among the required properties
        and that the current batched graph has reference Hessians.
        """
        # check `hessian` is required
        cond_1 = "hessian" in self.required_properties.true_fields()
        # check `hessian` is required for the current batched graph
        cond_2 = (
            graph.globals.sample_hessian_rows is not None
            and graph.globals.is_dummy_for_init is None
        )
        return cond_1 and cond_2
