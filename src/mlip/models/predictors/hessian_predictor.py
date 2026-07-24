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

    * An array of indices of shape `(G, R)`, used to subsample the full
      Hessian matrix. A (sub-) Hessian of shape `(N, R, 3)` is then returned.

    * `None`, in this case, the full Hessian matrix of shape `(N, 3, N, 3)`
      is iteratively computed (row by row) and returned.

    * `Array(False)`, in which case no Hessian is returned. This is useful to skip
      the additional AD pass, e.g. in mixed labels training.

    Where `N` is the number of total graph nodes including padding nodes,
    and `R` number of Hessian rows`.
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
            # Skip Hessian prediction, parent class checks for other properties.
            return super().__call__(graph)

        if graph.globals.sample_hessian_rows is None:
            # Full Hessian computed iteratively.
            return self._iterative_hessian_prediction(graph)

        # Predicting a sample of the hessian.
        get_hessian_terms = jax.jacrev(
            self.compute_sum_forces_subsample, 0, has_aux=True
        )
        hessian_terms, graph = get_hessian_terms(graph.nodes.positions, graph)
        hessian_terms = hessian_terms.transpose(1, 0, -1)

        # Returned Hessian terms are of shape:
        # `(N, 3, N, 3)` in case of full Hessian computed iteratively.
        # `(N, R, 3)` in case of subsampling.
        # `N = number of graph nodes including padding nodes`,
        # `R = number of Hessian rows`.
        graph = graph.replace_nodes(hessian=hessian_terms)

        return graph

    def _iterative_hessian_prediction(self, graph: Graph) -> Graph:
        # Hessian inference is recommended to be done on single-graph batches
        # for efficiency. To reduce padding cost, N would otherwise have to be
        # set to the maximum system size.
        N = graph.nodes.positions.shape[0]

        forces_fn = functools.partial(self.compute_sum_forces_subsample, graph=graph)
        _, vjp_fun, graph = jax.vjp(forces_fn, graph.nodes.positions, has_aux=True)

        # Full basis for cotangent force vectors
        basis_matrices = jax.numpy.eye(N * 3).reshape(N * 3, N, 3)

        def single_pass_hessian(_, basis_matrices):
            def compute_hessian_row(basis):
                return vjp_fun(basis)[0]

            hessian_rows = compute_hessian_row(basis_matrices)
            return _, hessian_rows

        def iterative_hessian(basis_matrices):
            # Backpropgate 1 force cotangent at a time
            _, hessian_rows = jax.lax.scan(single_pass_hessian, None, basis_matrices)
            return jnp.reshape(hessian_rows, (-1, N, 3))

        hessian_terms = iterative_hessian(basis_matrices)

        hessian_terms = hessian_terms.reshape(N, 3, N, 3)[:-1, :, :-1, :]

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

        if graph.globals.sample_hessian_rows is None:
            # Return all force terms.
            sum_gradients_subsample = -forces

        else:
            # Sample and return sum of sampled force terms.
            force_vector = forces.flatten()
            sample_hessian_rows = graph.globals.sample_hessian_rows

            # Discard hessian rows corresponding to padding graphs.
            mask = graph.graph_mask()
            mask_ndim = jnp.stack([
                mask for _ in range(sample_hessian_rows.shape[-1])
            ]).T

            # Sampled force terms of shape `(G, R)` where
            # `G` is the number of graphs in the batch and
            # `R` number of sampled force terms per graph
            # (i.e. Hessian rows)
            sampled_force_terms = jnp.where(
                mask_ndim,
                force_vector[sample_hessian_rows],
                0,
            )

            # Sum over the batch dimension so that each force
            # term in the sum belongs to a distinct graph,
            # for their gradients not to overlap.
            sum_gradients_subsample = jnp.sum(-sampled_force_terms, axis=0)
        return sum_gradients_subsample, graph

    def _hessian_required(self, graph: Graph) -> bool:
        """Checks whether the Hessian is among the required properties
        and that the current batched graph has reference Hessians.
        """
        # Check `hessian` among the Predictor's required properties
        cond_1 = "hessian" in self.required_properties.true_fields()

        # Check `hessian` is required for the current batched graph.
        # By default, `hessian_rows = None` triggers Hessian inference (if requested).
        # Otherwise, `hessian_rows = array(False)` opts-out from Hessian prediction
        # for CombinedGraphDataset training.
        hessian_rows = graph.globals.sample_hessian_rows
        cond_2 = hessian_rows is None or hessian_rows.ndim > 0

        # Skip Hessian prediction during model initialization
        cond_3 = graph.globals.is_dummy_for_init is None

        return cond_1 and cond_2 and cond_3
