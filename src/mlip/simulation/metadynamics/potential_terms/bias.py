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
from jax import Array

from mlip.graph import Graph
from mlip.simulation.metadynamics.potential_terms.collective_variables import (
    CollectiveVariable,
)


def _cv_difference(s: Array, centers: Array, periodic: bool) -> Array:
    """Compute the signed differences between a CV value and all hill centers.

    For non-periodic CVs returns `s - c` directly. For 2π-periodic CVs,
    wraps the difference to (-π, π) via `arctan2(sin(s - c), cos(s - c))`.
    """
    if periodic:
        return jnp.arctan2(jnp.sin(s - centers), jnp.cos(s - centers))
    return s - centers


class BiasPotential:
    """N-dimensional Gaussian-hill bias potential over one or more collective variables.

    Attributes:
        cvs: The collective variables defining the bias coordinates.
        sigmas: Gaussian hill width along each CV axis, same order as `cvs`.
    """

    def __init__(
        self, collective_variables: list[CollectiveVariable], sigmas: list[float]
    ):
        if len(collective_variables) != len(sigmas):
            raise ValueError("Must provide the same number of CVs and sigmas.")
        if len(collective_variables) == 0:
            raise ValueError("At least one collective variable must be provided.")
        self.cvs = collective_variables
        self.sigmas = jnp.asarray(sigmas)

    def __call__(self, graph: Graph) -> Array:
        """Return V_bias(s) = Σ_k h_k · exp(-Σ_i d(s_i,c_i_k)² / 2σ_i²)."""
        cv_values = self.compute_cvs(graph)

        gaussian_centers = graph.globals.features["gaussian_centers"]
        gaussian_heights = graph.globals.features["gaussian_heights"]
        num_gaussians = graph.globals.features["num_gaussians"]

        mask = jnp.arange(gaussian_centers.shape[0]) < num_gaussians
        deltas = [
            _cv_difference(cv_values[i], gaussian_centers[:, i], self.cvs[i].periodic)
            for i in range(len(self.cvs))
        ]
        deltas = jnp.stack(deltas, axis=-1)

        exponent = -0.5 * jnp.sum((deltas / self.sigmas) ** 2, axis=-1)
        gaussians = gaussian_heights * jnp.exp(exponent)
        return jnp.sum(jnp.where(mask, gaussians, 0.0))

    def compute_cvs(self, graph: Graph) -> Array:
        """Return the current CV values as a 1-D array of shape `(num_cvs,)`."""
        return jnp.array([cv(graph) for cv in self.cvs])
