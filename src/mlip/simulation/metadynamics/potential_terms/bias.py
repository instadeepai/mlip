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
    """Base class for a metadynamics Gaussian-hill bias potential."""

    def __call__(self, graph: Graph) -> Array:
        """Return the total bias potential energy (eV) for the given graph."""
        raise NotImplementedError

    def compute_cvs(self, graph: Graph) -> Array:
        """Return the current CV values as a 1-D array of shape `(num_cvs,)`."""
        raise NotImplementedError


class BiasPotential1D(BiasPotential):
    """One-dimensional Gaussian-hill bias for a single collective variable.

    Attributes:
        cv: The collective variable defining the bias coordinate.
        sigma: Gaussian hill width along the CV axis.
    """

    def __init__(self, collective_variable: CollectiveVariable, sigma: float):
        self.cv = collective_variable
        self.sigma = sigma

    def __call__(self, graph: Graph) -> Array:
        """Return V_bias(s) = Σ_k h_k · exp(-d(s1,c1_k)² / 2σ²)"""
        cv_value = self.cv(graph)

        gaussian_centers = graph.globals.features["gaussian_centers"][:, 0]
        gaussian_heights = graph.globals.features["gaussian_heights"]
        num_gaussians = graph.globals.features["num_gaussians"]

        mask = jnp.arange(gaussian_centers.shape[0]) < num_gaussians
        delta = _cv_difference(cv_value, gaussian_centers, self.cv.periodic)
        gaussians = gaussian_heights * jnp.exp(-(delta**2) / (2 * self.sigma**2))
        return jnp.sum(jnp.where(mask, gaussians, 0.0))

    def compute_cvs(self, graph: Graph) -> Array:
        return jnp.array([self.cv(graph)])


class BiasPotential2D(BiasPotential):
    """Two-dimensional Gaussian-hill bias for a pair of collective variables.

    Attributes:
        cv_1: The first collective variable.
        cv_2: The second collective variable.
        sigma_1: Gaussian hill width along the first CV axis.
        sigma_2: Gaussian hill width along the second CV axis.
    """

    def __init__(
        self,
        collective_variable_1: CollectiveVariable,
        collective_variable_2: CollectiveVariable,
        sigma_1: float,
        sigma_2: float,
    ):
        self.cv_1 = collective_variable_1
        self.cv_2 = collective_variable_2
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

    def __call__(self, graph: Graph) -> Array:
        """Return V_bias(s1,s2) = Σ_k h_k exp(-d(s1,c1_k)²/2σ1² - d(s2,c2_k)²/2σ2²)."""
        s1 = self.cv_1(graph)
        s2 = self.cv_2(graph)

        gaussian_centers = graph.globals.features["gaussian_centers"]
        centers_1 = gaussian_centers[:, 0]
        centers_2 = gaussian_centers[:, 1]
        heights = graph.globals.features["gaussian_heights"]
        num_gaussians = graph.globals.features["num_gaussians"]

        mask = jnp.arange(centers_1.shape[0]) < num_gaussians
        delta_1 = _cv_difference(s1, centers_1, self.cv_1.periodic)
        delta_2 = _cv_difference(s2, centers_2, self.cv_2.periodic)
        gaussians = heights * jnp.exp(
            -(delta_1**2) / (2 * self.sigma_1**2) - (delta_2**2) / (2 * self.sigma_2**2)
        )
        return jnp.sum(jnp.where(mask, gaussians, 0.0))

    def compute_cvs(self, graph: Graph) -> Array:
        return jnp.array([self.cv_1(graph), self.cv_2(graph)])
