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

from typing import Any

import flax.linen as nn

from mlip.data.dataset_info import DatasetInfo
from mlip.graph import Graph
from mlip.models.config import MLIPNetworkConfig
from mlip.models.inference_context import InferenceContext
from mlip.typing import ModelParameters
from mlip.typing.properties import Properties


class MLIPNetwork(nn.Module):
    """Base class for GNN node-wise energy models.

    Energy models deriving from this class return node-wise
    contributions to the total energy, from the edge vectors of a graph,
    the atomic species of the nodes, and the edges themselves passed
    as `senders` and `receivers` indices.

    Our MLIP models are validated with Pydantic, and hold a reference to
    their `.Config` class describing the set of hyperparameters.
    """

    Config = MLIPNetworkConfig  # Must be overridden by the child classes

    config: MLIPNetworkConfig
    dataset_info: DatasetInfo

    @property
    def available_properties(self) -> Properties:
        """Default available properties."""
        return Properties(
            stress=True,
            hessian=True,
        )

    @property
    def is_moe_model(self) -> bool:
        """Whether this model uses mixture-of-experts parameters."""
        return False

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        """Calculate node-wise energy contributions and update the input Graph.

        Args:
            graph: Input Graph object containing edge vectors, node species, senders,
                   receivers, and node mask.

        Returns:
            The updated Graph with per-node energy contributions stored in the
            "energy" node feature.
        """
        raise NotImplementedError(
            "The base 'MLIPNetwork' model does not implement a call function."
        )

    @nn.nowrap
    def prepare_experts_for_inference(
        self,
        params: ModelParameters,
        inference_context: InferenceContext,
    ) -> tuple["MLIPNetwork", ModelParameters]:
        """No-op base; MoE subclasses override to contract expert kernels."""
        return self, params

    def __init_subclass__(cls, **kwargs: Any):
        """This enforces that child classes will
        need to override the `Config` attribute.
        """
        super().__init_subclass__(**kwargs)
        if cls.Config is MLIPNetworkConfig and cls.__name__ != "MLIPNetworkV1":
            raise NotImplementedError(
                f"{cls.__name__} must override the `Config` attribute."
            )
