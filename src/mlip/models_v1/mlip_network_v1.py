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

import jax
import jax.numpy as jnp
import pydantic

from mlip.data.dataset_info import DatasetInfo
from mlip.graph import Graph
from mlip.models.mlip_network import MLIPNetwork
from mlip.typing.properties import Properties


class MLIPNetworkV1(MLIPNetwork):
    """Base class for GNN node-wise energy models.

    Energy models deriving from this class return node-wise
    contributions to the total energy, from the edge vectors of a graph,
    the atomic species of the nodes, and the edges themselves passed
    as `senders` and `receivers` indices.

    Our MLIP models are validated with Pydantic, and hold a reference to
    their `.Config` class describing the set of hyperparameters.
    """

    Config = pydantic.BaseModel  # Must be overridden by the child classes

    config: pydantic.BaseModel
    dataset_info: DatasetInfo

    available_properties: Properties = Properties(stress=True)

    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> jax.Array:
        """Compute node-wise energy summands. This function must be overridden by the
        implementation of `MLIPNetwork`.
        """
        raise NotImplementedError(
            "No energy model defined by MLIPNetwork.__call__, "
            "but must be overridden by its child classes."
        )

    def calculate(self, graph: Graph) -> Graph:
        """
        Calculate node-wise energy contributions and update the input Graph.

        Args:
            graph: Input Graph object containing edge vectors, node species, senders,
                receivers, and node mask.

        Returns:
            The updated Graph with per-node energy contributions stored in the
            "energy" node feature.
        """
        with jax.ensure_compile_time_eval():
            allowed_zs = jnp.array(sorted(self.dataset_info.allowed_atomic_numbers))
            # Set the non-available values to 666. May cause NaNs for OOB nodes on GPU.
            lookup_table = jnp.full(max(allowed_zs) + 1, 666, dtype=jnp.int32)
            lookup_table = lookup_table.at[allowed_zs].set(jnp.arange(allowed_zs.size))

        species = lookup_table[graph.nodes.atomic_numbers]
        # Replace dummy node species with 0.
        species = jnp.where(graph.node_mask(), species, 0)

        node_features = self(
            graph.edge_vectors(),
            species,
            graph.senders,
            graph.receivers,
        )
        graph = graph.replace_nodes(
            features={"energy": node_features * graph.node_mask()},
        )
        return graph

    def __init_subclass__(cls, **kwargs: Any):
        """This enforces that child classes will
        need to override the `Config` attribute.
        """
        super().__init_subclass__(**kwargs)
        if cls.Config is pydantic.BaseModel:
            raise NotImplementedError(
                f"{cls.__name__} must override the `Config` attribute."
            )
