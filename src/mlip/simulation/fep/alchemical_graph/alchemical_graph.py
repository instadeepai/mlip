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

import dataclasses
from typing import Self

from flax import struct
from jax import Array

from mlip.graph.graph import Graph, GraphGlobals


@struct.dataclass
class AlchemicalGraphGlobals(GraphGlobals):
    """Graph globals for an alchemical graph."""

    alchemical_lambda: Array | None = None
    alchemical_atom_indices: Array | None = None

    @classmethod
    def from_graph_globals(cls, graph_globals: GraphGlobals) -> Self:
        """Creates alchemical graph globals, copying over all base `GraphGlobals`
        fields. The alchemical-specific fields are left at their defaults and are
        expected to be set separately via `.replace()`.
        """
        base_fields = {
            field.name: getattr(graph_globals, field.name)
            for field in dataclasses.fields(graph_globals)
        }
        return cls(**base_fields)


@struct.dataclass
class AlchemicalGraph(Graph):
    globals: AlchemicalGraphGlobals

    @classmethod
    def from_graph(cls, graph: Graph) -> Self:
        """Creates an `AlchemicalGraph` from a `Graph`, copying over all base
        `Graph` fields and replacing `globals` with `AlchemicalGraphGlobals`.
        """
        base_fields = {
            field.name: getattr(graph, field.name)
            for field in dataclasses.fields(graph)
        }
        base_fields["globals"] = AlchemicalGraphGlobals.from_graph_globals(
            graph.globals
        )
        return cls(**base_fields)
