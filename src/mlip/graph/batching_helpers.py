# Copyright 2020 DeepMind Technologies Limited.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
"""Implementations in this file adapted from `jraph.utils`."""

import dataclasses
from collections.abc import Callable

import jax
import numpy as np

from mlip.graph.graph import Graph


def batch_graphs(graphs: list[Graph]) -> Graph:
    """Returns batched graph given a list of graphs.

    Adapted from `jraph.utils.batch_np`.

    Args:
        graphs: The list of graphs to batch.

    Returns:
        The batched graph.
    """
    # Calculates offsets for sender and receiver arrays, caused by concatenating
    # the nodes arrays.
    offsets = np.cumsum(np.array([0] + [np.sum(g.n_node) for g in graphs[:-1]]))

    def _map_concat(nests):
        def _concat(*args):
            return np.concatenate(args)

        return jax.tree.map(_concat, *nests)

    # When batching graphs, we need to ensure that either all graphs have long range
    # interactions or none of them do. If some graphs have long range interactions and
    # others don't, we need to raise an error.
    def _has_long_range_interactions(graphs: list[Graph]) -> bool:
        """Check if all graphs have long range interactions or none of them do."""

        def graph_has_long_range(g: Graph) -> bool:
            return (
                g.n_edge_long_range is not None
                and g.senders_long_range is not None
                and g.receivers_long_range is not None
                and g.edges_long_range is not None
            )

        if all(graph_has_long_range(g) for g in graphs):
            return True
        if all(not graph_has_long_range(g) for g in graphs):
            return False
        raise ValueError(
            "When batching graphs, all graphs must have long range interactions or none"
            " of them."
        )

    if _has_long_range_interactions(graphs):
        concat_n_edge_long_range = np.concatenate([g.n_edge_long_range for g in graphs])
        concat_senders_long_range = np.concatenate([
            g.senders_long_range + o for g, o in zip(graphs, offsets)
        ])
        concat_receivers_long_range = np.concatenate([
            g.receivers_long_range + o for g, o in zip(graphs, offsets)
        ])
        concat_edges_long_range = _map_concat([g.edges_long_range for g in graphs])
    else:
        concat_n_edge_long_range = None
        concat_senders_long_range = None
        concat_receivers_long_range = None
        concat_edges_long_range = None

    return Graph(
        n_node=np.concatenate([g.n_node for g in graphs]),
        n_edge=np.concatenate([g.n_edge for g in graphs]),
        nodes=_map_concat([g.nodes for g in graphs]),
        edges=_map_concat([g.edges for g in graphs]),
        globals=_map_concat([g.globals for g in graphs]),
        senders=np.concatenate([g.senders + o for g, o in zip(graphs, offsets)]),
        receivers=np.concatenate([g.receivers + o for g, o in zip(graphs, offsets)]),
        senders_long_range=concat_senders_long_range,
        receivers_long_range=concat_receivers_long_range,
        n_edge_long_range=concat_n_edge_long_range,
        edges_long_range=concat_edges_long_range,
    )


def pad_with_graphs(
    graph: Graph,
    n_node: int,
    n_edge: int,
    n_graph: int,
    n_edge_long_range: int | None = None,
) -> Graph:
    """Pads a graph to size by adding computation preserving graphs.

    Adapted from `jraph.utils.pad_with_graphs`.

    The graph is padded by first adding a padding graph which contains the
    padding nodes and edges, and then empty graphs without nodes or edges.

    The empty graphs and the padding graph do not interfere with the MLIP
    calculations on the original graph, and so are computation preserving.

    The padding graph requires at least one node and one graph.

    This function does not support `jax.jit`, because the shape of the output
    is data-dependent.

    Args:
      graph: `Graph` object to be padded with padding graph and empty graphs.
      n_node: The number of nodes in the padded graph.
      n_edge: The number of edges in the padded graph.
      n_graph: The number of graphs in the padded graph. Two is the lowest possible
               value, because we always have at least one graph in the original
               graph, and we need one padding graph for the padding.
      n_edge_long_range: The number of long range edges in the padded graph.
                If None, long range interactions are not padded.
    Raises:
      ValueError: If the passed `n_graph` is smaller than 2.
      RuntimeError: If the given graph is too large for the given padding.

    Returns:
      The padded graph.
    """
    if n_graph < 2:
        raise ValueError(
            f"n_graph is {n_graph}, which is smaller than minimum value of 2."
        )

    graph = jax.device_get(graph)

    pad_n_node = int(n_node - np.sum(graph.n_node))
    pad_n_edge = int(n_edge - np.sum(graph.n_edge))
    pad_n_graph = int(n_graph - graph.num_graphs)
    if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
        raise RuntimeError(
            "Given graph is too large for the given padding. Difference: "
            f"n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}."
        )
    if n_edge_long_range is not None:
        pad_n_edge_long_range = int(n_edge_long_range - np.sum(graph.n_edge_long_range))

        if pad_n_edge_long_range < 0:
            raise RuntimeError(
                "Given graph is too large for the given padding. Difference: "
                f"n_edge_long_range {pad_n_edge_long_range}."
            )

    pad_n_empty_graph = pad_n_graph - 1

    def tree_nodes_pad(leaf):
        return np.zeros((pad_n_node,) + leaf.shape[1:], dtype=leaf.dtype)

    def tree_edges_pad(leaf):
        return np.zeros((pad_n_edge,) + leaf.shape[1:], dtype=leaf.dtype)

    def tree_globs_pad(leaf):
        return np.zeros((pad_n_graph,) + leaf.shape[1:], dtype=leaf.dtype)

    if n_edge_long_range is not None:
        n_edge_long_range_padded = np.concatenate([
            np.array([pad_n_edge_long_range], dtype=np.int32),
            np.zeros(pad_n_empty_graph, dtype=np.int32),
        ])
        senders_long_range_padded = np.zeros(pad_n_edge_long_range, dtype=np.int32)
        receivers_long_range_padded = np.zeros(pad_n_edge_long_range, dtype=np.int32)

        def tree_long_range_edges_pad(leaf):
            return np.zeros((pad_n_edge_long_range,) + leaf.shape[1:], dtype=leaf.dtype)

        edges_long_range_padded = jax.tree.map(
            tree_long_range_edges_pad, graph.edges_long_range
        )
    else:
        n_edge_long_range_padded = None
        senders_long_range_padded = None
        receivers_long_range_padded = None
        edges_long_range_padded = None

    padding_graph = Graph(
        n_node=np.concatenate([
            np.array([pad_n_node], dtype=np.int32),
            np.zeros(pad_n_empty_graph, dtype=np.int32),
        ]),
        n_edge=np.concatenate([
            np.array([pad_n_edge], dtype=np.int32),
            np.zeros(pad_n_empty_graph, dtype=np.int32),
        ]),
        nodes=jax.tree.map(tree_nodes_pad, graph.nodes),
        edges=jax.tree.map(tree_edges_pad, graph.edges),
        globals=jax.tree.map(tree_globs_pad, graph.globals),
        senders=np.zeros(pad_n_edge, dtype=np.int32),
        receivers=np.zeros(pad_n_edge, dtype=np.int32),
        n_edge_long_range=n_edge_long_range_padded,
        senders_long_range=senders_long_range_padded,
        receivers_long_range=receivers_long_range_padded,
        edges_long_range=edges_long_range_padded,
    )
    return batch_graphs([graph, padding_graph])


# NaN-fill factories for optional Graph fields that are `Prediction` targets
# (see `mlip.typing.prediction.Prediction`). Using NaN as a sentinel lets
# downstream loss and eval-metric code mask out samples whose dataset did not
# provide the field. Only `Prediction`-targeted fields are listed here; other
# optional fields are either always present or handled elsewhere.
_GLOBAL_PAD_FACTORIES: dict[str, Callable[[Graph], np.ndarray]] = {
    "energy": lambda g: np.full_like(g.globals.weight, np.nan),
    "stress": lambda g: np.full((*g.globals.weight.shape, 3, 3), np.nan),
    "pressure": lambda g: np.full_like(g.globals.weight, np.nan),
    "charge": lambda g: np.full_like(g.globals.weight, np.nan),
    "dipole_moment": lambda g: np.full((*g.globals.weight.shape, 3), np.nan),
}

_NODE_PAD_FACTORIES: dict[str, Callable[[Graph], np.ndarray]] = {
    "forces": lambda g: np.full_like(g.nodes.positions, np.nan),
    "partial_charges": lambda g: np.full(g.nodes.positions.shape[:1], np.nan),
}


def homogenize_graph_fields(graphs: list[Graph]) -> list[Graph]:
    """Fill missing `Prediction`-targeted fields with NaN so graphs from
    heterogeneous datasets share the same pytree structure.

    NaN is used as a sentinel so loss functions can detect and mask out samples
    whose dataset did not provide the field. Only fields that live in
    `Prediction` are homogenized here.

    Args:
        graphs: List of graphs that may have heterogeneous optional fields.

    Returns:
        A new list of graphs with uniform pytree structure.
    """
    if not graphs:
        return graphs

    present_globals: set[str] = set()
    present_nodes: set[str] = set()
    for g in graphs:
        for f in _GLOBAL_PAD_FACTORIES:
            if getattr(g.globals, f) is not None:
                present_globals.add(f)
        for f in _NODE_PAD_FACTORIES:
            if getattr(g.nodes, f) is not None:
                present_nodes.add(f)

    def _fill(graph: Graph) -> Graph:
        global_updates = {
            f: _GLOBAL_PAD_FACTORIES[f](graph)
            for f in present_globals
            if getattr(graph.globals, f) is None
        }
        node_updates = {
            f: _NODE_PAD_FACTORIES[f](graph)
            for f in present_nodes
            if getattr(graph.nodes, f) is None
        }
        if global_updates:
            graph = graph.replace_globals(**global_updates)
        if node_updates:
            graph = graph.replace_nodes(**node_updates)
        return graph

    return [_fill(g) for g in graphs]


def _field_signature(graph: Graph) -> frozenset[str]:
    """Per-graph signature of non-None optional fields across all three graph
    components. Includes `features.*` dict keys so e.g. two graphs with
    `nodes.features={"x": ...}` vs `nodes.features={}` are flagged as
    incompatible."""
    present: set[str] = set()
    for comp_name in ("nodes", "edges", "globals"):
        comp = getattr(graph, comp_name)
        for f in dataclasses.fields(comp):
            val = getattr(comp, f.name)
            if f.name == "features":
                present.update(f"{comp_name}.features.{k}" for k in (val or {}))
            elif val is not None:
                present.add(f"{comp_name}.{f.name}")
    return frozenset(present)


def validate_batch_compatible(graphs: list[Graph]) -> None:
    """Raise if graphs have heterogeneous optional-field presence.

    This is the condition that `homogenize_graph_fields` would fix. Surfacing
    it here with a clear message avoids the cryptic `jax.tree.map` structure
    mismatch that `batch_graphs()` would otherwise produce later.
    """
    if len(graphs) < 2:
        return
    ref = _field_signature(graphs[0])
    for g in graphs[1:]:
        this = _field_signature(g)
        if this != ref:
            diff = sorted(ref.symmetric_difference(this))
            raise ValueError(
                "Cannot batch graphs with heterogeneous optional fields: "
                f"{diff} are present on some graphs but not others. Pass "
                "`homogenize=True` to GraphDataset so missing fields are "
                "NaN-padded before batching, or fix the upstream data "
                "pipeline so all graphs share the same optional fields."
            )
