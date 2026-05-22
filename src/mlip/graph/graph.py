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

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Self, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import IrrepsArray
from flax import struct
from jax.typing import ArrayLike

from mlip.graph.mask_helpers import (
    get_graph_padding_mask,
    get_node_padding_mask,
)
from mlip.graph.neighborhood import get_neighborhood
from mlip.typing.prediction import Prediction
from mlip.utils.safe_norm import safe_divide

if TYPE_CHECKING:
    from mlip.data.chemical_system import ChemicalSystem

Positions: TypeAlias = ArrayLike  # [num_nodes, 3]
DisplacementVectors: TypeAlias = ArrayLike  # [num_edges, 3]
ShiftVectors: TypeAlias = ArrayLike  # [num_edges, 3]
AtomicSpecies: TypeAlias = ArrayLike  # [num_nodes]
AtomicNumbers: TypeAlias = ArrayLike  # [num_nodes]
Forces: TypeAlias = ArrayLike  # [num_nodes, 3]
Cell: TypeAlias = ArrayLike  # [num_graphs, 3, 3]
Stress: TypeAlias = ArrayLike  # [num_graphs, 3, 3]
Energy: TypeAlias = ArrayLike  # [num_graphs]
Pressure: TypeAlias = ArrayLike  # [num_graphs]
WeightingFactors: TypeAlias = ArrayLike  # [num_graphs]
PartialCharges: TypeAlias = ArrayLike  # [num_nodes]
Charge: TypeAlias = ArrayLike  # [num_graphs]
DipoleMoment: TypeAlias = ArrayLike  # [num_graphs, 3]
DatasetIdx: TypeAlias = ArrayLike  # [num_graphs]
SpinMultiplicity: TypeAlias = ArrayLike  # [num_graphs]
HessianRows: TypeAlias = ArrayLike  # [num_graphs, num_rows] | Array(True)


@struct.dataclass
class GraphNodes:
    """Features of the `Graph` object related to nodes.

    Attributes:
        atomic_numbers: The atomic numbers of the nodes.
        positions: The positions of the nodes.
        forces: The forces on the nodes.
        partial_charges: The partial charges of the nodes.
        hessian: The Hessian matrix (second derivatives of energy w.r.t. node positions.
        features: Any additional node features stored inside a dictionary / PyTree.
    """

    atomic_numbers: AtomicNumbers | None = None
    positions: Positions | None = None
    forces: Forces | None = None

    partial_charges: PartialCharges | None = None
    hessian: ArrayLike | None = None

    features: dict[str, ArrayLike | IrrepsArray] = struct.field(default_factory=dict)


@struct.dataclass
class GraphEdges:
    """Features of the `Graph` object related to edges.

    Attributes:
        shifts: Shift vectors to compute edge vectors from positions taking into account
                PBCs. Either this needs to be specified or the `displ_fun` attribute
                instead, but not both. If both exist, shifts will be used.
        displ_fun: Alternative to `shifts` to compute edge vectors from positions
                   taking into account PBCs. The displacement function should be
                   vmapped already, meaning it can take in a position matrix for
                   senders and receivers and output the edge vector matrix. Moreover,
                   the displacement function must be wrapped in
                   `jax.tree_util.Partial` in order to be compatible with jitting.
                   Note that if the displacement function pathway is applied,
                   stress cannot be calculated as a property. In the future,
                   these two pathways may be unified.
        features: Any additional edge features stored inside a dictionary / PyTree.
    """

    shifts: ShiftVectors | None = None
    displ_fun: Callable[[Positions, Positions], DisplacementVectors] | None = None

    features: dict[str, ArrayLike | IrrepsArray] = struct.field(default_factory=dict)


@struct.dataclass
class GraphGlobals:
    """Global features of the `Graph` object.

    Attributes:
        cell: The cell definition. Important for PBCs.
        weight: The weight of the graph, which can be used inside a loss function.
        energy: The energy.
        stress: The stress.
        pressure: The pressure.
        charge: The total charge of the graph.
        non_corrected_charge: The total charge of the graph before correction. Required
                              for the total charge term of the loss during training.
        spin_multiplicity: The spin multiplicity of the graph.
        sample_hessian_rows: Indices of force terms to sample for sampled Hessian
                     prediction.
        dataset_idx: An index pointing to which dataset this graph belongs to.
        is_dummy_for_init: Whether this graph is a dummy graph just used for model
            initialization. By default, this is set to `None` which means false (but
            false is not used to allow shape-based evaluation of this field). Will be
            set to `np.array(True)` for the dummy initialization graph.
        features: Any additional global features stored inside a dictionary / PyTree.
    """

    cell: Cell
    weight: WeightingFactors
    energy: Energy | None = None
    stress: Stress | None = None
    pressure: Pressure | None = None
    charge: Charge | None = None
    dipole_moment: DipoleMoment | None = None
    non_corrected_charge: Charge | None = None
    spin_multiplicity: SpinMultiplicity | None = None
    sample_hessian_rows: HessianRows | None = None
    is_dummy_for_init: ArrayLike | None = None
    dataset_idx: DatasetIdx | None = None

    features: dict[str, ArrayLike | IrrepsArray] = struct.field(default_factory=dict)


@struct.dataclass
class Graph:
    """The `Graph` class defining a single graph or a batch of graphs.

    Modeled after `jraph.GraphsTuple`, but with additional methods.

    Attributes:
        nodes: The node features of the graph.
        edges: The edge features of the graph.
        globals: The global features of the graph.
        n_node: The number of nodes in the graph (or a vector if a batch).
        n_edge: The number of edges in the graph (or a vector if a batch).
        senders: The sender indices of the edges of the graph.
        receivers: The receiver indices of the edges of the graph.
        n_edge_long_range: The number of long range edges in the graph.
        senders_long_range: The sender indices of the long range edges.
        receivers_long_range: The receiver indices of the long range edges.
        edges_long_range: Edge information for long-range edges if present.
    """

    # Graph properties
    nodes: GraphNodes
    edges: GraphEdges
    globals: GraphGlobals

    # Graph definition
    n_node: ArrayLike
    n_edge: ArrayLike
    senders: ArrayLike
    receivers: ArrayLike
    n_edge_long_range: ArrayLike | None = None
    senders_long_range: ArrayLike | None = None
    receivers_long_range: ArrayLike | None = None
    edges_long_range: GraphEdges | None = None

    @classmethod
    def from_chemical_system(
        cls,
        chemical_system: ChemicalSystem,
        graph_cutoff_angstrom: float,
        long_range_cutoff_angstrom: float | None = None,
    ) -> Self:
        """Create a `Graph` object from a chemical system and dataset info.

        This includes computing the senders/receivers/shifts for the system and
        otherwise just transferring data 1-to-1 to the graph.

        Args:
            chemical_system: The chemical system object.
            graph_cutoff_angstrom: The graph distance cutoff in Angstrom.
            long_range_cutoff_angstrom: The long range distance cutoff in Angstrom.
                If None, long range interactions are not computed.

        Returns:
            The `Graph` object for the given chemical system.
        """
        senders, receivers, shift_vectors = get_neighborhood(
            positions=chemical_system.positions,
            cutoff=graph_cutoff_angstrom,
            pbc=chemical_system.pbc,
            cell=chemical_system.cell,
        )

        if long_range_cutoff_angstrom is not None:
            senders_long_range, receivers_long_range, shifts_long_range = (
                get_neighborhood(
                    positions=chemical_system.positions,
                    cutoff=long_range_cutoff_angstrom,
                    pbc=chemical_system.pbc,
                    cell=chemical_system.cell,
                )
            )
            n_edge_long_range = np.array([senders_long_range.shape[0]])
            edges_long_range = GraphEdges(shifts=shifts_long_range, displ_fun=None)
        else:
            senders_long_range = None
            receivers_long_range = None
            n_edge_long_range = None
            edges_long_range = None

        cell = (
            np.zeros((3, 3)) if chemical_system.cell is None else chemical_system.cell
        )
        energy = np.array(
            0.0 if chemical_system.energy is None else chemical_system.energy
        )

        partial_charges = chemical_system.partial_charges
        charge = chemical_system.charge
        dipole_moment = chemical_system.dipole_moment
        if partial_charges is not None:
            if charge is None:
                charge = round(np.sum(partial_charges))
            if dipole_moment is None:
                centered_positions = chemical_system.positions - np.mean(
                    chemical_system.positions, axis=0
                )
                dipole_moment = np.einsum(
                    "i, ij -> j", partial_charges, centered_positions
                )

        graph = cls(
            nodes=GraphNodes(
                positions=chemical_system.positions,
                forces=chemical_system.forces,
                hessian=chemical_system.hessian,
                atomic_numbers=chemical_system.atomic_numbers,
                partial_charges=partial_charges,
            ),
            edges=GraphEdges(shifts=shift_vectors, displ_fun=None),
            globals=jax.tree.map(
                lambda x: x[None, ...],
                GraphGlobals(
                    cell=cell,
                    energy=energy,
                    stress=chemical_system.stress,
                    weight=np.asarray(chemical_system.weight),
                    charge=np.asarray(charge, dtype=np.int32)
                    if charge is not None
                    else None,
                    spin_multiplicity=np.asarray(
                        chemical_system.spin_multiplicity, dtype=np.int32
                    )
                    if chemical_system.spin_multiplicity is not None
                    else None,
                    dipole_moment=np.asarray(dipole_moment)
                    if dipole_moment is not None
                    else None,
                ),
            ),
            n_node=np.array([chemical_system.positions.shape[0]]),
            n_edge=np.array([senders.shape[0]]),
            receivers=receivers,
            senders=senders,
            n_edge_long_range=n_edge_long_range,
            senders_long_range=senders_long_range,
            receivers_long_range=receivers_long_range,
            edges_long_range=edges_long_range,
        )

        return graph

    @property
    def num_graphs(self) -> int:
        """Number of graphs in the (possibly batched, possibly padded) graph."""
        return self.n_node.shape[0]

    def compute_dipole_moment(self) -> DipoleMoment:
        """Compute the dipole moments of structures in the graph.

        The dipole moment is computed as the sum of the partial charges multiplied by
        the centered positions.

        Returns:
            The dipole moment of the graph.
        """
        barycenters = self.aggregate_per_graph(self.nodes.positions)  # [n_graphs, 3]
        barycenters = safe_divide(barycenters, self.n_node[:, None])  # [n_graphs, 3]
        expanded_barycenters = jnp.repeat(
            barycenters,
            self.n_node,
            total_repeat_length=self.nodes.positions.shape[0],
            axis=0,
        )  # [n_nodes, 3]
        centered_positions = self.nodes.positions - expanded_barycenters
        dipole_moment = self.aggregate_per_graph(
            self.nodes.partial_charges[:, None] * centered_positions
        )
        return dipole_moment

    # replace methods
    def replace_nodes(self, **kwargs) -> Self:
        """Returns the `Graph` object where `nodes` attribute are replaced.

        Keyword arguments are forwarded to the `.replace()` call on the nodes
        dataclass.
        """
        return self.replace(nodes=self.nodes.replace(**kwargs))

    def replace_edges(self, **kwargs) -> Self:
        """Returns the `Graph` object where `edges` attribute are replaced.

        Keyword arguments are forwarded to the `.replace()` call on the edges
        dataclass.
        """
        return self.replace(edges=self.edges.replace(**kwargs))

    def replace_globals(self, **kwargs) -> Self:
        """Returns the `Graph` object where `globals` attribute are replaced.

        Keyword arguments are forwarded to the `.replace()` call on the globals
        dataclass.
        """
        return self.replace(globals=self.globals.replace(**kwargs))

    def update_node_features(self, **kwargs) -> Self:
        """Returns the `Graph` object where `nodes` attribute are replaced.

        Keyword arguments are forwarded to the `.replace()` call on the nodes
        dataclass.
        """
        return self.replace_nodes(features=self.nodes.features | kwargs)

    def update_edge_features(self, **kwargs) -> Self:
        """Returns the `Graph` object where `edges` attribute are replaced.

        Keyword arguments are forwarded to the `.replace()` call on the nodes
        dataclass.
        """
        return self.replace_edges(features=self.edges.features | kwargs)

    def update_global_features(self, **kwargs) -> Self:
        """Returns the `Graph` object where `globals` attribute are replaced.

        Keyword arguments are forwarded to the `.replace()` call on the globals
        dataclass.
        """
        return self.replace_globals(features=self.globals.features | kwargs)

    def node_mask(self) -> jax.Array:
        """Evaluates the node padding mask array for the graph.

        `True` refers to a real node, while `False` refers to a dummy node in
        the (batched) graph.

        Returns:
            The node padding mask.
        """
        return get_node_padding_mask(self)

    def graph_mask(self) -> jax.Array:
        """Evaluates the graph padding mask array for the batched graph.

        `True` refers to a real graph, while `False` refers to a dummy graph in
        the batched graph.

        Returns:
            The graph padding mask.
        """
        return get_graph_padding_mask(self)

    def to_prediction(self) -> Prediction:
        """Creates a `Prediction` object from the current graph, which contains all the
        properties of the graph that are also part of a prediction.

        Returns:
            The prediction.
        """
        return Prediction(
            energy=self.globals.energy,
            forces=self.nodes.forces,
            stress=self.globals.stress,
            hessian=self.nodes.hessian,
            pressure=self.globals.pressure,
            partial_charges=self.nodes.partial_charges,
        )

    def request_full_hessian(self) -> Self:
        """Returns a graph that has `sample_hessian_rows=np.array(True)` in the globals.

        Required for inference pipelines.
        """
        return self.replace_globals(sample_hessian_rows=np.array(True))

    def _compute_vectors(
        self,
        senders: ArrayLike,
        receivers: ArrayLike,
        edges: GraphEdges,
        n_edge: ArrayLike,
        use_np: bool = False,
    ) -> ArrayLike:
        """Compute relative edge vectors from senders to receivers, applying PBC
        shifts or `displ_fun` if provided."""
        vectors_senders = self.nodes.positions[senders]  # [n_edges, 3]
        vectors_receivers = self.nodes.positions[receivers]  # [n_edges, 3]

        if edges.displ_fun is not None:
            assert edges.shifts is None
            vectors = edges.displ_fun(
                vectors_senders, vectors_receivers, self.globals.cell
            )
            return vectors

        if self.globals.cell is not None:
            assert edges.shifts is not None
            if use_np:
                np_ = np
                kwargs_num_edges = {}
            else:
                np_ = jnp
                kwargs_num_edges = {"total_repeat_length": senders.shape[0]}

            shifts = np_.einsum(
                "ei,eij->ej",
                edges.shifts,
                np_.repeat(
                    self.globals.cell,
                    n_edge,
                    axis=0,
                    **kwargs_num_edges,
                ),
            )
            vectors_senders -= shifts  # Minus sign to match results with ASE

        vectors = vectors_receivers - vectors_senders
        return vectors

    def edge_vectors(self, use_np: bool = False) -> ArrayLike:
        """Compute the relative edge vectors from senders to receivers.

        We use `displ_fun` if available, otherwise edge vectors are computed directly
        using the `positions` and `shifts`. In the case of PBCs, sender nodes are
        translated from the unit cell to the receiver's nearest neighbouring cell:

        .. code-block:: python

            # If `displ_fun` is None, this method returns:
            vectors = positions[receivers] - (positions[senders] - shifts @ cell)

            # Equivalent to this line from the ASE docs:
            D = positions[j] - positions[i] + S.dot(cell)

        Args:
            use_np: Whether to use `numpy` or `jax.numpy` for the computation. Default
                    is `False`, which means `jax.numpy` is used.

        Returns:
            The relative edge vectors, labelled `D` by ASE.
        """
        return self._compute_vectors(
            self.senders,
            self.receivers,
            self.edges,
            self.n_edge,
            use_np,
        )

    def long_range_edge_vectors(self, use_np: bool = False) -> ArrayLike:
        """Compute the relative long-range edge vectors from senders to receivers.

        Mirrors `self.edge_vectors()` for the long range graph.

        Args:
            use_np: Whether to use `numpy` or `jax.numpy` for the
                    computation. Default is `False`, which means
                    `jax.numpy` is used.

        Returns:
            The relative long-range edge vectors.
        """
        return self._compute_vectors(
            self.senders_long_range,
            self.receivers_long_range,
            self.edges_long_range,
            self.n_edge_long_range,
            use_np,
        )

    def aggregate_per_graph(self, feature: jax.Array) -> jax.Array:
        """Aggregate a per node feature into a per graph feature for batched graphs.

        This function aggregates a batched graph feature into a single value per graph
        and returns the aggregated feature as a vector of length `n_graph`.

        Args:
            feature: The feature to aggregate, of shape [n_node, ...].

        Returns:
            The aggregated feature, of shape [n_graph, ...].
        """
        n_graph = self.num_graphs
        segment_ids = jnp.repeat(
            jnp.arange(n_graph),
            self.n_node,
            total_repeat_length=self.nodes.positions.shape[0],
        )
        feature_per_graph = jax.ops.segment_sum(
            feature, segment_ids, num_segments=n_graph
        )

        return feature_per_graph
