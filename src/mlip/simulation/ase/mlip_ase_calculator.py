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

import logging
from math import ceil

import jax
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from matscipy.neighbours import neighbour_list

from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.models import ForceField
from mlip.simulation.utils import create_graph_from_atoms
from mlip.utils.no_pbc_cell import get_no_pbc_cell

logger = logging.getLogger("mlip")


class MLIPForceFieldASECalculator(Calculator):
    """Atomic Simulation Environment (ASE) Calculator for JAX models.

    Implemented properties are energy and forces.
    """

    implemented_properties = [
        "energy",
        "forces",
    ]

    def __init__(
        self,
        atoms: Atoms,
        edge_capacity_multiplier: float,
        force_field: ForceField,
        allow_nodes_to_change: bool = False,
        node_capacity_multiplier: float = 1.0,
    ) -> None:
        """Constructor.

        Args:
            atoms: Initial atomic structure.
            edge_capacity_multiplier: Factor to multiply the number of edges by to
                                      obtain the edge capacity including padding.
            force_field: Force field model used to compute the predictions.
            allow_nodes_to_change: Whether the number or types of atoms/nodes may
                                   change for the same instance of this class. Defaults
                                   to ``False``.
            node_capacity_multiplier: Factor to multiply the number of nodes by to
                                      obtain the node capacity including padding.
                                      Defaults to 1.0.
        """
        self.atoms = atoms
        self.num_atoms = len(self.atoms)
        self.model_apply_fun = jax.jit(force_field.predictor.apply)
        self.model_params = force_field.params
        self.graph_cutoff_angstrom = force_field.cutoff_distance
        self.allowed_atomic_numbers = force_field.allowed_atomic_numbers
        self.edge_capacity_multiplier = edge_capacity_multiplier
        self.allow_nodes_to_change = allow_nodes_to_change
        self.node_capacity_multiplier = node_capacity_multiplier

        if np.any(atoms.pbc):
            senders, receivers, shifts = neighbour_list(
                quantities="ijS",
                cell=atoms.cell,
                pbc=atoms.pbc,
                positions=atoms.positions,
                cutoff=self.graph_cutoff_angstrom,
            )
        else:
            cell, cell_origin = get_no_pbc_cell(
                atoms.positions, self.graph_cutoff_angstrom
            )
            senders, receivers, shifts = neighbour_list(
                quantities="ijS",
                cell=cell,
                cell_origin=cell_origin,
                pbc=atoms.pbc,
                positions=atoms.positions,
                cutoff=self.graph_cutoff_angstrom,
            )

        num_edges = len(senders)

        _displacement_fun = None
        self.base_graph = create_graph_from_atoms(
            self.atoms,
            senders,
            receivers,
            _displacement_fun,
            self.allowed_atomic_numbers,
            cell=self.atoms.cell,
            shifts=shifts,
        )
        self.current_edge_capacity = ceil(self.edge_capacity_multiplier * num_edges)
        self.current_node_capacity = ceil(
            self.node_capacity_multiplier * len(self.atoms)
        )
        Calculator.__init__(self)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Compute properties (``forces`` and/or ``energy``) and save them in
        ``self.results`` dictionary for ASE simulation.

        Args:
            atoms: Atomic structure. Defaults to ``None``.
            properties: List of what needs to be calculated.
                        Can be any combination of ``"energy"``, ``"forces"``.
                        Defaults to ``None``.
            system_changes: List of what has changed since last calculation.
                            Can be any combination of these six: ``"positions"``,
                            ``"numbers"``, ``"cell"``, ``"pbc"``, ``initial_charges``
                            and ``"initial_magmoms"``.
                            Defaults to ``ase.calculators.calculator.all_changes``.
        """
        if atoms is None:
            raise ValueError("Variable atoms should not be None.")
        if properties is None:
            properties = ["energy", "forces"]
        Calculator.calculate(self, atoms, properties, system_changes)

        # compute new edge info
        if np.any(atoms.pbc):
            senders, receivers, shifts = neighbour_list(
                quantities="ijS",
                cell=atoms.cell,
                pbc=atoms.pbc,
                positions=atoms.positions,
                cutoff=self.graph_cutoff_angstrom,
            )
        else:
            cell, cell_origin = get_no_pbc_cell(
                atoms.positions, self.graph_cutoff_angstrom
            )
            senders, receivers, shifts = neighbour_list(
                quantities="ijS",
                cell=cell,
                cell_origin=cell_origin,
                pbc=atoms.pbc,
                positions=atoms.positions,
                cutoff=self.graph_cutoff_angstrom,
            )

        # See if padding still enough
        num_edges = len(senders)
        if self.current_edge_capacity < num_edges:
            self.current_edge_capacity = ceil(self.edge_capacity_multiplier * num_edges)
            logger.debug(
                "The edge capacity has been reset to %s.", self.current_edge_capacity
            )
        if self.allow_nodes_to_change and self.current_node_capacity < len(atoms):
            self.current_node_capacity = ceil(
                self.node_capacity_multiplier * len(atoms)
            )
            logger.debug(
                "The edge capacity has been reset to %s.", self.current_edge_capacity
            )

        if self.allow_nodes_to_change:
            _displacement_fun = None
            graph = create_graph_from_atoms(
                atoms,
                senders,
                receivers,
                _displacement_fun,
                self.allowed_atomic_numbers,
                cell=atoms.cell,
                shifts=shifts,
            )
        else:
            graph = self.base_graph._replace(
                senders=senders,
                receivers=receivers,
                nodes=self.base_graph.nodes._replace(positions=atoms.positions),
                edges=self.base_graph.edges._replace(shifts=shifts),
                n_edge=np.array([len(senders)]),
            )

        # Batch with dummy
        batched_graph = next(
            dynamically_batch(
                [graph],
                n_node=self.current_node_capacity + 1,
                n_edge=self.current_edge_capacity + 1,
                n_graph=2,
            )
        )

        # Run predictions
        predictions = self.model_apply_fun(self.model_params, batched_graph)
        if "energy" in properties:
            self.results["energy"] = np.array(predictions.energy[0])
        if "forces" in properties:
            self.results["forces"] = np.array(predictions.forces)[: len(atoms), :]
