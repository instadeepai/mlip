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

from mlip.data.chemical_system import ChemicalSystem
from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.graph import Graph
from mlip.models import ForceField

logger = logging.getLogger("mlip")


class MLIPForceFieldASECalculator(Calculator):
    """Atomic Simulation Environment (ASE) Calculator for JAX models.

    Implemented properties are energy, forces, and partial charges (if the underlying
    force field predicts it).
    """

    implemented_properties = [
        "energy",
        "forces",
        "charges",
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
                                   to `False`. If this is set to false, the node
                                   capacity multiplier will not be updated correctly,
                                   and an error will be raised if the number of nodes
                                   does change between calls to the calculate function.
            node_capacity_multiplier: Factor to multiply the number of nodes by to
                                      obtain the node capacity including padding.
                                      Defaults to 1.0.
        """
        self.atoms = atoms
        self.num_atoms = len(self.atoms)
        self.model_apply_fun = jax.jit(force_field.predictor.apply)
        self.model_params = force_field.params
        self.graph_cutoff_angstrom = force_field.cutoff_distance
        self.long_range_cutoff_angstrom = force_field.long_range_cutoff_distance
        self.allowed_atomic_numbers = force_field.allowed_atomic_numbers
        self.edge_capacity_multiplier = edge_capacity_multiplier
        self.allow_nodes_to_change = allow_nodes_to_change
        self.node_capacity_multiplier = node_capacity_multiplier
        self.force_field = force_field

        chem_system = ChemicalSystem.from_ase_atoms(self.atoms)
        self.base_graph = Graph.from_chemical_system(
            chem_system,
            self.graph_cutoff_angstrom,
            long_range_cutoff_angstrom=self.long_range_cutoff_angstrom,
        )

        num_edges = len(self.base_graph.senders)
        self.current_edge_capacity = ceil(self.edge_capacity_multiplier * num_edges)
        self.current_node_capacity = ceil(
            self.node_capacity_multiplier * len(self.atoms)
        )
        if self.long_range_cutoff_angstrom is not None:
            num_long_range_edges = len(self.base_graph.senders_long_range)
            self.current_long_range_edge_capacity = ceil(
                self.edge_capacity_multiplier * num_long_range_edges
            )
        else:
            self.current_long_range_edge_capacity = None
        Calculator.__init__(self)

    def _prepare_graph(self, atoms: Atoms) -> Graph:
        """Prepare a batched Graph object for the given atoms.

        Args:
            atoms: The atoms object.
        Returns:
            The prepared graph, batched with a dummy graph for prediction.
        """
        chem_system = ChemicalSystem.from_ase_atoms(atoms, get_property_fields=False)

        graph = Graph.from_chemical_system(
            chem_system,
            self.graph_cutoff_angstrom,
            long_range_cutoff_angstrom=self.long_range_cutoff_angstrom,
        )

        if not self.allow_nodes_to_change:
            if len(atoms) != len(self.atoms):
                raise RuntimeError(
                    "The number of nodes changed in between two 'calculate' calls, but "
                    "'allow_nodes_to_change=False' was set."
                )

        # See if padding still enough
        num_edges = len(graph.senders)
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
                "The node capacity has been reset to %s.", self.current_node_capacity
            )
        n_edge_long_range = None
        if self.long_range_cutoff_angstrom is not None:
            num_long_range_edges = len(graph.senders_long_range)
            if self.current_long_range_edge_capacity < num_long_range_edges:
                self.current_long_range_edge_capacity = ceil(
                    self.edge_capacity_multiplier * num_long_range_edges
                )
                logger.debug(
                    "The long-range edge capacity has been reset to %s.",
                    self.current_long_range_edge_capacity,
                )
            n_edge_long_range = self.current_long_range_edge_capacity + 1

        # Batch with dummy to ensure fixed size nodes and edges between calls
        batched_graph = next(
            dynamically_batch(
                [graph],
                n_node=self.current_node_capacity + 1,
                n_edge=self.current_edge_capacity + 1,
                n_graph=2,
                n_edge_long_range=n_edge_long_range,
            )
        )

        return batched_graph

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Compute properties and save them in results dictionary for ASE simulation.

        Args:
            atoms: Atomic structure. Defaults to `None`.
            properties: List of what needs to be calculated.
                        Can be any combination of `"energy"`, `"forces"`.
                        Defaults to `None`.
            system_changes: List of what has changed since last calculation.
                            Can be any combination of these six: `"positions"`,
                            `"numbers"`, `"cell"`, `"pbc"`, `"initial_charges"`
                            and `"initial_magmoms"`.
                            Defaults to `ase.calculators.calculator.all_changes`.
        """
        if atoms is None:
            raise ValueError("Variable atoms should not be None.")
        if properties is None:
            properties = ["energy", "forces"]
        Calculator.calculate(self, atoms, properties, system_changes)

        batched_graph = self._prepare_graph(atoms)

        # Run predictions
        predictions = self.model_apply_fun(
            self.model_params, batched_graph
        ).to_prediction()

        energy = (
            predictions.energy[0] if predictions.energy.shape else predictions.energy
        )
        self.results["energy"] = np.array(energy)

        if predictions.forces is not None:
            self.results["forces"] = np.array(predictions.forces)[: len(atoms), :]

        if predictions.partial_charges is not None:
            self.results["charges"] = np.array(predictions.partial_charges)[
                : len(atoms)
            ]
