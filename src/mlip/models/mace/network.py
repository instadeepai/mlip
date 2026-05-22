# MIT License
# Copyright (c) 2022 mace-jax
# See https://github.com/ACEsuit/mace-jax/blob/main/MIT.md
#
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


import e3nn_jax as e3nn
import jax.numpy as jnp

from mlip.data.dataset_info import DatasetInfo
from mlip.graph import Graph
from mlip.models.blocks import (
    AtomicEnergiesBlock,
    ChargeIndexAssignmentBlock,
    MaskPaddedNodeOutputsBlock,
    SpeciesAssignmentBlock,
)
from mlip.models.charge_utils import correct_partial_charge_feature
from mlip.models.mace.blocks import MaceEmbeddingBlock
from mlip.models.mace.config import MaceConfig
from mlip.models.mace.layer import MaceLayer
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.options import parse_activation
from mlip.models.readout import select_head
from mlip.typing.properties import Properties


class Mace(MLIPNetwork):
    """The MACE model flax module. It is derived from the
    :class:`~mlip.models.mlip_network.MLIPNetwork` class.

    References:
        * Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner,
          and Gábor Csányi. Mace: Higher order equivariant message passing
          neural networks for fast and accurate force fields, 2023.
          URL: https://arxiv.org/abs/2206.07697.

    Attributes:
        config: Hyperparameters / configuration for the MACE model, see
                :class:`~mlip.models.mace.config.MaceConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = MaceConfig

    config: MaceConfig
    dataset_info: DatasetInfo

    @property
    def available_properties(self) -> Properties:
        return Properties(
            stress=True,
            hessian=True,
            partial_charges=self.config.predict_partial_charges,
        )

    @property
    def num_species(self):
        return len(self.dataset_info.allowed_atomic_numbers)

    @property
    def num_charges(self):
        if self.config.use_total_charge_embedding:
            # We add +1 to the num_charge in order to account for the placeholder charge
            # index (0) that shifts the max index value to
            # len(available_total_charges) + 1.
            num_charges = len(self.dataset_info.available_total_charges) + 1
        else:
            num_charges = None
        return num_charges

    def setup(self):
        self.species_assignment_block = SpeciesAssignmentBlock(self.dataset_info)
        if self.config.use_total_charge_embedding:
            self.charge_idx_assignment_block = ChargeIndexAssignmentBlock(
                self.dataset_info
            )
        self.embedding_block = self._get_embedding_block()
        self.layers = [self._get_layer(i) for i in range(self.config.num_layers)]
        self.atomic_energies_block = AtomicEnergiesBlock(
            dataset_info=self.dataset_info,
            skip_atomic_energies_addition=not self.config.add_atomic_energies,
        )
        self.energy_mask_block = MaskPaddedNodeOutputsBlock(feature_names=("energy",))
        if self.config.predict_partial_charges:
            self.partial_charge_mask_block = MaskPaddedNodeOutputsBlock(
                feature_names=("partial_charges",)
            )

    def _get_embedding_block(self) -> MaceEmbeddingBlock:
        """Return the initial node and edge embedding block."""
        return MaceEmbeddingBlock(
            num_species=self.num_species,
            num_charges=self.num_charges,
            num_channels=self.config.num_channels,
            l_max=self.config.l_max,
            r_max=self.dataset_info.graph_cutoff_angstrom,
            num_rbf=self.config.num_rbf,
            radial_envelope=self.config.radial_envelope,
            avg_r_min=self.dataset_info.avg_r_min_angstrom,
            activation_fn=parse_activation(self.config.embed_activation),
        )

    def _get_layer(self, i: int) -> MaceLayer:
        """Return the i-th MaceLayer module."""

        cfg = self.config
        last_layer = i == cfg.num_layers - 1
        use_residuals = (i > 0) or cfg.residual_connection_first_layer

        # Target of EquivariantProductBasisBlock and skip-connections
        node_irreps = e3nn.Irreps.spherical_harmonics(cfg.node_symmetry)

        # Source of InteractionBlock
        source_irreps = e3nn.Irreps("0e") if i == 0 else node_irreps

        # Target of InteractionBlock = source of EquivariantProductBasisBlock
        if not cfg.include_pseudotensors:
            interaction_irreps = e3nn.Irreps.spherical_harmonics(cfg.l_max)
        else:
            interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(cfg.l_max))

        # Target of Readout blocks and MLP hidden layer
        readout_mlp_irreps, output_irreps = cfg.readout_irreps
        output_irreps = e3nn.Irreps(output_irreps)
        if self.config.predict_partial_charges:
            output_irreps = (output_irreps + e3nn.Irreps("1x0e")).simplify()

        readout_mlp_irreps = e3nn.Irreps(readout_mlp_irreps)

        return MaceLayer(
            use_residuals=use_residuals,
            last_layer=last_layer,
            l_max=cfg.l_max,
            num_species=self.num_species,
            num_channels=cfg.num_channels,
            correlation=cfg.correlation,
            interaction_irreps=interaction_irreps,
            node_irreps=node_irreps,
            source_irreps=source_irreps,
            output_irreps=output_irreps,
            num_rbf=cfg.num_rbf,
            radial_mlp_hidden=cfg.radial_mlp_hidden,
            radial_mlp_activation=cfg.radial_mlp_activation,
            avg_num_neighbors=self.dataset_info.avg_num_neighbors,
            activation=cfg.activation,
            readout_mlp_irreps=readout_mlp_irreps,
            soft_normalization=cfg.soft_normalization,
            num_readout_heads=cfg.num_readout_heads,
            gate_nodes=cfg.gate_nodes,
            deterministic_scatter_ops=cfg.deterministic_scatter_ops,
            use_gaunt_tp_message_passing=cfg.use_gaunt_tp_message_passing,
            symmetric_contraction_backend=cfg.symmetric_contraction_backend,
        )

    def __call__(self, graph: Graph) -> Graph:

        node_outputs = []

        graph = self.species_assignment_block(graph)
        if self.config.use_total_charge_embedding:
            graph = self.charge_idx_assignment_block(graph)
        graph = self.embedding_block(graph)

        for i in range(self.config.num_layers):
            graph = self.layers[i](graph)
            node_outputs.append(graph.nodes.features["outputs"])

        # stack per-layer outputs → [num_nodes, num_layers, num_heads, Nx0e]
        node_outputs = e3nn.stack(node_outputs, axis=1)

        # Sum over layers to get [num_nodes, num_heads, Nx0e]
        node_outputs = jnp.sum(node_outputs.array, axis=1)
        graph = graph.update_node_features(outputs=node_outputs)

        graph = select_head(graph)
        graph = graph.update_node_features(
            energy=graph.nodes.features["outputs"][..., 0]
        )
        graph = self.energy_mask_block(graph)

        if self.config.predict_partial_charges:
            graph = graph.update_node_features(
                partial_charges=graph.nodes.features["outputs"][..., 1]
            )
            graph = self.partial_charge_mask_block(graph)
            # We store the total charge from the non-corrected partial charges for
            # the total charge term of the loss during training.
            graph = graph.update_global_features(
                non_corrected_charge=graph.aggregate_per_graph(
                    graph.nodes.features["partial_charges"]
                )
            )
            graph = correct_partial_charge_feature(graph)

        # Add atomic energies, then mask padded nodes (matches esen, nequip, visnet)
        graph = self.atomic_energies_block(graph)
        graph = self.energy_mask_block(graph)
        return graph
