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


import flax.linen as nn

from mlip.data.dataset_info import DatasetInfo
from mlip.graph import Graph
from mlip.models.blocks import (
    AtomicEnergiesBlock,
    ChargeIndexAssignmentBlock,
    MaskPaddedNodeOutputsBlock,
    SpeciesAssignmentBlock,
)
from mlip.models.charge_utils import correct_partial_charge_feature
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.options import parse_activation
from mlip.models.readout import select_head
from mlip.models.visnet.blocks import VisnetEmbeddingBlock, VisnetMultiHeadReadoutBlock
from mlip.models.visnet.config import VisnetConfig
from mlip.models.visnet.layer import VisnetLayer
from mlip.typing.properties import Properties


class Visnet(MLIPNetwork):
    """The ViSNet model flax module. It is derived from the
    :class:`~mlip.models.mlip_network.MLIPNetwork` class.

    References:
        * Yusong Wang, Tong Wang, Shaoning Li, Xinheng He, Mingyu Li, Zun Wang,
          Nanning Zheng, Bin Shao, and Tie-Yan Liu. Enhancing geometric
          representations for molecules with equivariant vector-scalar interactive
          message passing. Nature Communications, 15(1), January 2024.
          ISSN: 2041-1723. URL: https://dx.doi.org/10.1038/s41467-023-43720-2.


    Attributes:
        config: Hyperparameters / configuration for the ViSNet model, see
                :class:`~mlip.models.visnet.config.VisnetConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
        available_properties: Model available properties,
                              see :class:`~mlip.typing.properties.Properties`.
    """

    Config = VisnetConfig

    config: VisnetConfig
    dataset_info: DatasetInfo

    @property
    def available_properties(self) -> Properties:
        return Properties(
            stress=True,
            hessian=True,
            partial_charges=self.config.predict_partial_charges,
        )

    def setup(self) -> None:
        """Initializes model layers and checks that the number of hidden channels is
        evenly divisible by the number of attention heads.
        """
        assert self.config.num_channels % self.config.num_heads == 0, (
            f"The number of hidden channels ({self.config.num_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({self.config.num_heads})"
        )

        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.allowed_atomic_numbers)

        if self.config.use_total_charge_embedding:
            # We add +1 to the num_charge in order to account for the placeholder charge
            # index (0) that shifts the max index value to
            # len(available_total_charges) + 1.
            num_charges = len(self.dataset_info.available_total_charges) + 1
        else:
            num_charges = None

        self.species_assignment_block = SpeciesAssignmentBlock(
            dataset_info=self.dataset_info,
        )
        if self.config.use_total_charge_embedding:
            self.charge_idx_assignment_block = ChargeIndexAssignmentBlock(
                dataset_info=self.dataset_info,
            )

        self.embedding_block = VisnetEmbeddingBlock(
            l_max=self.config.l_max,
            num_channels=self.config.num_channels,
            num_rbf=self.config.num_rbf,
            radial_basis=self.config.radial_basis,
            trainable_rbf=self.config.trainable_rbf,
            graph_cutoff_angstrom=self.dataset_info.graph_cutoff_angstrom,
            num_species=num_species,
            num_charges=num_charges,
            activation_fn=parse_activation(self.config.embed_activation),
            deterministic_scatter_ops=self.config.deterministic_scatter_ops,
        )

        VisnetLayerCls = nn.remat(VisnetLayer) if self.config.use_remat else VisnetLayer

        self.visnet_layers = [
            VisnetLayerCls(
                num_heads=self.config.num_heads,
                num_channels=self.config.num_channels,
                activation=self.config.activation,
                attn_activation=self.config.attn_activation,
                graph_cutoff_angstrom=self.dataset_info.graph_cutoff_angstrom,
                vecnorm_type=self.config.vecnorm_type,
                last_layer=i == self.config.num_layers - 1,
                l_max=self.config.l_max,
                deterministic_scatter_ops=self.config.deterministic_scatter_ops,
            )
            for i in range(self.config.num_layers)
        ]

        self.readout_block = VisnetMultiHeadReadoutBlock(
            num_heads=self.config.num_readout_heads,
            num_channels=self.config.num_channels,
            activation=self.config.activation,
            vecnorm_type=self.config.vecnorm_type,
            l_max=self.config.l_max,
            predict_partial_charges=self.config.predict_partial_charges,
        )

        self.atomic_energies_block = AtomicEnergiesBlock(
            dataset_info=self.dataset_info,
            skip_atomic_energies_addition=not self.config.add_atomic_energies,
        )
        self.energy_mask_block = MaskPaddedNodeOutputsBlock(feature_names=("energy",))
        if self.config.predict_partial_charges:
            self.partial_charge_mask_block = MaskPaddedNodeOutputsBlock(
                feature_names=("partial_charges",)
            )

    def __call__(self, graph: Graph) -> Graph:
        """Runs the ViSNet model forward pass on an input Graph and returns an updated
        Graph object with node-wise contributions.

        Features in output graph:
        - node-wise energy : graph.nodes.features["energy"]

        Args:
            graph: Input Graph containing atomic positions, species, and topology.

        Returns:
            Updated Graph with node-wise features.
        """
        graph = self.species_assignment_block(graph)
        if self.config.use_total_charge_embedding:
            graph = self.charge_idx_assignment_block(graph)

        graph = graph.update_edge_features(vectors=graph.edge_vectors())

        graph = self.embedding_block(graph)

        for visnet_layer in self.visnet_layers:
            graph = visnet_layer(graph)

        graph = self.readout_block(graph)

        graph = select_head(graph)
        graph = graph.update_node_features(
            energy=graph.nodes.features["outputs"][..., 0]
        )
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

        graph = self.atomic_energies_block(graph)

        graph = self.energy_mask_block(graph)

        return graph
