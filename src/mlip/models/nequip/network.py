# Copyright (c) 2021 The President and Fellows of Harvard College
# Copyright (c) 2025 The NequIP Developers
#
# Licensed under the MIT License (https://opensource.org/licenses.MIT)
#
# Copyright (c) 2026 InstaDeep Ltd
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
from mlip.models.nequip.blocks import NequipEmbeddingBlock, NequipMultiHeadReadoutBlock
from mlip.models.nequip.config import NequipConfig
from mlip.models.nequip.layer import NequipLayer
from mlip.models.nequip.nequip_helpers import split_target_node_irreps
from mlip.models.options import parse_activation
from mlip.models.readout import select_head
from mlip.typing.properties import Properties

AVG_R_MIN = None


class Nequip(MLIPNetwork):
    """The NequIP model flax module. It is derived from the
    :class:`~mlip.models.mlip_network.MLIPNetwork` class.

    References:
        * Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger,
          Jonathan P. Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E. Smidt,
          and Boris Kozinsky. E(3)-equivariant graph neural networks for data-efficient
          and accurate interatomic potentials. Nature Communications, 13(1), May 2022.
          ISSN: 2041-1723. URL: https://dx.doi.org/10.1038/s41467-022-29939-5.

    Attributes:
        config: Hyperparameters / configuration for the NequIP model, see
                :class:`~mlip.models.nequip.config.NequipConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = NequipConfig

    config: NequipConfig
    dataset_info: DatasetInfo

    @property
    def available_properties(self) -> Properties:
        return Properties(
            stress=True,
            hessian=True,
            partial_charges=self.config.predict_partial_charges,
        )

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        """Apply the NequIP model to the input graph to compute per-node energies.

        Args:
            graph: Input graph containing node and edge information.

        Returns:
            Graph with per-node energy predictions `graph.nodes.features["energy"]`.
        """

        graph = SpeciesAssignmentBlock(self.dataset_info)(graph)
        if self.config.use_total_charge_embedding:
            graph = ChargeIndexAssignmentBlock(self.dataset_info)(graph)

        avg_num_neighbors = self.config.avg_num_neighbors
        if avg_num_neighbors is None:
            avg_num_neighbors = self.dataset_info.avg_num_neighbors

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

        # Embedding block
        graph = NequipEmbeddingBlock(
            num_species=num_species,
            num_charges=num_charges,
            target_irreps=self.config.target_irreps,
            l_max=self.config.l_max,
            num_rbf=self.config.num_rbf,
            r_max=self.dataset_info.graph_cutoff_angstrom,
            avg_r_min=AVG_R_MIN,
            radial_envelope=self.config.radial_envelope,
            activation_fn=parse_activation(self.config.embed_activation),
        )(graph)

        # Compute expected source/target irreps for checking I/O inside the layers.
        spherical_embedding_irreps = graph.edges.features["spherical_embedding"].irreps
        target_node_irreps = e3nn.Irreps(self.config.target_irreps)

        # Only scalar channels, non-scalar channels accumulate over layers.
        source_node_irreps = target_node_irreps.filter("0e").regroup()

        # Interaction layers
        for _ in range(self.config.num_layers):
            graph = NequipLayer(
                target_irreps=self.config.target_irreps,
                source_node_irreps=source_node_irreps,
                l_max=self.config.l_max,
                num_species=num_species,
                num_rbf=self.config.num_rbf,
                use_residual_connection=self.config.use_residual_connection,
                nonlinearities=self.config.gate_nonlinearities,
                radial_mlp_hidden=self.config.radial_mlp_hidden,
                radial_mlp_activation=self.config.radial_mlp_activation,
                radial_mlp_variance_scale=self.config.radial_mlp_variance_scale,
                avg_num_neighbors=avg_num_neighbors,
                deterministic_scatter_ops=self.config.deterministic_scatter_ops,
            )(graph)

            # Update `source_node_irreps` for next layer = scalars + nonscalars.
            irreps_scalars, _, irreps_nonscalars = split_target_node_irreps(
                source_node_irreps, spherical_embedding_irreps, target_node_irreps
            )
            source_node_irreps = irreps_scalars + irreps_nonscalars

        # Readout block
        graph = NequipMultiHeadReadoutBlock(
            source_node_irreps=source_node_irreps,
            num_heads=self.config.num_readout_heads,
            predict_partial_charges=self.config.predict_partial_charges,
        )(graph)

        graph = select_head(graph)

        node_outputs = graph.nodes.features["outputs"].array
        graph = graph.update_node_features(energy=node_outputs[..., 0])
        if self.config.predict_partial_charges:
            graph = graph.update_node_features(partial_charges=node_outputs[..., 1])
            graph = MaskPaddedNodeOutputsBlock(feature_names=("partial_charges",))(
                graph
            )
            # We store the total charge from the non-corrected partial charges for
            # the total charge term of the loss during training.
            graph = graph.update_global_features(
                non_corrected_charge=graph.aggregate_per_graph(
                    graph.nodes.features["partial_charges"]
                )
            )
            graph = correct_partial_charge_feature(graph)

        # Apply atomic energies block
        graph = AtomicEnergiesBlock(
            dataset_info=self.dataset_info,
            skip_atomic_energies_addition=not self.config.add_atomic_energies,
        )(graph)

        graph = MaskPaddedNodeOutputsBlock(feature_names=("energy",))(graph)
        return graph
