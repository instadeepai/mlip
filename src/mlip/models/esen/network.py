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
import jax

from mlip.data.dataset_info import DatasetInfo
from mlip.data.helpers.dummy_init_graph import get_dummy_graph_for_model_init
from mlip.graph import Graph
from mlip.models.blocks import (
    AtomicEnergiesBlock,
    ChargeIndexAssignmentBlock,
    MaskPaddedNodeOutputsBlock,
    SpeciesAssignmentBlock,
)
from mlip.models.charge_utils import correct_partial_charge_feature
from mlip.models.esen.blocks import EsenEmbeddingBlock
from mlip.models.esen.coefficient_mapping import CoefficientMapping
from mlip.models.esen.config import EsenConfig
from mlip.models.esen.layer import ESENLayer
from mlip.models.esen.moe import (
    GlobalsEmbedding,
    MoERouter,
    contract_moe_params,
    resolve_routing_globals,
)
from mlip.models.esen.normalisation import get_normalization_layer
from mlip.models.inference_context import (
    InferenceContext,
    apply_inference_context_to_graph,
)
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.options import parse_activation
from mlip.models.readout import MultiHeadReadoutBlock, select_head
from mlip.typing import ModelParameters
from mlip.typing.properties import Properties


class Esen(MLIPNetwork):
    """The Esen model flax module. It is derived from the
    :class:`~mlip.models_v1.mlip_network.MLIPNetwork` class.

    References:
        * Saro Passaro, Lawrence Zitnick.
          Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs.
          URL: https://arxiv.org/pdf/2302.03655

        * Xiang Fu, Brandon M. Wood, Luis Barroso-Luque, Daniel S. Levine, Meng Gao,
          Misko Dzamba, C. Lawrence Zitnick. Learning Smooth and Expressive Interatomic
          Potentials for Physical Property Prediction.
          URL: https://arxiv.org/abs/2502.12147

        * Brandon M. Wood et al. UMA: A Family of Universal Models for Atoms.
          URL: https://arxiv.org/pdf/2506.23971

    Attributes:
        config: Hyperparameters / configuration for the Esen model, see
                :class:`~mlip.models.esen.config.EsenConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
        available_properties: Model available properties,
                              see :class:`~mlip.typing.properties.Properties`.
    """

    Config = EsenConfig

    config: EsenConfig
    dataset_info: DatasetInfo

    @property
    def is_moe_model(self) -> bool:
        """Whether this eSEN instance uses MoE parameters."""
        return self.config.moe is not None

    @property
    def available_properties(self) -> Properties:
        return Properties(
            stress=True,
            hessian=True,
            partial_charges=self.config.predict_partial_charges,
        )

    def setup(self) -> None:
        """Initializes the model layers."""

        self.num_species = self.config.num_species
        if self.num_species is None:
            self.num_species = len(self.dataset_info.allowed_atomic_numbers)
        if self.config.use_total_charge_embedding:
            # We add +1 to the num_charge in order to account for the placeholder charge
            # index (0) that shifts the max index value to
            # len(available_total_charges) + 1.
            self.num_charges = len(self.dataset_info.available_total_charges) + 1
        else:
            self.num_charges = None

        self.species_assignment_block = SpeciesAssignmentBlock(
            dataset_info=self.dataset_info,
        )

        if self.config.use_total_charge_embedding:
            self.charge_idx_assignment_block = ChargeIndexAssignmentBlock(
                dataset_info=self.dataset_info,
            )

        self.edge_channels_list = [
            self.config.num_rbf + 2 * self.config.edge_channels,
            self.config.edge_channels,
            self.config.edge_channels,
        ]
        self.mapping_reduced = CoefficientMapping(self.config.l_max, self.config.m_max)

        self.embedding_block = EsenEmbeddingBlock(
            graph_cutoff_angstrom=self.dataset_info.graph_cutoff_angstrom,
            l_max=self.config.l_max,
            num_species=self.num_species,
            num_charges=self.num_charges,
            sphere_channels=self.config.sphere_channels,
            radial_envelope=self.config.radial_envelope,
            radial_basis=self.config.radial_basis,
            num_rbf=self.config.num_rbf,
            basis_width_scalar=self.config.basis_width_scalar,
            cosine_cutoff=self.config.cosine_cutoff,
            trainable_rbf=self.config.trainable_rbf,
            edge_channels=self.config.edge_channels,
            m_max=self.config.m_max,
            mapping_reduced=self.mapping_reduced,
            edge_channels_list=self.edge_channels_list,
            activation_fn=parse_activation(self.config.embed_activation),
            use_quaternion=self.config.use_quaternion,
        )

        self.moe_config = self.config.moe
        self.num_experts = (
            None if self.moe_config is None else self.moe_config.num_experts
        )
        if self.moe_config is not None:
            self.globals_embedding = GlobalsEmbedding(
                embed_dim=self.moe_config.embed_dim,
                routing_globals=self.moe_config.routing_globals,
                embedding_type=self.moe_config.embedding_type,
                scale=self.moe_config.embedding_scale,
            )
            self.router = MoERouter(
                num_experts=self.moe_config.num_experts,
                hidden_dims=self.moe_config.router_hidden_dims,
                activation=self.moe_config.router_activation,
            )

        self.esen_layers = [
            ESENLayer(
                sphere_channels=self.config.sphere_channels,
                hidden_channels=self.config.hidden_channels,
                l_max=self.config.l_max,
                m_max=self.config.m_max,
                mapping_reduced=self.mapping_reduced,
                edge_channels_list=self.edge_channels_list,
                graph_cutoff_angstrom=self.dataset_info.graph_cutoff_angstrom,
                norm_type=self.config.norm_type,
                act_type=self.config.act_type,
                num_experts=self.num_experts,
                deterministic_scatter_ops=self.config.deterministic_scatter_ops,
                use_remat_edgewise=self.config.use_remat_edgewise,
            )
            for _ in range(self.config.num_layers)
        ]

        self.pre_readout_norm = get_normalization_layer(
            self.config.norm_type,
            l_max=self.config.l_max,
            num_channels=self.config.sphere_channels,
        )

        num_output_features = 1 + self.config.predict_partial_charges
        self.readout_block = MultiHeadReadoutBlock(
            num_heads=self.config.num_readout_heads,
            features=(
                self.config.sphere_channels,
                self.config.hidden_channels,
                num_output_features,
            ),
            activation=nn.silu,
            mlp_kernel_init=nn.initializers.lecun_normal(),
            use_equiv=False,
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

    def get_moe_coefficients(self, graph: Graph) -> jax.Array:
        if self.moe_config is None:
            raise ValueError(
                "Can't compute MoE coefficients from an eSEN model "
                "with config.moe=None."
            )

        routing_globals = resolve_routing_globals(
            graph, self.moe_config.routing_globals
        )
        embedding = self.globals_embedding(**routing_globals)
        return self.router(embedding)

    def __call__(self, graph: Graph) -> Graph:
        """Runs the Esen model forward pass on an input Graph and returns an updated
        Graph object with node-wise contributions.

        Features in output graph:
        - node-wise energy : graph.nodes.features["energy"]

        Args:
            graph: Input Graph containing atomic positions and topology.

        Returns:
            Updated Graph with node-wise features.
        """
        graph = self.species_assignment_block(graph)
        if self.config.use_total_charge_embedding:
            graph = self.charge_idx_assignment_block(graph)

        graph = graph.update_edge_features(vectors=graph.edge_vectors())

        graph = self.embedding_block(graph)

        if self.moe_config is not None:
            graph_moe_coeffs = self.get_moe_coefficients(graph)
            graph = graph.update_global_features(moe_coefficients=graph_moe_coeffs)

        for esen_layer in self.esen_layers:
            graph = esen_layer(graph)

        # Pre-readout norm
        node_feats = self.pre_readout_norm(graph.nodes.features["latent"])
        graph = graph.update_node_features(latent=node_feats[:, 0, :])

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

    @nn.nowrap
    def prepare_experts_for_inference(
        self,
        params: ModelParameters,
        inference_context: InferenceContext,
    ) -> tuple["Esen", ModelParameters]:
        """Contract MoE expert parameters into a single-expert eSEN for inference.

        Runs the router on a dummy graph populated with the given inference context
        to obtain routing coefficients, then linearly combines expert kernels into
        standard dense kernels.  Returns a new (non-MoE) model and its contracted
        parameters.
        """
        moe_config = self.config.moe
        if moe_config is None:
            return self, params

        missing_routing_globals = [
            global_name
            for global_name in moe_config.routing_globals
            if getattr(inference_context, global_name) is None
        ]
        if missing_routing_globals:
            raise ValueError(
                "Cannot specialize MoE eSEN for inference because the inference "
                "context is missing required routing globals: "
                f"{missing_routing_globals}."
            )

        graph = apply_inference_context_to_graph(
            get_dummy_graph_for_model_init(),
            inference_context,
        )
        coeffs = self.apply(
            {"params": params},
            graph,
            method=type(self).get_moe_coefficients,
        )

        contracted_model = type(self)(
            config=self.config.model_copy(update={"moe": None}),
            dataset_info=self.dataset_info,
        )
        contracted_params = contract_moe_params(params, coeffs)
        return contracted_model, contracted_params
