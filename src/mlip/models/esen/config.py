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
from typing import Literal, TypeAlias, cast, get_args

from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Self

from mlip.models.config import MLIPNetworkConfig
from mlip.models.options import Activation, RadialBasis, RadialEnvelope
from mlip.typing.fields import PositiveInt

logger = logging.getLogger("mlip")

MoERoutingGlobal: TypeAlias = Literal["charge", "spin_multiplicity", "dataset_idx"]


def supported_moe_routing_globals() -> tuple[MoERoutingGlobal, ...]:
    return cast(tuple[MoERoutingGlobal, ...], get_args(MoERoutingGlobal))


class EsenMoEConfig(BaseModel):
    """Configuration for the Mixture-of-Experts routing in eSEN.

    Attributes:
        num_experts: Number of expert parameter sets.
        routing_globals: Graph-level features used as router input.
        embed_dim: Dimension of each individual global embedding. The total
            router input dimension is `embed_dim * len(routing_globals)`.
        router_hidden_dims: Hidden layer sizes for the router MLP.
        router_activation: Activation function used in the router MLP.
        embedding_type: How global values are embedded.
            `"pos_emb"` (positional / sinusoidal), `"lin_emb"` (linear),
            or `"rand_emb"` (learned lookup table).
        embedding_scale: Multiplicative scale applied to the embeddings.
    """

    num_experts: PositiveInt
    routing_globals: tuple[MoERoutingGlobal, ...] = ("charge",)
    embed_dim: PositiveInt = 64
    router_hidden_dims: list[PositiveInt] = [64, 64]
    router_activation: Activation = Activation.SILU
    embedding_type: Literal["pos_emb", "lin_emb", "rand_emb"] = "pos_emb"
    embedding_scale: float = 1.0

    @field_validator("routing_globals")
    @classmethod
    def validate_routing_globals(
        cls, value: tuple[MoERoutingGlobal, ...]
    ) -> tuple[MoERoutingGlobal, ...]:
        if len(value) == 0:
            raise ValueError("routing_globals must contain at least one global name.")
        return value

    @model_validator(mode="after")
    def validate_positional_embedding(self) -> "EsenMoEConfig":
        if self.embedding_type == "pos_emb" and self.embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even when embedding_type='pos_emb'.")
        return self


class EsenConfig(MLIPNetworkConfig):
    """The configuration / hyperparameters of the eSEN model.

    Attributes:
        num_species: The number of elements (atomic species descriptors) allowed.
                     If `None` (default), infer the value from the atomic energies
                     map in the dataset info.
        num_layers: Number of eSEN layers. Default is 4.
        sphere_channels: The number of channels for the node embedding. Default is 128.
        hidden_channels: The number of channels outputs for convolution layers and
                         MLPs in the network. Default is 128.
        edge_channels: The number of channels for the edge embedding. Default is 128.
        l_max: Highest degree of spherical harmonics used for the directional encoding
               of edge vectors, and during the convolution block. Default is 2.
        m_max: Cap on m number in the convolution layer, m features above that order
               are removed. Default is 2.
        add_atomic_energies: Whether to add atomic energies to the final energies.
                             Default is `True`.
        radial_envelope: The radial envelope function, by default it
                         is `"polynomial_envelope"`.
                         The only other option is `"soft_envelope"`.
        radial_basis: Type of radial basis function used. Two options
                      available: "bessel", "gauss", and "expnorm". Default is "gauss".
        num_rbf: Number of radial basis used in edge embedding. Default is
                            set to 32 (512 used in UMA small)
        cosine_cutoff: Whether to use the cosine cutoff envelope function
                       in the radial embedding block. Defaults to False.
        norm_type: Specifies the type of normalisation used. Three options are
                   available: "layer_norm", "layer_norm_sh", "rms_norm_sh".
                   Default is "rms_norm_sh"
        act_type: Activation type for Edgewise (convolution). Only one option available,
                  "gate", used a default.
        num_readout_heads: Number of readout heads. The default is 1.
        moe: Optional MoE configuration. When `None` (default), eSEN
             behaves as a standard model with no expert routing.
        predict_partial_charges: Whether the model will be trained to predict charges.
        use_coulomb_term: Whether to use the Coulomb term in the model for long
                          range interactions. Default is False.
        use_total_charge_embedding: Whether to use the total charge embedding. Default
                                   is False.
        embed_activation: Activation function for the embedding block. Default is
                        "silu".
        deterministic_scatter_ops: Whether to use deterministic scatter operations in
            the forward pass, ensuring deterministic energy outputs. Setting to
            `True` makes prediction slower. Default is `False`.
    """

    num_species: int | None = None
    num_layers: int = 2
    sphere_channels: int = 16
    hidden_channels: int = 16
    edge_channels: int = 16
    l_max: int = 2
    m_max: int = 2
    radial_envelope: RadialEnvelope = RadialEnvelope.POLYNOMIAL
    radial_basis: str | RadialBasis = RadialBasis.GAUSS
    num_rbf: int = 16
    basis_width_scalar: float = 2.0
    cosine_cutoff: bool = False
    norm_type: str = "rms_norm_sh"
    act_type: str = "gate"
    trainable_rbf: bool = False
    num_readout_heads: PositiveInt = 1
    moe: EsenMoEConfig | None = None
    predict_partial_charges: bool = False
    use_coulomb_term: bool = False
    use_total_charge_embedding: bool = False
    embed_activation: Activation = Activation.SILU
    deterministic_scatter_ops: bool = False

    @model_validator(mode="after")
    def _enforce_partial_charges_for_coulomb_term(self) -> Self:
        if self.use_coulomb_term:
            self.predict_partial_charges = True
        return self

    # chg_spin_emb_type: Literal["pos_emb", "lin_emb", "rand_emb"] = "pos_emb"
    # # NOTE Not built yet
    # cs_emb_grad: bool = False, # NOTE Not built yet
    # dataset_emb_grad: bool = False, # NOTE Not built yet
    # dataset_list: list[str] | None = None, # NOTE Not built yet
    # use_dataset_embedding: bool = True, # NOTE Not built yet
