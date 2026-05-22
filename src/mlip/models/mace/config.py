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

from typing import Literal

import e3nn_jax as e3nn
from pydantic import Field, model_validator
from typing_extensions import Self

from mlip.models.config import MLIPNetworkConfig
from mlip.models.options import Activation, RadialEnvelope
from mlip.typing.fields import Irreps, NonNegativeInt, PositiveInt


class MaceConfig(MLIPNetworkConfig):
    """The configuration / hyperparameters of the MACE model.

    Attributes:
        num_layers: Number of MACE layers. Default is 2.
        num_channels: The number of channels. Default is 128.
        l_max: Highest degree of spherical harmonics used for the directional encoding
               of edge vectors, and during the convolution block. Default is 3, it is
               recommended to keep it at 3.
        node_symmetry: Highest degree of node features kept after the node-wise power
                       expansion of features, also called Atomic Cluster Expansion
                       (ACE). The default behaviour is to assign `l_max`, although
                       high values of `node_symmetry` may have a significant impact
                       on runtime. It should be less or equal to `l_max`.
        correlation: Maximum correlation order, by default it is 3.
        readout_irreps: Irreps for the readout block, passed as a tuple of irreps
                        string representations for each of the layers in the
                        readout block. Currently, this MACE model only supports
                        two layers, and it defaults to `("16x0e", "0e")`.
        num_readout_heads: Number of readout heads. The default is 1. For fine-tuning,
                           additional heads must be added.
        include_pseudotensors: If `False` (default), only parities `p = (-1)**l`
                               will be kept.
                               If `True`, all parities will be kept,
                               e.g., `"1e"` pseudo-vectors returned by the cross
                               product on R3.
        num_rbf: The number of Bessel basis functions to use (default is 8).
        activation: The activation function used in the non-linear readout block.
                    The options are `"silu"`, `"elu"`, `"relu"`, `"tanh"`,
                    `"sigmoid"`, and `"swish"`. The default is `"silu"`.
        radial_envelope: The radial envelope function, by default it
                         is `"polynomial_envelope"`.
                         The only other option is `"soft_envelope"`.
        radial_mlp_hidden: Sizes of the radial MLP hidden layers.
                           Default is [64, 64, 64].
        radial_mlp_activation: Activation function for the radial MLP.
                               Default is `"silu"`.
        add_atomic_energies: Whether to add atomic energies to the final energies.
                             Default is `True`.
        avg_num_neighbors: The mean number of neighbors for atoms. If `None`
                           (default), use the value from the dataset info.
                           It is used to rescale messages by this value.
        avg_r_min: The mean minimum neighbour distance in Angstrom. If `None`
                   (default), use the value from the dataset info.
        num_species: The number of elements (atomic species descriptors) allowed.
                     If `None` (default), infer the value from the atomic energies
                     map in the dataset info.
        gate_nodes: Whether to use a gating for the self-interaction.
                    Default is `False`.
                    See our white paper for a description of this option that is
                    not present in the original MACE architecture.
        residual_connection_first_layer: Include a skip connection on the first
                                         layer, the default is false.
        soft_normalization: Node features will be regularized so that their norm
                            stay below this parameter's value (soft saturation).
                            The default is None.
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
        symmetric_contraction_backend: Which backend to use for the symmetric
            contraction, if wanting to revert to e3nn. Default is `e3j`.
        use_gaunt_tp_message_passing: Whether to use Gaunt tensor products as a
            replacement for Clebsch-Gordan tensor products in the message passing block.
            Default is `False`.
    """

    num_layers: PositiveInt = 2
    num_channels: PositiveInt = 128
    l_max: NonNegativeInt = 3
    node_symmetry: NonNegativeInt | None = None
    correlation: PositiveInt = 3
    readout_irreps: tuple[Irreps, ...] = ("16x0e", "0e")
    num_readout_heads: PositiveInt = 1
    include_pseudotensors: bool = False
    num_rbf: PositiveInt = 8
    activation: Activation = Activation.SILU
    radial_envelope: RadialEnvelope = RadialEnvelope.POLYNOMIAL
    radial_mlp_hidden: list[PositiveInt] = Field(default_factory=lambda: [64, 64, 64])
    radial_mlp_activation: Activation = Activation.SILU
    avg_num_neighbors: float | None = None
    avg_r_min: float | None = None
    num_species: int | None = None
    gate_nodes: bool = False
    residual_connection_first_layer: bool = False
    soft_normalization: float | None = None
    predict_partial_charges: bool = False
    use_coulomb_term: bool = False
    use_total_charge_embedding: bool = False
    embed_activation: Activation = Activation.SILU
    deterministic_scatter_ops: bool = False
    symmetric_contraction_backend: Literal[
        "e3j", "e3nn", "e3nn_symmetric", "gaunt_tp"
    ] = "e3j"
    use_gaunt_tp_message_passing: bool = False

    @model_validator(mode="after")
    def _validate_readout_irreps(self) -> Self:
        """Assert readout MLP has two layers of irreps type."""
        if len(self.readout_irreps) != 2:
            raise ValueError(
                "Readout irreps has to be of length 2 in the current version!"
            )
        if not all(isinstance(r, (e3nn.Irreps, str)) for r in self.readout_irreps):
            raise ValueError(
                "The representations inside the readout irreps must be of type string."
            )
        return self

    @model_validator(mode="after")
    def _validate_correlation(self) -> Self:
        """Assert correlation is less than 5."""
        if self.correlation >= 5:
            raise ValueError("correlation > 5 requires a quantum super computer.")
        return self

    @model_validator(mode="after")
    def _validate_and_set_node_symmetry(self):
        if self.node_symmetry is None:
            self.node_symmetry = self.l_max

        elif self.node_symmetry > self.l_max:
            raise ValueError("Message symmetry must be lower or equal to 'l_max'")
        return self

    @model_validator(mode="after")
    def _enforce_partial_charges_for_coulomb_term(self) -> Self:
        if self.use_coulomb_term:
            self.predict_partial_charges = True
        return self
