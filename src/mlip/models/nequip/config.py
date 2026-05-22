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
from pydantic import Field, model_validator
from typing_extensions import Self

from mlip.models.config import MLIPNetworkConfig
from mlip.models.options import Activation, RadialEnvelope
from mlip.typing import Irreps, NonNegativeInt, PositiveFloat, PositiveInt


class NequipConfig(MLIPNetworkConfig):
    """The configuration / hyperparameters of the NequIP model.

    Attributes:
        num_layers: Number of NequIP layers. Default is 2.
        target_irreps: Target O3 representation space for node features at each
                       layer, with number of channels that may depend on the
                       degree `l`. Each layer attempts to produce these irreps,
                       filtered to what is reachable via the tensor product.
                       Default `"128x0e + 128x0o + 64x1o + 64x1e + 4x2e + 4x2o"`.
        l_max: Maximal degree of spherical harmonics used for the angular encoding of
               edge vectors. Default is 3.
        num_rbf: The number of Bessel basis functions to use (default is 8).
        radial_envelope: The radial envelope function, by default it
                         is `"polynomial_envelope"`.
                         The only other option is `"soft_envelope"`.
        radial_mlp_hidden: Sizes of the MLP hidden layers. Default is [64, 64].
        radial_mlp_activation: Activation function for radial MLP. Default is `swish`.
        radial_mlp_variance_scale: Variance scaling parameter passed to the fan-in
                                   normal initializer of the MLP internal layers.
                                   See `jax.nn.initializers.variance_scaling`.
                                   Default is 4.0.
        num_readout_heads: Number of readout heads. The default is 1.
        add_atomic_energies: Whether to add atomic energies to the final energies.
                             Default is `True`.
        avg_num_neighbors: The mean number of neighbors for atoms. If `None`
                           (default), use the value from the dataset info.
                           It is used to rescale messages by this value.
        num_species: The number of elements (atomic species descriptors) allowed.
                     If `None` (default), infer the value from the atomic energies
                     map in the dataset info.
        predict_partial_charges: Whether the model will be trained to predict charges.
        use_coulomb_term: Whether to use the Coulomb term in the model for long
                          range interactions. Default is False.
        use_total_charge_embedding: Whether to use the total charge embedding. Default
                                    is False.
        embed_activation: Activation function for the embedding block. Default is
                        "silu".
        use_residual_connection: Whether to add the species-linear residual connection
                                 in each NequIP layer. Default is `True`.
        gate_nonlinearities: Per-parity activations used by the gate nonlinearity in
                             each NequIP layer. Keys are `"e"` (even) and `"o"`
                             (odd). Default is `{"e": Activation.SWISH,
                             "o": Activation.TANH}`.
        deterministic_scatter_ops: Whether to use deterministic scatter operations in
            the forward pass, ensuring deterministic energy outputs. Setting to
            `True` makes prediction slower. Default is `False`.
    """

    num_layers: PositiveInt = 2
    target_irreps: Irreps = "128x0e + 128x0o + 64x1o + 64x1e + 4x2e + 4x2o"
    l_max: NonNegativeInt = 3
    num_rbf: PositiveInt = 8
    radial_envelope: RadialEnvelope = RadialEnvelope.POLYNOMIAL
    radial_mlp_hidden: list[PositiveInt] = Field(default_factory=lambda: [64, 64])
    radial_mlp_activation: Activation = Activation.SWISH
    radial_mlp_variance_scale: PositiveFloat = 4.0
    num_readout_heads: PositiveInt = 1
    avg_num_neighbors: float | None = None
    num_species: PositiveInt | None = None
    predict_partial_charges: bool = False
    use_coulomb_term: bool = False
    use_total_charge_embedding: bool = False
    embed_activation: Activation = Activation.SILU
    use_residual_connection: bool = True
    gate_nonlinearities: dict[str, Activation] = {
        "e": Activation.SWISH,
        "o": Activation.TANH,
    }
    deterministic_scatter_ops: bool = False

    @model_validator(mode="after")
    def _validate_target_irreps(self) -> Self:
        """Assert target node irreps are sorted."""
        target_irreps = e3nn.Irreps(self.target_irreps)
        target_irreps_sorted = target_irreps.sort().irreps
        if not target_irreps == target_irreps_sorted:
            raise ValueError(
                "Target irreps should be sorted by degree l and matching parities "
                f"p = (-1)**l coming first.\nDid you mean {target_irreps_sorted}?"
            )
        return self

    @model_validator(mode="after")
    def _enforce_partial_charges_for_coulomb_term(self) -> Self:
        if self.use_coulomb_term:
            self.predict_partial_charges = True
        return self
