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


from pydantic import model_validator
from typing_extensions import Self

from mlip.models.config import MLIPNetworkConfig
from mlip.models.options import Activation, RadialBasis
from mlip.models.visnet.visnet_helpers import VecNormType
from mlip.typing import NonNegativeInt, PositiveInt


class VisnetConfig(MLIPNetworkConfig):
    """Hyperparameters for the ViSNet model.

    Attributes:
        num_layers: Number of ViSNet layers. Default is 2.
        num_channels: The number of channels. Default is 256.
        l_max: Highest harmonic order included in the Spherical Harmonics series.
               Default is 2.
        num_heads: Number of heads in the attention block. Default is 8.
        num_rbf: Number of basis functions used in the embedding block. Default is 32.
        trainable_rbf: Whether to add learnable weights to each of the radial embedding
                       basis functions. Default is `False`.
        activation: Activation function for the output block. Options are "silu"
                    (default), "ssp" (which is shifted softplus), "tanh", "sigmoid", and
                    "swish".
        attn_activation: Activation function for the attention block. Options are "silu"
                         (default), "ssp" (which is shifted softplus), "tanh",
                         "sigmoid", and "swish".
        vecnorm_type: The type of the vector norm. The options are "none" (default),
                      "max_min", and "rms".
        num_readout_heads: Number of readout heads. The default is 1.
        radial_basis: The type of radial embedding. Options are "bessel", "gauss" and
                  "expnorm" (default).
        add_atomic_energies: Whether to add atomic energies to the final energies.
                             Default is `True`.
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
        deterministic_scatter_ops: Whether to use deterministic scatter operations in
            the forward pass, ensuring deterministic energy outputs. Setting to
            `True` makes prediction slower. Default is `False`.
        use_remat: Whether to rematerialize `VisnetLayer`. Can result in
            significant memory cost reduction, but higher runtime. Can be used for
            training, and switched off at inference. Default is `False`.
            Note that config values of a trained force field can be adjusted after
            loading via `force_field = force_field.replace_config(**kwargs)`.
    """

    num_layers: PositiveInt = 4
    num_channels: PositiveInt = 256
    l_max: NonNegativeInt = 2
    num_heads: PositiveInt = 8
    num_rbf: PositiveInt = 32
    trainable_rbf: bool = False
    activation: Activation = Activation.SILU
    attn_activation: Activation = Activation.SILU
    vecnorm_type: VecNormType = VecNormType.NONE
    num_readout_heads: PositiveInt = 1
    radial_basis: RadialBasis | str = RadialBasis.EXPNORM
    num_species: PositiveInt | None = None
    predict_partial_charges: bool = False
    use_coulomb_term: bool = False
    use_total_charge_embedding: bool = False
    embed_activation: Activation = Activation.SILU
    deterministic_scatter_ops: bool = False
    use_remat: bool = False

    @model_validator(mode="after")
    def _enforce_partial_charges_for_coulomb_term(self) -> Self:
        if self.use_coulomb_term:
            self.predict_partial_charges = True
        return self
