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

from typing import Optional, Union

import pydantic

from mlip.models.options import Activation, RadialEnvelope
from mlip.typing import Irreps, NonNegativeInt, PositiveFloat, PositiveInt


class NequipConfig(pydantic.BaseModel):
    """The configuration / hyperparameters of the NequIP model.

    Attributes:
        num_layers: Number of NequIP layers. Default is 2.
        node_irreps: The O3 representation space of node features, with number of
                     channels that may depend on the degree `l`.
                     Default is `"128x0e + 128x0o + 64x1o + 64x1e + 4x2e + 4x2o"`.
        l_max: Maximal degree of spherical harmonics used for the angular encoding of
               edge vectors. Default is 3.
        num_bessel: The number of Bessel basis functions to use (default is 8).
        radial_net_nonlinearity: Activation function for radial MLP.
                                 Default is raw_swish.
        radial_net_n_hidden: Number of hidden features in radial MLP. Default is 64.
        radial_net_n_layers: Number of layers in radial MLP. Default is 2.
        radial_envelope: The radial envelope function, by default it
                         is ``"polynomial_envelope"``.
                         The only other option is ``"soft_envelope"``.
        scalar_mlp_std: Standard deviation of weight init. of radial MLP.
                        Default is 4.0.
        atomic_energies: How to treat the atomic energies. If set to ``None`` (default)
                         or the string ``"average"``, then the average atomic energies
                         stored in the dataset info are used. It can also be set to the
                         string ``"zero"`` which means not to use any atomic energies
                         in the model. Lastly, one can also pass an atomic energies
                         dictionary via this parameter different from the one in the
                         dataset info, that is used.
        avg_num_neighbors: The mean number of neighbors for atoms. If ``None``
                           (default), use the value from the dataset info.
                           It is used to rescale messages by this value.
        num_species: The number of elements (atomic species descriptors) allowed.
                     If ``None`` (default), infer the value from the atomic energies
                     map in the dataset info.
    """

    num_layers: PositiveInt = 2
    node_irreps: Irreps = "128x0e + 128x0o + 64x1o + 64x1e + 4x2e + 4x2o"
    l_max: NonNegativeInt = 3
    num_bessel: PositiveInt = 8
    radial_net_nonlinearity: Activation = Activation.SWISH
    radial_net_n_hidden: PositiveInt = 64
    radial_net_n_layers: PositiveInt = 2
    radial_envelope: RadialEnvelope = RadialEnvelope.POLYNOMIAL
    scalar_mlp_std: PositiveFloat = 4.0
    atomic_energies: Optional[Union[str, dict[int, float]]] = None
    avg_num_neighbors: Optional[float] = None
    num_species: Optional[PositiveInt] = None
