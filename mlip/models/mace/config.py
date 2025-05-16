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

import e3nn_jax as e3nn
from pydantic import BaseModel, model_validator
from typing_extensions import Self

from mlip.models.options import Activation, RadialEnvelope
from mlip.typing.fields import Irreps, NonNegativeInt, PositiveInt


class MaceConfig(BaseModel):
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
        num_bessel: The number of Bessel basis functions to use (default is 8).
        activation: The activation function used in the non-linear readout block.
                    The options are `"silu"`, `"elu"`, `"relu"`, `"tanh"`,
                    `"sigmoid"`, and `"swish"`. The default is `"silu"`.
        radial_envelope: The radial envelope function, by default it
                         is `"polynomial_envelope"`.
                         The only other option is `"soft_envelope"`.
        symmetric_tensor_product_basis: Whether to use a symmetric tensor product basis
                                        (default is `False`).
        atomic_energies: How to treat the atomic energies. If set to `None` (default)
                         or the string `"average"`, then the average atomic energies
                         stored in the dataset info are used. It can also be set to the
                         string `"zero"` which means not to use any atomic energies
                         in the model. Lastly, one can also pass an atomic energies
                         dictionary via this parameter different from the one in the
                         dataset info, that is used.
        avg_num_neighbors: The mean number of neighbors for atoms. If `None`
                           (default), use the value from the dataset info.
                           It is used to rescale messages by this value.
        avg_r_min: The mean minimum neighbour distance in Angstrom. If `None`
                   (default), use the value from the dataset info.
        num_species: The number of elements (atomic species descriptors) allowed.
                     If `None` (default), infer the value from the atomic energies
                     map in the dataset info.
    """

    num_layers: PositiveInt = 2
    num_channels: PositiveInt = 128
    l_max: NonNegativeInt = 3
    node_symmetry: Optional[PositiveInt] = None
    correlation: PositiveInt = 3
    readout_irreps: tuple[Irreps, ...] = ("16x0e", "0e")
    num_readout_heads: PositiveInt = 1
    include_pseudotensors: bool = False
    num_bessel: PositiveInt = 8
    activation: Activation = Activation.SILU
    radial_envelope: RadialEnvelope = RadialEnvelope.POLYNOMIAL
    symmetric_tensor_product_basis: bool = False
    atomic_energies: Optional[Union[str, dict[int, float]]] = None
    avg_num_neighbors: Optional[float] = None
    avg_r_min: Optional[float] = None
    num_species: Optional[int] = None

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
