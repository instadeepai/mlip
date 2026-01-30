# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Thorben Frank (https://github.com/thorben-frank/mlff) under MIT license.

import pydantic

from mlip.models.cutoff import CutoffFunction
from mlip.models.radial_basis import RadialBasis
from mlip.models.options import Activation
from mlip.typing import PositiveInt


class So3kratesConfig(pydantic.BaseModel):
    """Hyperparameters for the So3krates model.

    Attributes:
        num_layers: Number of So3krates layers. Default is 3.
        num_channels: The number of channels. Default is 128.
        num_heads: Number of heads in the attention block. Default is 4.
        num_rbf: Number of basis functions used in the embedding block. Default is 32.
        activation: Activation function for the output block. Options are "silu"
                    (default), "ssp" (which is shifted softplus), "tanh", "sigmoid", and
                    "swish".
        radial_cutoff_fn: The type of the cutoff / radial envelope function.
        radial_basis_fn: The type of the radial basis function.
        chi_irreps: The irreps of the spherical harmonic coorindates (SPHCs).
        sphc_normalization: Normalization constant for initializing spherical harmonic
                            coordinates (SPHCs). If set to ``None``, SPHCs are initialized
                            to zero.
        residual_mlp_1: Whether to apply a residual MLP after the first (feature + 
                        geometric) update block inside each So3krates layer.
        residual_mlp_2: Whether to apply a residual MLP after the interaction block inside
                        each So3krates layer.
        normalization: Whether to apply LayerNorm to scalar node features before major
                       update blocks inside each So3krates layer.
        zbl_repulsion: Whether to include an explicit Ziegler-Biersac-Littmark (ZBL)
                       short-range nuclear repulsion term in the predicted energies.
        zbl_repulsion_shift: Constant energy shift subtracted from the ZBL repulsion
                             contribution.
        atomic_energies: How to treat the atomic energies. If set to ``None`` (default)
                         or the string ``"average"``, then the average atomic energies
                         stored in the dataset info are used. It can also be set to the
                         string ``"zero"`` which means not to use any atomic energies
                         in the model. Lastly, one can also pass an atomic energies
                         dictionary via this parameter different from the one in the
                         dataset info, that is used.
        num_species: The number of elements (atomic species descriptors) allowed.
                     If ``None`` (default), infer the value from the atomic energies
                     map in the dataset info.
    """

    num_layers: PositiveInt = 3
    num_channels: PositiveInt = 128
    num_heads: PositiveInt = 4
    num_rbf: PositiveInt = 32
    activation: Activation = Activation.SILU
    radial_cutoff_fn: CutoffFunction = CutoffFunction.PHYS
    radial_basis_fn: RadialBasis = RadialBasis.BERNSTEIN
    chi_irreps: str = "1e + 2e + 3e + 4e"
    sphc_normalization: float | None = None
    residual_mlp_1: bool = True
    residual_mlp_2: bool = False
    normalization: bool = True
    zbl_repulsion: bool = True
    zbl_repulsion_shift: float = 0.0
    atomic_energies: str | dict[int, float] | None = None
    num_species: PositiveInt | None = None
