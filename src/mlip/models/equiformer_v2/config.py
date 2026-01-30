# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Yi-Lun Liao (https://github.com/atomicarchitects/equiformer_v2) under MIT license.

import pydantic

from mlip.models.equiformer_v2.utils import AttnActType, FeedForwardType
from mlip.typing import PositiveInt, NonNegativeInt
from mlip.models.equiformer_v2.layernorm import LayerNormType


class EquiformerV2Config(pydantic.BaseModel):
    """The configuration / hyperparameters of the EquiformerV2 model.

    Attributes:
        num_layers: Number of EquiformerV2 layers. Default is 9.
        lmax: Maximum degree of the spherical harmonics (1 to 10).
        mmax: Maximum order of the spherical harmonics (0 to lmax).
        sphere_channels: Number of spherical channels. Default is 128.
        num_edge_channels: Number of channels for the edge invariant features. Default is 128.
        atom_edge_embedding: Whether to use / share atomic embedding along with relative distance.
                             Options are "none", "isolated" (default) and "shared".
        num_rbf: Number of basis functions used in the embedding block. Default is 600.
        attn_hidden_channels: Number of hidden channels used during SO(2) graph attention. Use 64
                              or 96 (not necessarily).
        num_heads: Number of heads in the attention block. Default is 8.
        attn_alpha_channels: Number of channels for alpha vector in each attention head.
        attn_value_channels: Number of channels for value vector in each attention head.
        ffn_hidden_channels: Number of hidden channels used during feedforward network.
        norm_type: Type of normalization layer. Options are "layer_norm", "layer_norm_sh" (default)
                   and "rms_norm_sh".
        grid_resolution: Resolution of SO3Grid used in Activation. Examples are 18, 16, 14, None
                         (default, decided automatically).
        use_m_share_rad: Whether all m components within a type-L vector of one channel share
                         radial function weights.
        use_attn_renorm: Whether to re-normalize attention weights.
        ff_type: Type of feedforward network. Options are "gate", "grid", "grid_sep" (default),
                 "s2" and "s2_sep". See :class:`~mlip.models.equiformer_v2.utils.FeedForwardType`
                 for its corresponding options in original EquiformerV2 repo.
        attn_act_type: Type of activation function used in the attention block. Options are "gate",
                       "s2_sep" (default) and "s2". See
                       :class:`~mlip.models.equiformer_v2.utils.AttnActType` for its corresponding
                       options in original EquiformerV2 repo.
        alpha_drop: Dropout rate for attention weights. Use 0.0 or 0.1 (default).
        drop_path_rate: Graph drop path rate. Use 0.0 or 0.05 (default).
        avg_num_nodes: The mean number of atoms per graph. If `None`, use the value from the
                       dataset info. Default is value from IS2RE (100k).
        avg_num_neighbors: The mean number of neighbors for atoms. If `None`, use the value
                           from the dataset info. Default is value from IS2RE (100k). It is
                           used to rescale messages by this value.
        atomic_energies: How to treat the atomic energies. If set to ``None`` (default) or the
                         string ``"average"``, then the average atomic energies stored in the
                         dataset info are used. It can also be set to the string ``"zero"`` which
                         means not to use any atomic energies in the model. Lastly, one can also
                         pass an atomic energies dictionary via this parameter different from the
                         one in the dataset info, that is used.
        num_species: The number of elements (atomic species descriptors) allowed.
                     If ``None`` (default), infer the value from the atomic energies
                     map in the dataset info.
        direct_force: Whether to predict forces using a direct force head instead of using
                      auto-grad. Default is `False`.
    """

    num_layers: PositiveInt = 9
    lmax: PositiveInt = 6
    mmax: NonNegativeInt = 2
    sphere_channels: PositiveInt = 128
    num_edge_channels: PositiveInt = 128
    atom_edge_embedding: str = 'isolated'
    num_rbf: PositiveInt = 600
    attn_hidden_channels: PositiveInt = 64
    num_heads: PositiveInt = 8
    attn_alpha_channels: PositiveInt = 64
    attn_value_channels: PositiveInt = 16
    ffn_hidden_channels: PositiveInt = 128
    norm_type: LayerNormType = LayerNormType.LAYER_NORM_SH
    grid_resolution: PositiveInt | None = 18 # Original default: None
    use_m_share_rad: bool = False
    use_attn_renorm: bool = True
    ff_type: FeedForwardType = FeedForwardType.GRID_SEP
    attn_act_type: AttnActType = AttnActType.S2_SEP
    alpha_drop: float = 0.1
    drop_path_rate: float = 0.05
    avg_num_neighbors: float | None = None # Original: 23.395238876342773 (OC20)
    avg_num_nodes: float | None = None # Original: 77.81317 (OC20)
    atomic_energies: str | dict[int, float] | None = None
    num_species: PositiveInt | None = None
    direct_force: bool = False
