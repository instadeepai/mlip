# Copyright 2025 Zhongguancun Academy
# Based on code initially developed by Thorben Frank (https://github.com/thorben-frank/mlff) under MIT license.

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp

from mlip.data.dataset_info import DatasetInfo
from mlip.models.atomic_energies import get_atomic_energies
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.so3krates.blocks import (
    AlignedLinear, MLP, ResidualMLP, FeatureBlock, GeometricBlock, InteractionBlock,
    ZBLRepulsion, aligned_norm
)
from mlip.models.so3krates.config import So3kratesConfig
from mlip.models.cutoff import parse_cutoff
from mlip.models.radial_basis import parse_radial_basis
from mlip.utils.safe_norm import safe_norm


class So3krates(MLIPNetwork):
    """The So3krates model flax module. It is derived from the
    :class:`~mlip.models.mlip_network.MLIPNetwork` class.

    References:
        * Frank Thorben, Oliver Unke and Klaus-Robert Müller. So3krates: Equivariant
          attention for interactions on arbitrary length-scales in molecular systems.
          Advances in Neural Information Processing Systems, 35, Dec 2022.
          URL: https://proceedings.neurips.cc/paper_files/paper/2022/hash/bcf4ca90a8d405201d29dd47d75ac896-Abstract-Conference.html

    Attributes:
        config: Hyperparameters / configuration for the So3krates model, see
                :class:`~mlip.models.so3krates.config.So3kratesConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = So3kratesConfig

    config: So3kratesConfig
    dataset_info: DatasetInfo

    @nn.compact
    def __call__(
        self,
        edge_vectors: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        **_kwargs, # ignore any additional kwargs
    ) -> jax.Array:

        e3nn.config("gradient_normalization", "path")

        r_max = self.dataset_info.cutoff_distance_angstrom

        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.atomic_energies_map)

        avg_num_neighbors = self.config.avg_num_neighbors
        if avg_num_neighbors is None:
            avg_num_neighbors = self.dataset_info.avg_num_neighbors

        rad_filter_features = [self.config.rad_hidden_channels, self.config.num_channels]
        sph_filter_features = [self.config.sph_hidden_channels, self.config.num_channels]

        so3krates_kwargs = dict(
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            num_channels=self.config.num_channels,
            num_rbf=self.config.num_rbf,
            l_max=self.config.l_max,
            irreps_mul=self.config.irreps_mul,
            fb_rad_filter_features=rad_filter_features,
            gb_rad_filter_features=rad_filter_features,
            fb_sph_filter_features=sph_filter_features,
            gb_sph_filter_features=sph_filter_features,
            radial_basis_fn=self.config.radial_basis_fn,
            sphc_normalization=self.config.sphc_normalization,
            scalar_num_scale=self.config.scalar_num_scale,
            num_ib_linear=self.config.num_ib_linear,
            residual_mlp_1=self.config.residual_mlp_1,
            residual_mlp_2=self.config.residual_mlp_2,
            normalization=self.config.normalization,
            activation=self.config.activation,
            cutoff=r_max,
            num_species=num_species,
            avg_num_neighbors=avg_num_neighbors,
        )

        representation_model = So3kratesBlock(**so3krates_kwargs)

        # This will be used by the ZBL repulsion term
        distances = safe_norm(edge_vectors, axis=-1)
        cutoffs = parse_cutoff(self.config.radial_cutoff_fn)(r_max)(distances)

        node_energies = representation_model(
            edge_vectors, distances, cutoffs, node_species, senders, receivers
        )
        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies

        if self.config.zbl_repulsion:
            index_to_z = tuple(sorted(self.dataset_info.atomic_energies_map.keys()))
            e_rep = ZBLRepulsion(index_to_z)(
                node_species, distances, cutoffs, senders, receivers
            )
            node_energies += e_rep - self.config.zbl_repulsion_shift

        atomic_energies_ = get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species
        )
        atomic_energies_ = jnp.asarray(atomic_energies_)
        node_energies += atomic_energies_[node_species]  # [n_nodes, ]

        return node_energies


class So3kratesBlock(nn.Module):
    num_layers: int
    num_channels: int
    num_species: int
    num_rbf: int
    l_max: int
    irreps_mul: int
    fb_rad_filter_features: tuple[int, ...]
    gb_rad_filter_features: tuple[int, ...]
    fb_sph_filter_features: tuple[int, ...]
    gb_sph_filter_features: tuple[int, ...]
    cutoff: float = 5.0
    radial_basis_fn: str = 'phys'
    sphc_normalization: float | None = None
    scalar_num_scale: int | None = None
    num_ib_linear: int | None = None
    activation: str = 'silu'
    num_heads: int = 4
    residual_mlp_1: bool = False
    residual_mlp_2: bool = False
    normalization: bool = False
    # In the original So3krates repo, this scaling factor does not exist. But for deeper networks,
    # this is necessary to ensure that the initial loss does not explode.
    avg_num_neighbors: float = 1.0

    @nn.compact
    def __call__(self,
        edge_vectors: jax.Array,
        distances: jax.Array,
        cutoffs: jax.Array,
        node_species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> jax.Array:
        edge_feats = parse_radial_basis(self.radial_basis_fn)(
            self.cutoff, self.num_rbf, trainable=False
        )(distances)

        # This implementation differs from the original So3krates repo: (1) the coefficents are
        # different (So3krates uses physics convention, while here uses math convention), and
        # (2) the output components follow a m-ordering in So3krates, while it is in Cartesian
        # here. These are equivalent.
        edge_vectors = e3nn.IrrepsArray('1e', edge_vectors)
        edge_sh = e3nn.spherical_harmonics(range(1, self.l_max + 1), edge_vectors, True)
        edge_sh = e3nn.concatenate([edge_sh] * self.irreps_mul)

        # Initalize node features and spherical harmonic coordinates (SPHCs)
        node_feats = nn.Embed(self.num_species, self.num_channels)(node_species)
        if self.sphc_normalization is not None:
            chi = e3nn.scatter_sum(
                edge_sh * cutoffs[:, None], dst=receivers, output_size=node_species.shape[0]
            ) / self.sphc_normalization
        else:
            chi = None

        for _ in range(self.num_layers):
            node_feats, chi = So3kratesLayer(
                self.fb_rad_filter_features,
                self.gb_rad_filter_features,
                self.fb_sph_filter_features,
                self.gb_sph_filter_features,
                self.activation,
                self.num_heads,
                self.residual_mlp_1,
                self.residual_mlp_2,
                self.normalization,
                self.scalar_num_scale,
                self.num_ib_linear,
                self.avg_num_neighbors
            )(
                node_feats=node_feats,
                chi=chi,
                edge_feats=edge_feats,
                edge_sh=edge_sh,
                cutoffs=cutoffs,
                senders=senders,
                receivers=receivers
            )

        node_energies = MLP([node_feats.shape[-1], 1], self.activation)(
            node_feats
        ).squeeze(axis=-1)

        return node_energies


class So3kratesLayer(nn.Module):
    fb_rad_filter_features: tuple[int, ...]
    gb_rad_filter_features: tuple[int, ...]
    fb_sph_filter_features: tuple[int, ...]
    gb_sph_filter_features: tuple[int, ...]
    activation: str = 'silu'
    num_heads: int = 4
    residual_mlp_1: bool = False
    residual_mlp_2: bool = False
    normalization: bool = False
    scalar_num_scale: int | None = None
    num_ib_linear: int | None = None
    avg_num_neighbors: float = 1.0

    @nn.compact
    def __call__(
        self,
        node_feats: jax.Array,
        chi: e3nn.IrrepsArray | None,
        edge_feats: jax.Array,
        edge_sh: e3nn.IrrepsArray,
        cutoffs: jax.Array,
        senders: jax.Array,
        receivers: jax.Array
    ) -> tuple[jax.Array, e3nn.IrrepsArray]:
        if chi is not None:
            if self.scalar_num_scale is not None:
                chi_senders, chi_receivers = AlignedLinear(
                    2 * chi.irreps.regroup().mul_gcd * self.scalar_num_scale, split=2
                )(chi)
                chi_ij = chi_senders[senders] - chi_receivers[receivers]
            else:
                chi_ij = chi[senders] - chi[receivers]
            chi_scalar = aligned_norm(chi_ij)
        else:
            chi_scalar = None

        # first block
        node_feats_pre = nn.LayerNorm()(node_feats) if self.normalization else node_feats

        diff_node_feats = FeatureBlock(
            self.num_heads,
            rad_features=self.fb_rad_filter_features,
            sph_features=self.fb_sph_filter_features,
            activation=self.activation,
            avg_num_neighbors=self.avg_num_neighbors
        )(
            node_feats=node_feats_pre,
            edge_feats=edge_feats,
            chi_scalar=chi_scalar,
            cutoffs=cutoffs,
            senders=senders,
            receivers=receivers
        )

        node_feats = node_feats + diff_node_feats
        node_feats_pre = nn.LayerNorm()(node_feats) if self.normalization else node_feats

        diff_chi = GeometricBlock(
            rad_features=self.gb_rad_filter_features,
            sph_features=self.gb_sph_filter_features,
            activation=self.activation,
            avg_num_neighbors=self.avg_num_neighbors
        )(
            edge_sh=edge_sh,
            node_feats=node_feats_pre,
            edge_feats=edge_feats,
            chi_scalar=chi_scalar,
            cutoffs=cutoffs,
            senders=senders,
            receivers=receivers
        )

        if chi is None:
            chi = diff_chi
        else:
            chi = chi + diff_chi

        # second block
        if self.residual_mlp_1:
            node_feats = ResidualMLP(activation=self.activation)(node_feats)

        node_feats_pre = nn.LayerNorm()(node_feats) if self.normalization else node_feats

        diff_node_feats, diff_chi = InteractionBlock(
            self.scalar_num_scale, self.num_ib_linear
        )(node_feats_pre, chi)

        node_feats = node_feats + diff_node_feats
        chi = chi + diff_chi

        if self.residual_mlp_2:
            node_feats = ResidualMLP(activation=self.activation)(node_feats)

        return node_feats, chi
