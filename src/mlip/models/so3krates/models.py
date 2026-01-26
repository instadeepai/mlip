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
    MLP, ResidualMLP, FeatureBlock, GeometricBlock, InteractionBlock, ZBLRepulsion
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
    ) -> jax.Array:

        r_max = self.dataset_info.cutoff_distance_angstrom

        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.atomic_energies_map)

        # Is it necessary to allow users to modify here?
        rad_filter_features = [self.config.num_channels, self.config.num_channels]
        sph_filter_features = [self.config.num_channels // 4, self.config.num_channels]

        so3krates_kwargs = dict(
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            num_channels=self.config.num_channels,
            num_rbf=self.config.num_rbf,
            chi_irreps=self.config.chi_irreps,
            fb_rad_filter_features=rad_filter_features,
            gb_rad_filter_features=sph_filter_features,
            fb_sph_filter_features=rad_filter_features,
            gb_sph_filter_features=sph_filter_features,
            radial_basis_fn=self.config.radial_basis_fn,
            sphc_normalization=self.config.sphc_normalization,
            residual_mlp_1=self.config.residual_mlp_1,
            residual_mlp_2=self.config.residual_mlp_2,
            normalization=self.config.normalization,
            activation=self.config.activation,
            cutoff=r_max,
            num_species=num_species,
            avg_num_neighbors=self.dataset_info.avg_num_neighbors
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
    chi_irreps: str
    fb_rad_filter_features: list[int]
    gb_rad_filter_features: list[int]
    fb_sph_filter_features: list[int]
    gb_sph_filter_features: list[int]
    cutoff: float = 5.0
    radial_basis_fn: str = 'phys'
    sphc_normalization: float | None = None
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
            self.cutoff, self.num_rbf
        )(distances)

        # This implementation differs from the original So3krates repo: (1) the coefficents are
        # different (So3krates uses physics convention, while here uses math convention), and
        # (2) the output components follow a m-ordering in So3krates, while it is in Cartesian
        # here. These are equivalent.
        edge_vectors = e3nn.IrrepsArray('1e', edge_vectors)
        chi_irreps = e3nn.Irreps(self.chi_irreps)
        edge_sh = e3nn.spherical_harmonics(chi_irreps, edge_vectors, True)

        # Initalize node features and spherical harmonic coordinates (SPHCs)
        node_feats = nn.Embed(self.num_species, self.num_channels)(node_species)
        if self.sphc_normalization is None:
            chi = e3nn.zeros(chi_irreps, (node_species.shape[0],), dtype=edge_vectors.dtype)
        else:
            chi = e3nn.scatter_sum(
                edge_sh * cutoffs[:, None], dst=receivers, output_size=node_species.shape[0]
            ) / self.sphc_normalization

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
    fb_rad_filter_features: list[int]
    gb_rad_filter_features: list[int]
    fb_sph_filter_features: list[int]
    gb_sph_filter_features: list[int]
    activation: str = 'silu'
    num_heads: int = 4
    residual_mlp_1: bool = False
    residual_mlp_2: bool = False
    normalization: bool = False
    avg_num_neighbors: float = 1.0

    @nn.compact
    def __call__(
        self,
        node_feats: jax.Array,
        chi: e3nn.IrrepsArray,
        edge_feats: jax.Array,
        edge_sh: e3nn.IrrepsArray,
        cutoffs: jax.Array,
        senders: jax.Array,
        receivers: jax.Array
    ) -> tuple[jax.Array, e3nn.IrrepsArray]:
        chi_ij = chi[senders] - chi[receivers]
        chi_scalar = e3nn.norm(chi_ij, squared=True, per_irrep=True).array

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

        node_feats = node_feats + diff_node_feats
        chi = chi + diff_chi

        # second block
        if self.residual_mlp_1:
            node_feats = ResidualMLP(activation=self.activation)(node_feats)

        node_feats_pre = nn.LayerNorm()(node_feats) if self.normalization else node_feats

        diff_node_feats, diff_chi = InteractionBlock()(node_feats_pre, chi)

        node_feats = node_feats + diff_node_feats
        chi = chi + diff_chi

        if self.residual_mlp_2:
            node_feats = ResidualMLP(activation=self.activation)(node_feats)

        return node_feats, chi
