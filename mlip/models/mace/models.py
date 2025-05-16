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

from typing import Callable, Optional

import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp

from mlip.data.dataset_info import DatasetInfo
from mlip.models.atomic_energies import get_atomic_energies
from mlip.models.blocks import (
    FullyConnectedTensorProduct,
    LinearNodeEmbeddingBlock,
    RadialEmbeddingBlock,
)
from mlip.models.mace.blocks import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
)
from mlip.models.mace.config import MaceConfig
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.options import parse_activation, parse_radial_envelope
from mlip.models.radial_embedding import bessel_basis
from mlip.utils.safe_norm import safe_norm


class Mace(MLIPNetwork):
    """The MACE model flax module. It is derived from the
    :class:`~mlip.models.mlip_network.MLIPNetwork` class.

    References:
        * Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner,
          and Gábor Csányi. Mace: Higher order equivariant message passing
          neural networks for fast and accurate force fields, 2023.
          URL: https://arxiv.org/abs/2206.07697.

    Attributes:
        config: Hyperparameters / configuration for the MACE model, see
                :class:`~mlip.models.mace.config.MaceConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = MaceConfig

    config: MaceConfig
    dataset_info: DatasetInfo

    @nn.compact
    def __call__(
        self,
        edge_vectors: jnp.ndarray,
        node_species: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> jnp.ndarray:

        e3nn.config("path_normalization", "path")
        e3nn.config("gradient_normalization", "path")

        r_max = self.dataset_info.cutoff_distance_angstrom

        avg_num_neighbors = self.config.avg_num_neighbors
        if avg_num_neighbors is None:
            avg_num_neighbors = self.dataset_info.avg_num_neighbors

        avg_r_min = self.config.avg_r_min
        if avg_r_min is None:
            avg_r_min = self.dataset_info.avg_r_min_angstrom

        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.atomic_energies_map)

        radial_envelope_fun = parse_radial_envelope(self.config.radial_envelope)

        node_symmetry = self.config.node_symmetry
        if node_symmetry is None:
            node_symmetry = self.config.l_max
        elif node_symmetry > self.config.l_max:
            raise ValueError("Message symmetry must be lower or equal to 'l_max'")

        readout_mlp_irreps, output_irreps = self.config.readout_irreps

        mace_block_kwargs = dict(
            r_max=r_max,
            num_channels=self.config.num_channels,
            avg_num_neighbors=avg_num_neighbors,
            num_interactions=self.config.num_layers,
            avg_r_min=avg_r_min,
            num_species=num_species,
            num_bessel=self.config.num_bessel,
            radial_basis=bessel_basis,
            radial_envelope=radial_envelope_fun,
            symmetric_tensor_product_basis=self.config.symmetric_tensor_product_basis,
            off_diagonal=False,
            l_max=self.config.l_max,
            node_symmetry=node_symmetry,
            include_pseudotensors=self.config.include_pseudotensors,
            num_readout_heads=self.config.num_readout_heads,
            readout_mlp_irreps=readout_mlp_irreps,
            correlation=self.config.correlation,
            gate=parse_activation(self.config.activation),
        )

        mace_block = MaceBlock(output_irreps=output_irreps, **mace_block_kwargs)

        # [n_nodes, num_interactions, num_heads, 0e]
        contributions = mace_block(edge_vectors, node_species, senders, receivers)
        # [n_nodes, num_interactions, num_heads]
        contributions = contributions.array[:, :, :, 0]

        sum_over_heads = jnp.sum(contributions, axis=2)  # [n_nodes, num_interactions]
        node_energies = jnp.sum(sum_over_heads, axis=1)  # [n_nodes, ]

        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies

        atomic_energies_ = get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species
        )
        atomic_energies_ = jnp.asarray(atomic_energies_)
        node_energies += atomic_energies_[node_species]  # [n_nodes, ]

        return node_energies


class MaceBlock(nn.Module):
    output_irreps: e3nn.Irreps  # Irreps of the output, default 1x0e
    r_max: float
    num_interactions: int  # Number of interactions (layers), default 2
    readout_mlp_irreps: (
        e3nn.Irreps
    )  # Hidden irreps of the MLP in last readout, default 16x0e
    avg_num_neighbors: float
    num_species: int
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray]
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray]
    num_bessel: int = 8
    num_channels: int | None = None
    avg_r_min: float = None
    l_max: int = 3  # Max spherical harmonic degree, default 3
    node_symmetry: int = 1  # Max degree of node features after cluster expansion
    correlation: int = (
        3  # Correlation order at each layer (~ node_features^correlation), default 3
    )
    gate: Callable = jax.nn.silu  # activation function
    soft_normalization: Optional[float] = None
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False
    include_pseudotensors: bool = False
    node_embedding: nn.Module = LinearNodeEmbeddingBlock
    num_readout_heads: int = 1
    residual_connection_first_layer: bool = False

    @nn.compact
    def __call__(
        self,
        edge_vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
    ) -> e3nn.IrrepsArray:
        assert edge_vectors.ndim == 2 and edge_vectors.shape[1] == 3
        assert node_species.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert edge_vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)

        output_irreps = e3nn.Irreps(self.output_irreps)
        readout_mlp_irreps = e3nn.Irreps(self.readout_mlp_irreps)

        num_channels = self.num_channels
        if num_channels is None:
            raise NotImplementedError(
                "Only constant multiplicities in `node_irreps` are supported "
                "for now. Please provide `num_channels` and `node_symmetry` "
                "explicitly."
            )

        # Target of EquivariantProductBasisBlock and skip-connections
        node_irreps = e3nn.Irreps.spherical_harmonics(self.node_symmetry)

        # Target of InteractionBlock = source of EquivariantProductBasisBlock
        if not self.include_pseudotensors:
            interaction_irreps = e3nn.Irreps.spherical_harmonics(self.l_max)
        else:
            interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(self.l_max))

        # Embeddings
        node_embed = self.node_embedding(
            self.num_species,
            num_channels * e3nn.Irreps("0e"),
        )
        radial_embed = RadialEmbeddingBlock(
            r_max=self.r_max,
            avg_r_min=self.avg_r_min,
            basis_functions=self.radial_basis,
            envelope_function=self.radial_envelope,
            num_bessel=self.num_bessel,
        )

        # Embeddings
        node_feats = node_embed(node_species).astype(
            edge_vectors.dtype
        )  # [n_nodes, feature * irreps]

        if not (hasattr(edge_vectors, "irreps") and hasattr(edge_vectors, "array")):
            edge_vectors = e3nn.IrrepsArray("1o", edge_vectors)

        radial_embeddings = radial_embed(safe_norm(edge_vectors.array, axis=-1))

        # Interactions
        outputs = []
        for i in range(self.num_interactions):
            selector_tp = (i == 0) and not self.residual_connection_first_layer
            last_layer = i == self.num_interactions - 1

            node_outputs, node_feats = MaceLayer(
                selector_tp=selector_tp,
                last_layer=last_layer,
                num_channels=num_channels,
                node_irreps=node_irreps,
                interaction_irreps=interaction_irreps,
                l_max=self.l_max,
                avg_num_neighbors=self.avg_num_neighbors,
                activation=self.gate,
                num_species=self.num_species,
                correlation=self.correlation,
                output_irreps=output_irreps,
                readout_mlp_irreps=readout_mlp_irreps,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                soft_normalization=self.soft_normalization,
                name=f"layer_{i}",
                num_readout_heads=self.num_readout_heads,
            )(
                edge_vectors,
                node_feats,
                node_species,
                radial_embeddings,
                senders,
                receivers,
                node_mask,
            )
            outputs += [node_outputs]  # list of [n_nodes, num_heads, output_irreps]

        return e3nn.stack(
            outputs, axis=1
        )  # [n_nodes, num_interactions, num_heads, output_irreps]


class MaceLayer(nn.Module):
    selector_tp: bool
    last_layer: bool
    num_channels: int
    node_irreps: e3nn.Irreps
    interaction_irreps: e3nn.Irreps
    activation: Callable
    num_species: int
    name: Optional[str]
    # InteractionBlock:
    l_max: int
    avg_num_neighbors: float
    # EquivariantProductBasisBlock:
    correlation: int
    symmetric_tensor_product_basis: bool
    off_diagonal: bool
    soft_normalization: Optional[float]
    # ReadoutBlock:
    output_irreps: e3nn.Irreps
    readout_mlp_irreps: e3nn.Irreps
    num_readout_heads: int = 1

    @nn.compact
    def __call__(
        self,
        edge_vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        radial_embeddings: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
    ):
        interaction_irreps = e3nn.Irreps(self.interaction_irreps)
        node_irreps = e3nn.Irreps(self.node_irreps)
        output_irreps = e3nn.Irreps(self.output_irreps)
        readout_mlp_irreps = e3nn.Irreps(self.readout_mlp_irreps)

        identity = jnp.eye(self.num_species)
        node_attr = identity[node_species]

        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)

        # residual connection:
        residual_connection = None

        if not self.selector_tp:
            # Setting output_irreps
            if self.last_layer:
                residual_connection_irrep_out = self.num_channels * e3nn.Irreps("0e")
            else:
                residual_connection_irrep_out = (
                    self.num_channels * e3nn.Irreps(self.node_irreps).regroup()
                )

            residual_connection = FullyConnectedTensorProduct(
                irreps_in1=node_feats.irreps,
                irreps_in2=self.num_species * e3nn.Irreps("0e"),
                irreps_out=residual_connection_irrep_out,
            )(x1=node_feats, x2=node_attr)

        # Interaction block
        node_feats = InteractionBlock(
            target_irreps=self.num_channels * interaction_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            l_max=self.l_max,
            activation=self.activation,
        )(
            edge_vectors=edge_vectors,
            node_feats=node_feats,
            radial_embeddings=radial_embeddings,
            receivers=receivers,
            senders=senders,
        )

        # selector tensor product (first layer only)
        if self.selector_tp:
            node_feats = FullyConnectedTensorProduct(
                irreps_in1=self.num_channels * interaction_irreps,
                irreps_in2=self.num_species * e3nn.Irreps("0e"),
                irreps_out=self.num_channels * interaction_irreps,
            )(x1=node_feats, x2=node_attr)

        # Exponentiate node features, keep degrees < node_symmetry only
        if self.last_layer:
            node_feats = EquivariantProductBasisBlock(
                target_irreps=self.num_channels * e3nn.Irreps("0e"),
                correlation=self.correlation,
                num_species=self.num_species,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
            )(node_feats=node_feats, node_species=node_species)
        else:
            node_feats = EquivariantProductBasisBlock(
                target_irreps=self.num_channels * node_irreps,
                correlation=self.correlation,
                num_species=self.num_species,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
            )(node_feats=node_feats, node_species=node_species)

        if self.soft_normalization is not None:

            def phi(n):
                n = n / self.soft_normalization
                return 1.0 / (1.0 + n * e3nn.sus(n))

            node_feats = e3nn.norm_activation(
                node_feats, [phi] * len(node_feats.irreps)
            )
        if residual_connection is not None:
            node_feats = (
                node_feats + residual_connection
            )  # [n_nodes, feature * hidden_irreps]

        # Multi-head readout
        node_outputs = []

        if not self.last_layer:
            for _head_idx in range(self.num_readout_heads):
                node_outputs += [
                    LinearReadoutBlock(output_irreps)(node_feats)
                ]  # [n_nodes, output_irreps]
        else:  # Non-linear readout for last layer
            for _head_idx in range(self.num_readout_heads):
                node_outputs += [
                    NonLinearReadoutBlock(
                        readout_mlp_irreps,
                        output_irreps,
                        activation=self.activation,
                    )(node_feats)
                ]  # [n_nodes, output_irreps]

        node_outputs = e3nn.stack(
            node_outputs, axis=1
        )  # [n_nodes, num_heads, output_irreps]

        return node_outputs, node_feats
