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

import functools
from typing import Callable, Union

import e3nn_jax as e3nn
import flax.linen as nn
import jax.numpy as jnp
from e3nn_jax.legacy import FunctionalTensorProduct

from mlip.data.dataset_info import DatasetInfo
from mlip.models.atomic_energies import get_atomic_energies
from mlip.models.blocks import FullyConnectedTensorProduct, Linear, RadialEmbeddingBlock
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.nequip.config import NequipConfig
from mlip.models.nequip.nequip_helpers import MLP, prod, tp_path_exists
from mlip.models.options import parse_activation, parse_radial_envelope
from mlip.models.radial_embedding import bessel_basis
from mlip.utils.safe_norm import safe_norm


class Nequip(MLIPNetwork):
    """The NequIP model flax module. It is derived from the
    :class:`~mlip.models.mlip_network.MLIPNetwork` class.

    References:
        * Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger,
          Jonathan P. Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E. Smidt,
          and Boris Kozinsky. E(3)-equivariant graph neural networks for data-efficient
          and accurate interatomic potentials. Nature Communications, 13(1), May 2022.
          ISSN: 2041-1723. URL: https://dx.doi.org/10.1038/s41467-022-29939-5.

    Attributes:
        config: Hyperparameters / configuration for the NequIP model, see
                :class:`~mlip.models.nequip.config.NequipConfig`.
        dataset_info: Hyperparameters dictated by the dataset
                      (e.g., cutoff radius or average number of neighbors).
    """

    Config = NequipConfig

    config: NequipConfig
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
        num_species = self.config.num_species
        if num_species is None:
            num_species = len(self.dataset_info.atomic_energies_map)

        radial_envelope_fun = parse_radial_envelope(self.config.radial_envelope)

        nequip_kwargs = dict(
            avg_num_neighbors=avg_num_neighbors,
            num_layers=self.config.num_layers,
            num_species=num_species,
            node_irreps=self.config.node_irreps,
            l_max=self.config.l_max,
            num_bessel=self.config.num_bessel,
            r_max=r_max,
            radial_net_nonlinearity=self.config.radial_net_nonlinearity,
            radial_net_n_hidden=self.config.radial_net_n_hidden,
            radial_net_n_layers=self.config.radial_net_n_layers,
            use_residual_connection=True,
            nonlinearities={"e": "swish", "o": "tanh"},
            avg_r_min=None,
            radial_basis=bessel_basis,
            radial_envelope=radial_envelope_fun,
            scalar_mlp_std=self.config.scalar_mlp_std,
        )

        nequip = NequipBlock(**nequip_kwargs)

        node_energies = nequip(edge_vectors, node_species, senders, receivers)

        mean = self.dataset_info.scaling_mean
        std = self.dataset_info.scaling_stdev
        node_energies = mean + std * node_energies

        atomic_energies_ = get_atomic_energies(
            self.dataset_info, self.config.atomic_energies, num_species
        )
        atomic_energies_ = jnp.asarray(atomic_energies_)
        node_energies += atomic_energies_[node_species]  # [n_nodes, ]

        return node_energies


class NequipBlock(nn.Module):
    avg_num_neighbors: int
    num_layers: int
    num_species: int
    node_irreps: str
    l_max: int
    num_bessel: int
    r_max: float
    radial_net_nonlinearity: str
    radial_net_n_hidden: int
    radial_net_n_layers: int
    avg_num_neighbors: float
    scalar_mlp_std: float
    use_residual_connection: bool
    nonlinearities: Union[str, dict[str, str]]
    avg_r_min: float
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray]
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray]

    @nn.compact
    def __call__(
        self,
        edge_vectors: jnp.ndarray,
        node_species: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> jnp.ndarray:
        node_irreps = e3nn.Irreps(self.node_irreps)

        # Nodes Embedding
        embedding_irreps = e3nn.Irreps(f"{self.num_species}x0e")
        identity = jnp.eye(self.num_species)
        node_attr = e3nn.IrrepsArray(embedding_irreps, identity[node_species])
        node_feats = Linear(irreps_out=e3nn.Irreps(node_irreps))(node_attr)

        # Edges Embedding
        if hasattr(edge_vectors, "irreps"):
            edge_vectors = edge_vectors.array
        scalar_dr_edge = safe_norm(edge_vectors, axis=-1)

        edge_sh = e3nn.spherical_harmonics(
            e3nn.Irreps.spherical_harmonics(self.l_max),
            edge_vectors,
            normalize=True,
        )

        embedded_dr_edge = RadialEmbeddingBlock(
            r_max=self.r_max,
            avg_r_min=self.avg_r_min,
            basis_functions=self.radial_basis,
            envelope_function=self.radial_envelope,
            num_bessel=self.num_bessel,
        )(scalar_dr_edge)

        # Starting Convolution Layers
        for _ in range(self.num_layers):
            node_feats = NequipLayer(
                node_irreps=node_irreps,
                use_residual_connection=self.use_residual_connection,
                nonlinearities=self.nonlinearities,
                radial_net_nonlinearity=self.radial_net_nonlinearity,
                radial_net_n_hidden=self.radial_net_n_hidden,
                radial_net_n_layers=self.radial_net_n_layers,
                num_bessel=self.num_bessel,
                avg_num_neighbors=self.avg_num_neighbors,
                scalar_mlp_std=self.scalar_mlp_std,
            )(
                node_feats,
                node_attr,
                edge_sh,
                senders,
                receivers,
                embedded_dr_edge.array,
            )

        # output block
        for mul, ir in node_feats.irreps:
            if ir == e3nn.Irrep("0e"):
                mul_second_to_final = mul // 2

        second_to_final_irreps = e3nn.Irreps(f"{mul_second_to_final}x0e")
        final_irreps = e3nn.Irreps("1x0e")

        node_feats = Linear(irreps_out=second_to_final_irreps)(node_feats)
        node_energies = Linear(irreps_out=final_irreps)(node_feats).array
        return jnp.ravel(node_energies)


class NequipLayer(nn.Module):
    """NequIP Convolution.

    Adapted from Google DeepMind materials discovery:
    https://github.com/google-deepmind/materials_discovery/blob/main/model/nequip.py

    Implementation follows the original paper by Batzner et al.

    nature.com/articles/s41467-022-29939-5 and partially
    https://github.com/mir-group/nequip.

    Args:
        node_irreps: representation of hidden/latent node-wise features
        use_residual_connection: use residual connection in network (recommended)
        nonlinearities: nonlinearities to use for even/odd irreps
        radial_net_nonlinearity: nonlinearity to use in radial MLP
        radial_net_n_hidden: number of hidden neurons in radial MLP
        radial_net_n_layers: number of hidden layers for radial MLP
        num_bessel: number of Bessel basis functions to use
        avg_num_neighbors: constant number of per-atom neighbors, used for internal
          normalization
        scalar_mlp_std: standard deviation of weight init of radial MLP

    Returns:
        Updated node features h after the convolution.
    """

    node_irreps: e3nn.Irreps
    use_residual_connection: bool
    nonlinearities: Union[str, dict[str, str]]
    radial_net_nonlinearity: str = "swish"
    radial_net_n_hidden: int = 64
    radial_net_n_layers: int = 2
    num_bessel: int = 8
    avg_num_neighbors: float = 1.0
    scalar_mlp_std: float = 4.0

    @nn.compact
    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,
        node_attrs: e3nn.IrrepsArray,
        edge_sh: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        edge_embedded: jnp.ndarray,
    ) -> e3nn.IrrepsArray:

        irreps_scalars = []
        irreps_nonscalars = []
        irreps_gate_scalars = []

        # get scalar target irreps
        for multiplicity, irrep in self.node_irreps:
            # need the additional Irrep() here for the build, even though irrep is
            # already of type Irrep()
            if e3nn.Irrep(irrep).l == 0 and tp_path_exists(  # noqa E741
                node_feats.irreps, edge_sh.irreps, irrep
            ):
                irreps_scalars += [(multiplicity, irrep)]

        irreps_scalars = e3nn.Irreps(irreps_scalars)

        # get non-scalar target irreps
        for multiplicity, irrep in self.node_irreps:
            # need the additional Irrep() here for the build, even though irrep is
            # already of type Irrep()
            if e3nn.Irrep(irrep).l > 0 and tp_path_exists(
                node_feats.irreps, edge_sh.irreps, irrep
            ):
                irreps_nonscalars += [(multiplicity, irrep)]

        irreps_nonscalars = e3nn.Irreps(irreps_nonscalars)

        # get gate scalar irreps
        if tp_path_exists(node_feats.irreps, edge_sh.irreps, "0e"):
            gate_scalar_irreps_type = "0e"
        else:
            gate_scalar_irreps_type = "0o"

        for multiplicity, _irreps in irreps_nonscalars:
            irreps_gate_scalars += [(multiplicity, gate_scalar_irreps_type)]

        irreps_gate_scalars = e3nn.Irreps(irreps_gate_scalars)

        # final layer output irreps are all three
        # note that this order is assumed by the gate function later, i.e.
        # scalars left, then gate scalar, then non-scalars
        h_out_irreps = irreps_scalars + irreps_gate_scalars + irreps_nonscalars

        if self.use_residual_connection:
            residual_connection = FullyConnectedTensorProduct(
                irreps_out=h_out_irreps,
            )(node_feats, node_attrs)

        # first linear, stays in current h-space
        node_feats = Linear(node_feats.irreps)(node_feats)

        # map node features onto edges for tp
        edge_features = node_feats[senders]

        # gather the instructions for the tp as well as the tp output irreps
        mode = "uvu"
        trainable = "True"
        irreps_after_tp = []
        instructions = []

        for i, (mul_in1, irreps_in1) in enumerate(node_feats.irreps):
            for j, (_, irreps_in2) in enumerate(edge_sh.irreps):
                for curr_irreps_out in irreps_in1 * irreps_in2:
                    if curr_irreps_out in h_out_irreps:
                        k = len(irreps_after_tp)
                        irreps_after_tp += [(mul_in1, curr_irreps_out)]
                        instructions += [(i, j, k, mode, trainable)]

        # sort irreps to be in a l-increasing order
        irreps_after_tp, p, _ = e3nn.Irreps(irreps_after_tp).sort()

        # sort instructions
        sorted_instructions = []

        for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
            sorted_instructions += [
                (
                    irreps_in1,
                    irreps_in2,
                    p[irreps_out],
                    mode,
                    trainable,
                )
            ]

        # TP between spherical harmonics embedding of the edge vector
        tp = FunctionalTensorProduct(
            irreps_in1=edge_features.irreps,
            irreps_in2=edge_sh.irreps,
            irreps_out=irreps_after_tp,
            instructions=sorted_instructions,
        )

        n_tp_weights = 0

        # get output dim of radial MLP / number of TP weights
        for ins in tp.instructions:
            if ins.has_weight:
                n_tp_weights += prod(ins.path_shape)

        mlp = MLP(
            (self.radial_net_n_hidden,) * self.radial_net_n_layers + (n_tp_weights,),
            self.radial_net_nonlinearity,
            use_bias=False,
            scalar_mlp_std=self.scalar_mlp_std,
        )

        # the TP weights (v dimension) are given by the FC
        weight = mlp(edge_embedded)

        edge_features = e3nn.utils.vmap(tp.left_right)(weight, edge_features, edge_sh)

        edge_features = edge_features.astype(node_feats.dtype)

        node_dtype = node_feats.dtype

        edge_feats = edge_features.remove_zero_chunks().simplify()
        # aggregate edge features on nodes
        node_feats = e3nn.scatter_sum(
            edge_feats,
            dst=receivers,
            output_size=node_feats.shape[0],
        )
        node_feats = node_feats.astype(node_dtype)

        # normalize by the average (not local) number of neighbors
        node_feats = node_feats / self.avg_num_neighbors

        # second linear, now we create extra gate scalars by mapping to h-out
        node_feats = Linear(h_out_irreps)(node_feats)

        if self.use_residual_connection:
            node_feats = node_feats + residual_connection

        # gate nonlinearity, applied to gate data, consisting of:
        # a) regular scalars,
        # b) gate scalars, and
        # c) non-scalars to be gated
        # in this order
        gate_fn = functools.partial(
            e3nn.gate,
            even_act=parse_activation(self.nonlinearities["e"]),
            odd_act=parse_activation(self.nonlinearities["o"]),
            even_gate_act=parse_activation(self.nonlinearities["e"]),
            odd_gate_act=parse_activation(self.nonlinearities["o"]),
        )

        node_feats = gate_fn(node_feats)
        node_feats = node_feats.astype(node_dtype)

        return node_feats
