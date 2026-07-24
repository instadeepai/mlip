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

import flax.linen as nn
import jax.numpy as jnp

from mlip.models.esen.network import Esen
from mlip.models.mace.network import Mace
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.nequip.network import Nequip
from mlip.models.visnet.network import Visnet
from mlip.models.visnet.visnet_helpers import VecNormType
from mlip.simulation.fep.alchemical_graph import AlchemicalGraph
from mlip.simulation.fep.alchemical_graph.masking import get_alchemical_edge_mask


class AlchemicalMLIPNetwork(nn.Module):
    """Mixin to apply the alchemical edge weight to an MLIPNetwork's equations.

    Computes the per-edge scale factor using `alchemical_lambda` and
    `alchemical_atom_indices`, and stores it as the `edge_scale` edge feature
    on both the short-range and (if present) long-range edges, then passes
    to the parent module's call method.

        alpha_ij = edge_weight    if i and j are on opposite sides of the boundary
        alpha_ij = 1              otherwise

    Classes that inherit must also inherit from an `MLIPNetwork` that applies the
    `edge_scale` feature in the model equations to weight alchemical edges.
    """

    def __call__(self, graph: AlchemicalGraph) -> AlchemicalGraph:
        if graph.globals.alchemical_lambda.shape[0] > 2:
            raise ValueError(
                "AlchemicalMLIPNetwork assumes a single real subgraph; got "
                f"alchemical_lambda.shape={graph.globals.alchemical_lambda.shape}."
            )
        edge_weight = graph.globals.alchemical_lambda[0, 0]
        boundary_mask, lr_boundary_mask = get_alchemical_edge_mask(graph)

        edge_scale = jnp.where(boundary_mask, edge_weight, 1.0)
        graph = graph.update_edge_features(edge_scale=edge_scale)

        if graph.edges_long_range is not None:
            lr_edge_scale = jnp.where(lr_boundary_mask, edge_weight, 1.0)
            graph = graph.update_long_range_edge_features(edge_scale=lr_edge_scale)

        return super().__call__(graph)


class AlchemicalMace(AlchemicalMLIPNetwork, Mace):
    """MACE variant with the alchemical edge weight applied to its equations.

    Implements the edge-scaling scheme described by both (arXiv:2404.10746,
    arXiv:2405.18171).

    The edge weight scales the per-edge messages of edges that cross the alchemical
    boundary inside the message-passing convolution in each layer. See
    `AlchemicalMLIPNetwork` for the shared mechanism.
    """


class AlchemicalNequip(AlchemicalMLIPNetwork, Nequip):
    """NequIP variant with the alchemical edge weight applied to its equations.

    The edge weight scales the per-edge messages of edges that cross the alchemical
    boundary inside the message-passing convolution in each layer. See
    `AlchemicalMLIPNetwork` for the shared mechanism.
    """


class AlchemicalVisnet(AlchemicalMLIPNetwork, Visnet):
    """ViSNet variant with the alchemical edge weight applied to its equations.

    The edge weight scales the per-edge messages of edges that cross the
    alchemical boundary, before every neighbour-aggregation step in each layer.
    See `AlchemicalMLIPNetwork` for the shared mechanism.
    """

    def setup(self) -> None:
        """Initializes model layers.

        Raises:
            ValueError: If `config.vecnorm_type` is `VecNormType.MAX_MIN`,
                as this results in a non-smooth alchemical path.
        """
        if VecNormType(self.config.vecnorm_type) == VecNormType.MAX_MIN:
            raise ValueError(
                "Cannot create an `AlchemicalVisnet` from a Visnet model using "
                "`vecnorm_type='max_min'`, as this results in a non-smooth alchemical "
                "path. Select a different model, or set `use_alchemical_mlip=False` "
                "in the `FEPSimulationSampler` config. Note that "
                "`vecnorm_type='none'` was used when testing `AlchemicalVisnet`, so "
                "performance may degrade for other settings."
            )
        super().setup()


class AlchemicalEsen(AlchemicalMLIPNetwork, Esen):
    """Esen variant with the alchemical edge weight applied to its equations.

    The edge weight scales the per-edge messages of edges that cross the
    alchemical boundary, before every neighbour-aggregation step in each layer.
    See `AlchemicalMLIPNetwork` for the shared mechanism.
    """


# Maps a plain `MLIPNetwork` class to its alchemical-edge-weighted counterpart.
_ALCHEMICAL_MLIP_NETWORKS: dict[type[MLIPNetwork], type[AlchemicalMLIPNetwork]] = {
    Mace: AlchemicalMace,
    Nequip: AlchemicalNequip,
    Visnet: AlchemicalVisnet,
    Esen: AlchemicalEsen,
}


def to_alchemical_mlip_network(mlip_network: MLIPNetwork) -> AlchemicalMLIPNetwork:
    """Convert an `MLIPNetwork` instance into its alchemical-edge-weighted variant.

    Alchemical variants add no new parameters, so the original model's params are used.

     Args:
         mlip_network: The plain MLIP network to convert. If already an
             `AlchemicalMLIPNetwork`, it is returned unchanged.

     Returns:
         The corresponding `AlchemicalMLIPNetwork` instance.

     Raises:
         ValueError: If no alchemical variant is registered for `type(mlip_network)`.
    """
    if isinstance(mlip_network, AlchemicalMLIPNetwork):
        return mlip_network

    alchemical_cls = _ALCHEMICAL_MLIP_NETWORKS.get(type(mlip_network))
    if alchemical_cls is None:
        raise ValueError(
            f"No alchemical variant registered for {type(mlip_network).__name__}. "
            f"Supported types: "
            f"{[cls.__name__ for cls in _ALCHEMICAL_MLIP_NETWORKS]}."
        )
    return alchemical_cls(
        config=mlip_network.config, dataset_info=mlip_network.dataset_info
    )
