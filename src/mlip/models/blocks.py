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
import logging
from dataclasses import dataclass
from typing import Callable

import e3j
import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from e3j.core.scalar_mixing import ScalarMixing
from e3j.utils.options import Layout
from flax.linen.linear import default_embed_init as default_nn_embed_init
from jax import Array
from jax.nn.initializers import Initializer

from mlip.data import DatasetInfo
from mlip.graph import Graph
from mlip.models.options import (
    Activation,
    GradientScaledKernelInit,
    RadialBasis,
    RadialEnvelope,
    get_layer_initializer_and_scale,
    parse_activation,
    parse_radial_envelope,
)
from mlip.models.radial_embedding import bessel_basis
from mlip.utils.jax_utils import segment_sum

SPECIES_PLACEHOLDER = 0
CHARGE_PLACEHOLDER = 0
# `AtomicEnergiesBlock` overrides ENERGY_PLACEHOLDER to 0.0 at call
# time when JAX_DEBUG_NANS is on, to keep the debugger usable.
ENERGIES_PLACEHOLDER = jnp.nan
# Note: the maximal atomic number 118 occupies the 119-th position in the table,
#       which starts at 0 (dummy atomic number). For JAX to return a place-holder
#       value for OOB errors even when the maximal Z is set, we pad the periodic
#       table to size 120.
PERIODIC_TABLE_SIZE = 120

CHARGE_IDX_OFFSET = 51
CHARGE_TABLE_SIZE = CHARGE_IDX_OFFSET * 2  # 102 : for placeholder value of 0 plus -50
# to 50 = 101 values.

NUM_POINTS_LINSPACE_RADIAL_PREFACTOR = 1000

logger = logging.getLogger("mlip")


@dataclass(frozen=True)
class SpeciesAssignmentBlock:
    """Map atomic numbers to contiguous species indices.

    Builds a compile-time lookup table from the allowed atomic numbers in
    `dataset_info` and uses it to populate `graph.nodes.features["species"]`
    with zero-based species indices.  Atomic numbers not present in the dataset
    are mapped to `SPECIES_PLACEHOLDER`.

    Attributes:
        dataset_info: The model's dataset_info whose `allowed_atomic_numbers`
                      defines the supported species.
    """

    dataset_info: DatasetInfo

    def __call__(self, graph: Graph) -> Graph:
        """Update graph.nodes.features['species']."""
        with jax.ensure_compile_time_eval():
            allowed_atomic_numbers = jnp.array(
                sorted(self.dataset_info.allowed_atomic_numbers)
            )
            # Non-available atomic numbers map to SPECIES_PLACEHOLDER.
            lookup_table = jnp.full(
                PERIODIC_TABLE_SIZE, SPECIES_PLACEHOLDER, dtype=jnp.int32
            )
            lookup_table = lookup_table.at[allowed_atomic_numbers].set(
                jnp.arange(allowed_atomic_numbers.size)
            )

        species = lookup_table[graph.nodes.atomic_numbers]
        return graph.update_node_features(species=species)


@dataclass(frozen=True)
class ChargeIndexAssignmentBlock:
    """Map charge values to contiguous charge indices.

    Builds a compile-time lookup table from the allowed charge values in
    `dataset_info` and uses it to populate `graph.nodes.features["charge_indices"]`
    with zero-based charge indices.  Charge values not present in the dataset
    are mapped to `CHARGE_PLACEHOLDER`.

    Attributes:
        dataset_info: The model's dataset_info whose `available_total_charges`
                      defines the supported charge values.
    """

    dataset_info: DatasetInfo

    def __call__(self, graph: Graph) -> Graph:
        """Update graph.nodes.features['charge_indices']."""
        if graph.globals.charge is None:
            raise ValueError(
                "Model was configured with `use_total_charge_embedding=True` but "
                "`graph.globals.charge` is None. Set the total charge on input "
                "structures, or pass `set_none_charges_to_zero` where available."
            )
        with jax.ensure_compile_time_eval():
            available_total_charges = jnp.array(
                list(sorted(self.dataset_info.available_total_charges))
            )
            if jnp.any(available_total_charges > 50) or jnp.any(
                available_total_charges < -50
            ):
                raise ValueError(
                    "Total charge indexing is only supported for charges"
                    " between -50 and 50."
                )
            available_charge_indices = available_total_charges + CHARGE_IDX_OFFSET
            # Non-available charge indices map to CHARGE_PLACEHOLDER.
            lookup_table = jnp.full(
                CHARGE_TABLE_SIZE, CHARGE_PLACEHOLDER, dtype=jnp.int32
            )
            lookup_table = lookup_table.at[
                available_charge_indices.astype(jnp.int32)
            ].set(jnp.arange(available_charge_indices.size) + 1)

        expanded_charge = jnp.repeat(
            graph.globals.charge,
            graph.n_node,
            total_repeat_length=graph.nodes.positions.shape[0],
        )
        expanded_charge_indices = expanded_charge + CHARGE_IDX_OFFSET
        charge_indices = lookup_table[expanded_charge_indices.astype(jnp.int32)]

        def raise_error_for_unseen_charges(charge_indices: jax.Array) -> None:
            unseen_charges = np.sum(
                jnp.where(charge_indices == CHARGE_PLACEHOLDER, 1, 0)
            )
            if unseen_charges > 0:
                raise ValueError(
                    "Some charge indices are not present in the dataset."
                    " Try using the `ensure_no_unseen_total_charges` option "
                    "in the dataset builder config."
                )

        jax.debug.callback(raise_error_for_unseen_charges, charge_indices)
        return graph.update_node_features(charge_indices=charge_indices)


class AtomicEnergiesBlock(nn.Module):
    """Add atomic energies to latent node energy summands.

    Typical atomic contributions (usually, the energy of core electrons) are
    initialized from the `dataset_info`. By default, they are not learnable.
    If the `scaling_mean` and `scaling_stdev` attributes are set in dataset_info,
    latent node features will also be shifted and rescaled prior to the addition
    of atomic energies.

    Attributes:
        dataset_info: The model's dataset_info containing the dictionary of
                      atomic energies.
        learnable: Whether to allow atomic energies to be learned, the default
                   is false.
        skip_atomic_energies_addition: Whether the atomic energies addition should
                                       be skipped and thus only the shifting/scaling is
                                       applied. Default is `False`.
    """

    dataset_info: DatasetInfo
    learnable: bool = False
    skip_atomic_energies_addition: bool = False

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        """Update graph.nodes.features['energy']."""

        placeholder = 0.0 if jax.config.jax_debug_nans else ENERGIES_PLACEHOLDER

        def _build_e0_row(single_map: dict[int, float]) -> jnp.ndarray:
            allowed_z = jnp.array(list(single_map.keys()))
            e0s = jnp.array(list(single_map.values()))
            row = jnp.full(PERIODIC_TABLE_SIZE, placeholder)
            row = row.at[allowed_z].set(e0s)
            row = row.at[0].set(0.0)
            return row

        e0s_map = self.dataset_info.atomic_energies_map
        multiple_datasets: bool = isinstance(e0s_map, list)
        with jax.ensure_compile_time_eval():
            if multiple_datasets:
                atomic_energies_lookup = jnp.stack([
                    _build_e0_row(single_map) for single_map in e0s_map
                ])
            else:
                atomic_energies_lookup = _build_e0_row(e0s_map)

        if self.learnable:
            atomic_energies_lookup = self.param(
                "atomic_energies",
                lambda _: atomic_energies_lookup,
            )

        atomic_numbers = graph.nodes.atomic_numbers
        node_energies = graph.nodes.features.get("energy")

        if node_energies is None:
            raise KeyError("AtomicEnergiesBlock requires an 'energy' node feature")

        # Expand per-graph dataset_idx to per-node for multi-dataset indexing.
        # When `dataset_idx` is not set on the graph (e.g. single-dataset
        # fine-tune on a multi-dataset pretrained model), default to 0 so the
        # first dataset's E0s is applied — matching `select_head`'s fallback.
        if multiple_datasets:
            dataset_idx = graph.globals.dataset_idx
            total_num_nodes = atomic_numbers.shape[0]
            if dataset_idx is None:
                node_dataset_idx = jnp.zeros(total_num_nodes, dtype=jnp.int32)
            else:
                node_dataset_idx = jnp.repeat(
                    dataset_idx, graph.n_node, total_repeat_length=total_num_nodes
                )

        # Optionally, shift and rescale latent node energy summands
        stdev = self.dataset_info.scaling_stdev
        mean = self.dataset_info.scaling_mean
        if (stdev, mean) != (1.0, 0.0):
            node_energies *= stdev
            node_energies += mean

        # Then add atomic energies and update graph
        if not self.skip_atomic_energies_addition:
            if multiple_datasets:
                node_energies += atomic_energies_lookup[
                    node_dataset_idx, atomic_numbers
                ]
            else:
                node_energies += atomic_energies_lookup[atomic_numbers]
        return graph.update_node_features(energy=node_energies)


class RadialEmbeddingBlock(nn.Module):
    """Transforms distances into feature vectors using radial basis functions (RBFs).

    Attributes:
        radial_basis: Type of radial basis functions to use
                   (e.g., Gaussian smearing, ExpNormal smearing, or Bessel).
        num_rbf: Number of radial basis functions.
        graph_cutoff_angstrom: Cutoff distance beyond which interactions are ignored
                               or smoothly suppressed.
        learnable: If True, the parameters of the radial basis functions are learnable.
                   Note that "Bessel" RBF type is not learnable.
        radial_envelope: Optional envelope function applied to the radial embeddings to
                         enforce smooth cutoff behavior. If None, no additional envelope
                         is applied, which is the default.
        avg_r_min: Optional minimum average distance used for normalization or
                   scaling of the radial features.
        return_as_irreps: If True, returns the embedding formatted as
                          `e3nn_jax.IrrepsArray` type. Default is False.
        basis_width_scalar: Only used in Gaussian smearing, scaling factor applied to
                            the width of the radial basis functions.
    """

    radial_basis: RadialBasis
    num_rbf: int
    graph_cutoff_angstrom: float
    learnable: bool
    radial_envelope: RadialEnvelope | None = None
    avg_r_min: float | None = None
    return_as_irreps: bool = False
    basis_width_scalar: float = 1.0
    cosine_cutoff: bool = True

    def setup(self) -> None:
        """Setup function."""
        if self.radial_basis == RadialBasis.EXPNORM:
            self.embed_fn = self._get_expnorm_embed_fn()
        elif self.radial_basis == RadialBasis.GAUSS:
            self.embed_fn = self._get_gauss_embed_fn()
        elif self.radial_basis == RadialBasis.BESSEL:
            self.embed_fn = self._get_bessel_embed_fn()
        else:
            raise ValueError("Given RBF type is not available.")

        self.pre_factor = 1.0
        if self.avg_r_min is not None:
            self.pre_factor = self._compute_pre_factor()

    def __call__(self, distances: jax.Array) -> jax.Array:
        """Call function for the radial embedding block.

        Args:
            distances: The distances for all the edges, i.e., length of the
                       edge vectors.

        Returns:
            The radial embeddings for each edge.
        """
        embedding = self.pre_factor * jnp.where(
            (distances == 0.0)[:, None], 0.0, self.embed_fn(distances)
        )  # [n_edges, num_rbf]

        if self.return_as_irreps:
            return e3nn.IrrepsArray(f"{embedding.shape[-1]}x0e", embedding)
        return embedding

    def _get_gauss_embed_fn(self) -> Callable[[jax.Array], jax.Array]:
        """Core embedding function for the Gaussian smearing method."""
        if self.num_rbf < 2:
            raise ValueError("For Gaussian Smearing, 'num_rbf' must be >= 2.")
        offset = jnp.linspace(0, self.graph_cutoff_angstrom, self.num_rbf)
        coeff = -0.5 / (self.basis_width_scalar * (offset[1] - offset[0]) + 1e-12) ** 2
        if self.learnable:
            self.offset = self.param(
                "offset", nn.initializers.constant(offset), (self.num_rbf,)
            )
            self.coeff = self.param("coeff", nn.initializers.constant(coeff), ())
        else:
            self.offset = offset
            self.coeff = coeff

        self.cutoff_fn = (
            functools.partial(
                parse_radial_envelope(self.radial_envelope),
                graph_cutoff_angstrom=self.graph_cutoff_angstrom,
            )
            if self.radial_envelope
            else None
        )

        def _embed_fn(distances: jax.Array) -> jax.Array:
            dist = distances[..., jnp.newaxis] - self.offset
            if self.cutoff_fn is not None:
                cutoffs = self.cutoff_fn(dist)
            else:
                cutoffs = 1.0
            return cutoffs * jnp.exp(self.coeff * jnp.square(dist))

        return _embed_fn

    def _get_expnorm_embed_fn(self) -> Callable[[jax.Array], jax.Array]:
        """Core embedding function for the ExpNormal smearing method."""
        start_value = jnp.exp(-self.graph_cutoff_angstrom)
        means = jnp.linspace(start_value, 1, self.num_rbf)
        betas = jnp.full((self.num_rbf,), (2 / self.num_rbf * (1 - start_value)) ** -2)
        if self.learnable:
            self.means = self.param(
                "means", nn.initializers.constant(means), (self.num_rbf,)
            )
            self.betas = self.param(
                "betas", nn.initializers.constant(betas), (self.num_rbf,)
            )
        else:
            self.means = means
            self.betas = betas

        self.alpha = 5.0 / self.graph_cutoff_angstrom

        radial_envelope = self.radial_envelope
        if radial_envelope is None:
            radial_envelope = RadialEnvelope.COSINE_CUTOFF
            logger.warning(
                "Option EXPNORM requires a radial envelope, but none was passed. "
                "Falling back to using cosine cutoff."
            )

        self.cutoff_fn = functools.partial(
            parse_radial_envelope(radial_envelope),
            graph_cutoff_angstrom=self.graph_cutoff_angstrom,
        )

        def _embed_fn(distances: jax.Array) -> jax.Array:
            dist = distances[..., jnp.newaxis]
            cutoffs = self.cutoff_fn(dist)
            return cutoffs * jnp.exp(
                (-1 * self.betas) * (jnp.exp(self.alpha * (-dist)) - self.means) ** 2
            )

        return _embed_fn

    def _get_bessel_embed_fn(self) -> Callable[[jax.Array], jax.Array]:
        """Core embedding function for the Bessel functions embedding."""
        if self.learnable:
            logger.warning(
                "Radial embedding was set to be learnable, but with option BESSEL it "
                "is not implemented to be learnable. Therefore, 'learnable=True' "
                "has no effect."
            )

        radial_envelope = self.radial_envelope
        if radial_envelope is None:
            radial_envelope = RadialEnvelope.POLYNOMIAL
            logger.warning(
                "Option BESSEL requires a radial envelope, but none was passed. "
                "Falling back to using polynomial envelope."
            )

        _radial_envelope = parse_radial_envelope(radial_envelope)

        def _embed_fn(distances: jax.Array) -> jax.Array:
            basis = bessel_basis(
                distances,
                self.graph_cutoff_angstrom,
                self.num_rbf,
            )  # [n_edges, num_rbf]
            cutoff = _radial_envelope(distances, self.graph_cutoff_angstrom)
            return basis * cutoff[:, None]  # [n_edges, num_rbf]

        return _embed_fn

    def _compute_pre_factor(self) -> float:
        with jax.ensure_compile_time_eval():
            samples = jnp.linspace(
                self.avg_r_min,
                self.graph_cutoff_angstrom,
                NUM_POINTS_LINSPACE_RADIAL_PREFACTOR,
                dtype=jnp.float32,
            )
            factor = jnp.mean(self.embed_fn(samples) ** 2).item() ** -0.5
        return factor


# NOTE: dataclass not nn.Module, to make easier conversion to e3j.
@dataclass(frozen=True)
class SphericalHarmonicsBlock:
    """Spherical harmonics encoding of edge vectors."""

    l_max: int
    normalize: bool = True
    normalization: str | None = None

    def __call__(self, edge_vectors: Array) -> e3nn.IrrepsArray:
        spherical_embedding = e3nn.spherical_harmonics(
            e3nn.Irreps.spherical_harmonics(self.l_max),
            edge_vectors,
            normalize=self.normalize,
            normalization=self.normalization,
        )
        return spherical_embedding


class NodeEmbeddingBlock(nn.Module):
    """Species embedding using a learned lookup table.

    Maps integer species indices to dense embedding vectors via `nn.Embed`.
    Returns a plain jax array. Models that require an `e3nn.IrrepsArray`
    should wrap the output themselves.

    Attributes:
        num_species: Number of distinct atomic species.
        num_channels: Embedding dimension.
        kernel_init: Initializer type for the embedding. Can be a
            `GradientScaledKernelInit` or a callable initializer. Default
            is the default init used by `nn.Embed`.
    """

    num_species: int
    num_channels: int
    kernel_init: Initializer | GradientScaledKernelInit = default_nn_embed_init

    def setup(self):
        initializer, scale = get_layer_initializer_and_scale(
            self.num_species,
            self.kernel_init,
        )

        self.embed = nn.Embed(
            num_embeddings=self.num_species,
            features=self.num_channels,
            embedding_init=initializer,
        )
        self.scale = scale

    @nn.compact
    def __call__(self, node_species: jnp.ndarray) -> jnp.ndarray:
        embed = self.embed(node_species)

        if self.scale != 1.0:
            embed *= self.scale

        return embed


class JointNodeEmbeddingBlock(nn.Module):
    """Joint node embedding block for species and total charge.

    Creates embeddings of size `num_channels` for the specie and total charge associated
    with each node that are concatenated :

    [n_nodes] -> [n_nodes, num_channels] -> [n_nodes, num_channels + num_channels]

    Those embeddings are then passed through a linear projection outputting an embedding
    of size `num_channels` :

    [n_nodes, num_channels + num_channels] -> [n_nodes, num_channels]

    The output is then passed through an activation function `activation_fn`.
    """

    num_species: int
    num_charge: int
    num_channels: int
    activation_fn: Callable
    kernel_init: Initializer | GradientScaledKernelInit = default_nn_embed_init

    def setup(self) -> None:
        self.species_embedder = NodeEmbeddingBlock(
            num_species=self.num_species,
            num_channels=self.num_channels,
            kernel_init=self.kernel_init,
        )
        self.charge_embedder = NodeEmbeddingBlock(
            num_species=self.num_charge,
            num_channels=self.num_channels,
            kernel_init=self.kernel_init,
        )
        self.joint_embedding_projection = nn.Dense(
            features=self.num_channels, use_bias=False
        )
        self.embedding_activation = self.activation_fn

    def __call__(
        self,
        node_species: jnp.ndarray,
        charge_indices: jnp.ndarray,
    ) -> jnp.ndarray:
        node_species_embedding = self.species_embedder(node_species)
        node_charge_embedding = self.charge_embedder(charge_indices)
        node_embedding = jnp.concatenate(
            [node_species_embedding, node_charge_embedding], axis=-1
        )
        node_embedding = self.joint_embedding_projection(node_embedding)
        node_embedding = self.embedding_activation(node_embedding)
        return node_embedding


class BetaSwish(nn.Module):
    """Swish (SiLU) activation with a learnable per-channel beta parameter.

    Computes `x * sigmoid(beta * x)`, where `beta` is initialised to 1
    (recovering standard Swish) and learned during training.
    """

    @nn.compact
    def __call__(self, x: Array) -> Array:
        beta = self.param("Beta", nn.initializers.ones, (x.shape[-1],))
        return x * nn.sigmoid(beta * x)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for scalar features, e.g. radial embeddings.

    Note that the activation function is only applied between hidden layers, not
    after the output layer.

    Attributes:
        layer_sizes: List of dimensions for each layer. Must include input and output
            dimensions. For example, [4, 16, 32] has 2 layers with input dimension 4.
        activation: Activation function to apply between layers.
        kernel_init: Initializer type for the hidden layers. Can be a
            `GradientScaledKernelInit` or a callable initializer.
            Default is `lecun_normal`, to match default `nn.Dense` initializer.
        use_bias: Whether to include bias parameters in all layers.
        use_layer_norm: If true, use LayerNorm in each hidden layer.
        normalize_activation: If true, wrap activation with e3nn.normalize_function.
        output_kernel_init: Initializer for the output layer. Default is `kernel_init`.
    """

    layer_sizes: list[int]
    activation: Activation | Callable[[Array], Array] | str
    kernel_init: Initializer | GradientScaledKernelInit
    use_bias: bool = False
    use_layer_norm: bool = False
    normalize_activation: bool = False
    output_kernel_init: Initializer | GradientScaledKernelInit | None = None

    def _get_activation(self):
        if self.activation is None:
            return None

        if callable(self.activation):
            act = self.activation
        elif self.activation == "beta_swish":
            act = BetaSwish()
        else:
            act = parse_activation(self.activation)

        if self.normalize_activation:
            act = e3nn.normalize_function(act)

        return act

    def setup(self):
        sizes = self.layer_sizes
        layers: list[nn.Dense] = []
        scales: list[float] = []
        for i, (dim_in, dim_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            is_output = i == len(sizes) - 2

            kernel_init = (
                self.output_kernel_init
                if is_output and self.output_kernel_init is not None
                else self.kernel_init
            )
            initializer, scale = get_layer_initializer_and_scale(dim_in, kernel_init)

            dense = nn.Dense(
                features=dim_out,
                kernel_init=initializer,
                use_bias=self.use_bias,
                bias_init=nn.initializers.zeros,
            )
            layers.append(dense)
            scales.append(scale)

        self.layers = layers
        self.scales = scales

        self.layer_norms = (
            [nn.LayerNorm(epsilon=1e-12) for _ in layers[:-1]]
            if self.use_layer_norm
            else []
        )
        self.activations = [self._get_activation() for _ in layers[:-1]]

    def __call__(self, x: Array) -> Array:
        depth = len(self.layers) - 1
        for i, (layer, scale) in enumerate(zip(self.layers, self.scales)):
            # Apply i-th linear layer
            x = layer(x)

            # Rescale output when `gradient_scale = True`.
            # Applied after Dense so that bias is also scaled (matches e3nn convention).
            if scale != 1.0:
                x *= scale

            if self.use_layer_norm and i < depth:
                x = self.layer_norms[i](x)

            # Apply activation (has weights in case of 'beta_swish').
            if i < depth and self.activations[i] is not None:
                x = self.activations[i](x)
        return x


@dataclass
class SO3Convolution:
    """Equivariant message passing layer using tensor products.

    This module updates node features by applying the following steps:

    1. Gather sender node features,
    2. Compute tensor product of sender node features with spherical
       embeddings of edge vectors to form messages,
    3. Mix messages with radial embeddings of edges -- typically obtained
       from a layer-dependent MLP representation of RBF encodings,
    4. Sum messages on target nodes.

    Note:
        This block will later be moved into e3j in order to provide fused
        implementation for better performance.

    Attributes:
        source_irreps: Irreps of the two tensor product inputs (node features and
            spherical embeddings).
        target_irreps: Output irreps of the tensor product.
        avg_num_neighbors: Average number of neighbors per node, used to normalize
            the aggregated messages.
        layout: Memory layout for irreps arrays.
        deterministic_scatter_ops: Whether to use deterministic scatter operations in
            the message passing convolution. If True, uses a slower but deterministic
            alternative to standard threaded scatter operations.
    """

    source_irreps: tuple[e3nn.Irreps, e3nn.Irreps]
    target_irreps: e3nn.Irreps
    avg_num_neighbors: float
    layout: Layout = Layout.TRAILING_CHANNELS
    deterministic_scatter_ops: bool = False

    def __post_init__(self):
        self.layout = Layout.parse(self.layout)

    @property
    def target(self) -> e3nn.Irreps:
        return e3nn.tensor_product(
            *self.source_irreps, filter_ir_out=self.target_irreps
        )

    def __call__(
        self,
        node_feats: Array,
        spherical_embedding: Array,
        edge_scalars: Array,
        senders: Array,
        receivers: Array,
    ):
        # Setup
        nodes, edges = self.source_irreps
        tensor_product = e3j.core.TensorProduct(
            source=(str(nodes), str(edges)),
            target=self.target_irreps,
            layout=self.layout,
            normalization="SQRT_DIM_OUT",
        )
        scalar_mixing = ScalarMixing(source=tensor_product.target, layout=self.layout)

        # Evaluate: gather, tensor_product, scalar_mixing, scatter_add
        sender_feats = node_feats[senders]
        messages = tensor_product(sender_feats, spherical_embedding)
        messages = scalar_mixing(edge_scalars, messages)

        # Scatter-add messages to receiver nodes
        receiver_feats = segment_sum(
            messages,
            receivers,
            num_segments=node_feats.shape[0],
            deterministic=self.deterministic_scatter_ops,
        )  # [n_nodes, irreps_dim]

        receiver_feats /= self.avg_num_neighbors
        return receiver_feats


class MaskPaddedNodeOutputsBlock(nn.Module):
    """Zero out the listed node features on padded (dummy) nodes.

    Padded graphs carry dummy nodes that must not contribute to downstream
    quantities (energy, partial charges, ...). This block applies
    `jnp.where(graph.node_mask(), feature, 0.0)` for each name in
    `feature_names`. Centralises a pattern previously inlined at the end of
    each model's `__call__`.

    Attributes:
        feature_names: Tuple of node feature keys to mask.
    """

    feature_names: tuple[str, ...]

    @nn.compact
    def __call__(self, graph: Graph) -> Graph:
        node_mask = graph.node_mask()
        updates = {
            name: jnp.where(node_mask, graph.nodes.features[name], 0.0)
            for name in self.feature_names
        }
        return graph.update_node_features(**updates)
