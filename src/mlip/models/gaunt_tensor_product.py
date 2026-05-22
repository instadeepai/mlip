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

from typing import Callable, Literal

import e3j
import e3nn_jax as e3nn
import flax.linen as nn
import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
from e3j.linen.linear_indexwise import LinearIndexwise

from mlip.graph import Graph
from mlip.models import options
from mlip.models.blocks import MLP, BetaSwish
from mlip.utils.jax_utils import segment_sum
from mlip.utils.spherical_designs.spherical_designs import get_spherical_design


def resolve_use_float64(use_float64: bool | None) -> bool:
    """Pick float64 vs float32 for VSTP precomputations.

    If `use_float64` is `None`, use JAX's `jax_enable_x64` setting so design
    points and cached harmonics match the default floating dtype (`float64` when
    x64 is enabled, else `float32`).
    """
    if use_float64 is not None:
        return use_float64
    return bool(jax.config.read("jax_enable_x64"))


def get_design_points(
    sources: tuple[e3nn.Irreps, ...],
    target: e3nn.Irreps,
    use_float64: bool | None = None,
) -> jnp.ndarray:
    """Get spherical harmonics evaluated on the points
    of a spherical design of order t, where t is the sum
    of the maximum orders of the sources and target representations.
    Args:
        sources: Sources irreps.
        target: Target irreps.
        use_float64: If `True` / `False`, load the design in float64 / float32.
        If `None`, follow :func:`resolve_use_float64` (JAX `jax_enable_x64`).
    Returns:
        design_points: Design points as a numpy array of shape (n_points, 3).
    """
    max_orders = [max(target.ls)] + [max(source.ls) for source in sources]
    max_sum_order = sum(max_orders)
    design_vectors = get_spherical_design(
        max_sum_order, use_float64=resolve_use_float64(use_float64)
    )
    return design_vectors


def get_harmonics_on_points(irreps: e3nn.Irreps, points: jnp.ndarray) -> jnp.ndarray:
    """Get spherical harmonics evaluated on the input points.
    Args:
        irreps: Irreps of the spherical harmonics to evaluate on.
        points: Points as a numpy array of shape (irreps.dim, n_points).
    Returns:
    """
    harmonics = e3nn.spherical_harmonics(
        irreps, points, normalize=True, normalization="norm"
    ).array
    return harmonics


class GauntTensorProduct:
    r"""Bilinear O(3)-equivariant Gaunt Tensor Product.

    The Gaunt tensor product of two irreps arrays :math:`x_1` and :math:`x_2`
    is defined as:

    .. math::

    (x_1 \otimes_{\mathrm{Gaunt}} x_2)^{\ell m} =
    \int_{\mathbb{S}^2} F_{x_1}(\hat{r}) F_{x_2}(\hat{r}) Y^{\ell m}(\hat{r}) d\hat{r}

    where $F_{x}(\hat{r})$ is the signal on the sphere associated with $x$,
    which is defined as:

    .. math::

    F_{x}(\hat{r}) = \sum_{\ell}\sum_{m=-\ell}^{\ell} x^{\ell m} Y^{\ell m}(\hat{r}).

    See `Luo et al. (2024)<https://openreview.net/forum?id=mhyQXJ6JsK>`_
    for more details.

    Our implementation uses a spherical design to evaluate the integrals.
    Although this method has a scales as \mathcal{O}(L^4), we found it
    is the fastest for values of $\ell_{\mathrm{max}}$ up to ~30.
    """

    def __init__(
        self,
        sources: tuple[str | e3nn.Irreps, str | e3nn.Irreps],
        target: str | None = None,
        lmax: int | None = None,
        use_float64: bool | None = None,
    ):
        """
        Initialize a GTP.

        Args:
            sources: tuple of representations of the sources.
            target: Output representation. If `None`, defaults to the tensor product
                of the sources (natural parity paths only).
            lmax: Maximum order of the output. Only used when `target` is `None`.
            use_float64: Dtype for design points and cached harmonics. `None` follows
                JAX `jax_enable_x64` via :func:`resolve_use_float64`.

        Returns:
            The initialized instance of the GauntTensorProduct.
        """
        # --- I/O irreps ---
        assert len(sources) == 2, (
            "Gaunt tensor product expects a binary tuple as source argument."
        )

        self.sources = (e3nn.Irreps(sources[0]), e3nn.Irreps(sources[1]))

        # Gaunt tensor product only accept natural irreps of type (l, (-1)**l) for now.
        assert all(ir.ir.p == (-1) ** (ir.ir.l) for ir in self.sources[0]), (
            f"Gaunt tensor product only accepts natural irreps\
            of type (l, (-1)**l) for now, got sources[0] = {self.sources[0]}."
        )

        assert all(ir.ir.p == (-1) ** (ir.ir.l) for ir in self.sources[1]), (
            "GTP only accepts natural irreps of type (l, (-1)**l) for now,"
            f" got sources[1] = {self.sources[1]}."
        )

        if target is None:
            tp = e3nn.tensor_product(*self.sources).filter(lmax=lmax)
            # Gaunt paths only contribute on natural (l, (-1)^l) irreps for SH inputs.
            self.target = e3nn.Irreps([
                (mul, ir) for mul, ir in tp if ir.p == (-1) ** ir.l
            ])
        else:
            self.target = e3nn.Irreps(target)

        assert all(ir.ir.p == (-1) ** (ir.ir.l) for ir in self.target), (
            "GTP only accepts natural irreps of type (l, (-1)**l) for now,"
            f" got target = {self.target}."
        )

        design_points = get_design_points(
            self.sources, self.target, use_float64=use_float64
        )
        self.num_design_points = design_points.shape[0]

        # --- Get harmonics terms on design points ---
        self.harmonics_dict = {
            "src_1": get_harmonics_on_points(self.sources[0], design_points),
            "src_2": get_harmonics_on_points(self.sources[1], design_points),
            "target": get_harmonics_on_points(self.target, design_points),
        }

    def __call__(
        self, x_1: e3nn.IrrepsArray, x_2: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:
        """Evaluate Gaunt Tensor Product on two inputs.

        Args:
            x_1: array of input features with leading channel axis
                shape `(*batch_dims, src_1.dim)`
            x_2: array of input features with leading channel axis
                shape `(*batch_dims, src_2.dim)`
        Returns:
            array of output features with leading channel axis
            shape `(*batch_dims, target.dim)`
        """
        assert x_1.shape[:-1] == x_2.shape[:-1], (
            "Inputs must have the same batch dimensions, got {} and {}".format(
                x_1.shape[:-1], x_2.shape[:-1]
            )
        )
        y_1 = x_1.array @ self.harmonics_dict["src_1"].T
        y_2 = x_2.array @ self.harmonics_dict["src_2"].T
        y = y_1 * y_2
        x = y @ self.harmonics_dict["target"] / self.num_design_points
        x = e3nn.IrrepsArray(self.target, x)
        return x


class GauntSymmetricContraction(nn.Module):
    """Symmetric contraction using the Gaunt Tensor Product."""

    source_irreps: str
    keep_irrep_out: str
    correlation: int
    num_species: int
    num_channels: int
    use_float64: bool | None = None

    def setup(self):

        source_irreps = e3nn.Irreps(self.source_irreps)
        # Gaunt tensor product only accept natural irreps of type (l, (-1)**l) for now.
        assert all(ir.ir.p == (-1) ** (ir.ir.l) for ir in source_irreps), (
            f"Gaunt tensor product only accepts natural irreps\
            of type (l, (-1)**l) for now, got source_irreps = {source_irreps}."
        )
        keep_irrep_out = e3nn.Irreps(self.keep_irrep_out)

        target_irreps = source_irreps

        for _ in range(self.correlation):
            target_irreps: e3nn.Irreps = e3nn.tensor_product(
                target_irreps, target_irreps
            ).filter(keep=keep_irrep_out)
            # Filter out pseudo-tensors and set mul to 1
            target_irreps = e3nn.Irreps([
                (1, ir) for mul, ir in target_irreps if ir.p == (-1) ** ir.l
            ])

        design_points = get_design_points(
            [source_irreps] * self.correlation,
            target_irreps,
            use_float64=self.use_float64,
        )
        self.num_design_points = design_points.shape[0]

        # --- Get harmonics terms on design points ---
        self.harmonics_src = get_harmonics_on_points(source_irreps, design_points)
        self.harmonics_target = get_harmonics_on_points(target_irreps, design_points)

        self.linear_in = [
            e3nn.flax.Linear(self.num_channels * source_irreps)
            for _ in range(self.correlation)
        ]
        self.gtp_out_irreps = target_irreps
        self.linear_out = LinearIndexwise(
            target_irreps,
            target_irreps,
            num_indices=self.num_species,
            num_channels=self.num_channels,
            layout="LEADING_CHANNELS",
            kernel_init="FAN_OUT",
            rescale_gradients=False,
        )
        self.biases = [
            self.param(
                f"gaunt_symmetric_contraction_bias_{i}",
                nn.initializers.normal(),
                (self.num_channels,),
            )
            for i in range(self.correlation)
        ]

    def __call__(
        self, x: e3nn.IrrepsArray, node_species: jnp.ndarray
    ) -> e3nn.IrrepsArray:

        x = x.axis_to_mul()
        x_0 = self.linear_in[0](x)
        y = x_0.mul_to_axis().array @ self.harmonics_src.T
        for c in range(1, self.correlation):
            x_c = self.linear_in[c](x)
            y_c = (
                x_c.mul_to_axis().array @ self.harmonics_src.T + self.biases[c][:, None]
            )
            y = y * y_c
        y = y @ self.harmonics_target / self.num_design_points
        y = self.linear_out(y, node_species)
        y = e3nn.IrrepsArray(self.linear_out.target_irreps, y)
        return y


class GauntMessagePassingBlock(nn.Module):
    """Gaunt-tensor-product version of the equivariant O(3) convolution

    Performs a single message-passing step:

    * Transform RBF radial embeddings with an MLP to form edge scalars,
    * Project sender node features with a Linear layer,
    * Expand spherical embeddings of edge vectors to #num_channels
        channels with a Linear layer,
    * Multiply sender features with spherical embeddings of edge vectors
        channel-wise using the Gaunt Tensor Product,
    * Reweight messages with the edge scalars,
    * Aggregate messages on receiver nodes.

    In equations, the messages are computed as:

    .. math::
    x^{c, \\ell m}_{ij} = \\sum_{\\ell_1, \\ell_2} w^{c, \\ell_1} w^{c, \\ell_2}
    w^{c, \\ell} (x^{c, \\ell_1}_{i} \\otimes Y^{\\ell_2}(\\hat{r}_{ij}))^{\\ell m},

    where :math:'x^{c, \\ell m}' are the node features (c denote the channel index)
    and :math:'w^{c, \\ell_1}, w^{c, \\ell_2}' and :math:'w^{c, \\ell}'
    are the parameters of the linear mixing layers.

    Then the messages are reweighted with the edge scalars
    and aggregated on the receiver nodes:

    .. math::
    x^{c, \\ell m}_{i} = \\sum_{j\\in\\mathcal{N}(i))} \alpha_{ij}^{c, \\ell}
    x^{c, \\ell m}_{ij} / N_{\text{avg}}.

    where :math:'\alpha_{ij}^{c, \\ell}' are the edge scalars
    computed using a MLP on the RBF.

    Using a Gaunt Tensor Product in place of the Clebsch-Gordan
    tensor product along with factorized weights allows to reduce
    the cost of the tensor product, see for instance
    `Heyraud et al. (2026)<https://arxiv.org/pdf/2603.08630>`_.

    Attributes:
        source_irreps: Expected irreps of the input node features, used to
            validate the graph at call time.
        l_max: Maximum degree of for the harmonic embeddings of edge vectors.
        target_irreps: Target irreps for the output node features.
        num_rbf: Number of Bessel radial basis functions, used to validate the
            radial embedding shape.
        radial_mlp_hidden: Dimensions of hidden layers for the radial MLP. The input
            and output layer sizes are dictated by `num_rbf` and the message irreps
            respectively.
        radial_mlp_activation: Activation for hidden layers of the radial MLP.
        avg_num_neighbors: Average number of neighbours per atom, used to
            normalise the aggregated messages.
        layout: internal channel layout for the convolution block.
        radial_mlp_variance_scale: Variance scaling parameter for initialization of the
            radial MLP weights, passed to `jax.nn.initializers.variance_scaling`.
            The default is None, in which case fan-in initializers are used with
            gradient scaling (matching MACE behaviour).
        deterministic_scatter_ops: Whether to use deterministic scatter operations in
            the message passing convolution.
        normalize_activation: Whether to normalize the activation function with
            respect to the L2-norm for a normal Gaussian measure.
            See `e3nn.normalize_function()`. The default is True.
    """

    source_irreps: e3nn.Irreps
    l_max: int
    target_irreps: e3nn.Irreps
    num_rbf: int
    radial_mlp_hidden: list[int]
    radial_mlp_activation: Callable | options.Activation | Literal["beta_swish"]
    avg_num_neighbors: float
    radial_mlp_variance_scale: float | None = None
    deterministic_scatter_ops: bool = False
    normalize_activation: bool = True

    @property
    def message_irreps(self) -> e3nn.Irreps:
        """Filtered tensor product representation of node features with harmonics."""
        return e3nn.tensor_product(
            self.source_irreps,
            e3nn.Irreps.spherical_harmonics(self.l_max),
            filter_ir_out=self.target_irreps,
        ).set_mul(1)

    @property
    def num_channels(self) -> int:
        """GCD of source multiplicities to factorize as a channel axis."""
        return e3nn.Irreps(self.source_irreps).mul_gcd

    @property
    def _src(self) -> e3nn.Irreps:
        """Internal irreps for message-passing.

        The multiplicities may be factorized to a channel axis.
        """
        source_irreps = e3nn.Irreps(self.source_irreps)
        gcd = source_irreps.mul_gcd
        src = e3nn.Irreps([(m // gcd, ir) for m, ir in source_irreps])
        return src

    @staticmethod
    def _fan_in_normal(scale: float) -> initializers.Initializer:
        return initializers.variance_scaling(scale, "fan_in", "normal")

    def setup(self):

        harmonics_irreps = e3nn.Irreps.spherical_harmonics(self.l_max)
        num_scalars = self.message_irreps.num_irreps * self.num_channels

        mlp_init = (
            self._fan_in_normal(self.radial_mlp_variance_scale)
            if self.radial_mlp_variance_scale is not None
            else options.GradientScaledKernelInit.FAN_IN_NORMAL
        )
        mlp_init_out = (
            self._fan_in_normal(1.0)
            if self.radial_mlp_variance_scale is not None
            else options.GradientScaledKernelInit.FAN_IN_NORMAL
        )
        mlp_activation = (
            BetaSwish()
            if self.radial_mlp_activation == "beta_swish"
            else self.radial_mlp_activation
        )
        self.radial_mlp = MLP(
            layer_sizes=[self.num_rbf, *self.radial_mlp_hidden, num_scalars],
            activation=mlp_activation,
            kernel_init=mlp_init,
            output_kernel_init=mlp_init_out,
            normalize_activation=self.normalize_activation,
            use_bias=False,
        )

        self.linear_in = e3j.linen.Linear(
            source_irreps=self.source_irreps,
            target_irreps=self.source_irreps,
            layout="E3NN",
            kernel_init="FAN_IN",
            rescale_gradients=True,
        )

        self.linear_proj = e3j.linen.Linear(
            source_irreps=harmonics_irreps,
            target_irreps=self.num_channels * harmonics_irreps,
            layout="E3NN",
            kernel_init="FAN_IN",
            rescale_gradients=True,
        )

        self.gaunt_tensor_product = GauntTensorProduct(
            sources=(self._src, harmonics_irreps),
            target=self.message_irreps,
        )

        self.linear_out = e3j.linen.Linear(
            source_irreps=self.num_channels * self.message_irreps,
            target_irreps=self.target_irreps,
            layout="E3NN",
            kernel_init="FAN_IN",
            rescale_gradients=True,
        )

    def __call__(self, graph: Graph) -> Graph:
        node_feats = graph.nodes.features["latent"]
        spherical_embedding = graph.edges.features["spherical_embedding"]
        radial_embedding = graph.edges.features["radial_embedding"].array
        senders = graph.senders
        receivers = graph.receivers

        # Compute scalar radial embeddings
        edge_scalars = self.radial_mlp(radial_embedding)

        # Linear in
        node_feats = e3nn.IrrepsArray(
            self.linear_in.target._to_e3nn(), self.linear_in(node_feats.array)
        )
        node_feats = node_feats.mul_to_axis()

        # Linear projection of spherical embeddings
        spherical_embedding = e3nn.IrrepsArray(
            self.linear_proj.target._to_e3nn(),
            self.linear_proj(spherical_embedding.array),
        )
        spherical_embedding = spherical_embedding.mul_to_axis()

        # Apply message passing:
        # Gather + gaunt tensor product + scalar mixing + scatter-add

        sender_feats = node_feats[senders]
        messages = self.gaunt_tensor_product(sender_feats, spherical_embedding)
        messages = messages.axis_to_mul()
        messages = messages * edge_scalars

        # Scatter-add messages to receiver nodes
        receiver_feats = segment_sum(
            messages,
            receivers,
            num_segments=node_feats.shape[0],
            deterministic=self.deterministic_scatter_ops,
        )  # [n_nodes, irreps_dim]

        receiver_feats /= self.avg_num_neighbors

        # Linear out
        node_feats = e3nn.IrrepsArray(
            self.target_irreps,
            self.linear_out(receiver_feats.array),
        )
        return graph.update_node_features(latent=node_feats)
