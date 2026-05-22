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

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Literal

import flax.linen as nn
import jax
import jax.numpy as jnp

from mlip.graph import Graph
from mlip.models.esen.config import (
    MoERoutingGlobal,
    supported_moe_routing_globals,
)
from mlip.models.options import Activation, parse_activation


def _fan_in_uniform_init(rng: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    in_features = shape[-2]
    bound = 1.0 / jnp.sqrt(in_features)
    return jax.random.uniform(rng, shape, minval=-bound, maxval=bound)


def _resolve_routing_global(
    graph: Graph,
    global_name: MoERoutingGlobal,
) -> jax.Array:
    value = getattr(graph.globals, global_name)
    if value is None:
        raise ValueError(
            "Missing required routing global "
            f"{global_name!r}. Expected one of {supported_moe_routing_globals()} "
            "to be populated on graph.globals before calling MoE-enabled eSEN."
        )

    value = jnp.asarray(value)
    n_graphs = graph.num_graphs

    if value.ndim == 0:
        return jnp.broadcast_to(value, (n_graphs,))
    if value.ndim == 1:
        if value.shape[0] == n_graphs:
            return value
        if value.shape[0] == 1:
            return jnp.broadcast_to(value, (n_graphs,))

    raise ValueError(
        "Routing global "
        f"{global_name!r} must be scalar-like or shape [n_graphs]. "
        f"Received shape {value.shape} for n_graphs={n_graphs}."
    )


def resolve_routing_globals(
    graph: Graph,
    routing_globals: tuple[MoERoutingGlobal, ...],
) -> dict[MoERoutingGlobal, jax.Array]:
    """Validate and normalize all required routing globals from the graph.

    Returns a dict mapping each global name to its `[n_graphs]` array.
    Raises `ValueError` if any global is missing or has an unexpected shape.
    """
    return {
        global_name: _resolve_routing_global(graph, global_name)
        for global_name in routing_globals
    }


def get_graph_moe_coefficients(
    graph: Graph,
    num_experts: int,
) -> jax.Array:
    """Extract and validate MoE coefficients from `graph.globals.features`.

    Returns the `[n_graphs, num_experts]` coefficient array.
    Raises `ValueError` if the coefficients are missing or have the wrong shape.
    """
    moe_coeffs = graph.globals.features.get("moe_coefficients")
    if moe_coeffs is None:
        raise ValueError(
            "MoE-enabled eSEN layer expected "
            "graph.globals.features['moe_coefficients'] to be populated before "
            "the routed dense layers ran."
        )

    moe_coeffs = jnp.asarray(moe_coeffs)
    n_graphs = graph.num_graphs

    if moe_coeffs.ndim != 2:
        raise ValueError(
            "graph.globals.features['moe_coefficients'] must have shape "
            f"[n_graphs, num_experts]. Received shape {moe_coeffs.shape}."
        )
    if moe_coeffs.shape[0] != n_graphs:
        raise ValueError(
            "graph.globals.features['moe_coefficients'] has the wrong graph "
            "dimension. "
            f"Expected {n_graphs}, received {moe_coeffs.shape[0]}."
        )
    if moe_coeffs.shape[1] != num_experts:
        raise ValueError(
            "graph.globals.features['moe_coefficients'] has the wrong expert "
            "dimension. "
            f"Expected {num_experts}, received {moe_coeffs.shape[1]}."
        )

    return moe_coeffs


def expand_graph_coeffs_to_edges(
    moe_coeffs: jax.Array,
    n_edge: jax.Array,
    total_edges: int,
) -> jax.Array:
    """Broadcast graph-level MoE coefficients to per-edge coefficients."""
    segment_ids = jnp.repeat(
        jnp.arange(moe_coeffs.shape[0]),
        jnp.atleast_1d(jnp.asarray(n_edge, dtype=jnp.int32)),
        total_repeat_length=total_edges,
    )
    return moe_coeffs[segment_ids]


def _standardize_contraction_coeffs(moe_coeffs: jax.Array) -> jax.Array:
    moe_coeffs = jnp.asarray(moe_coeffs)
    if moe_coeffs.ndim == 2:
        if moe_coeffs.shape[0] == 0:
            raise ValueError(
                "Contraction coefficients cannot be empty. "
                f"Received shape {moe_coeffs.shape}."
            )

        # Called eagerly via @nn.nowrap: bool() on a JAX array is safe here and
        # must not be moved into a JIT-traced context. Contraction produces one
        # dense kernel for the whole model, so batched graphs must share a
        # routing signature before reducing to a single coefficient vector.
        if not bool(jnp.allclose(moe_coeffs, moe_coeffs[:1], atol=1e-6, rtol=1e-6)):
            raise ValueError(
                "Contraction expects a single routing signature across the batch. "
                f"Received non-identical coefficients with shape {moe_coeffs.shape}."
            )
        moe_coeffs = moe_coeffs[0]

    if moe_coeffs.ndim != 1:
        raise ValueError(
            "Contraction coefficients must have shape [num_experts] or "
            f"[n_graphs, num_experts]. Received shape {moe_coeffs.shape}."
        )

    return moe_coeffs


def _contract_expert_kernel(
    expert_kernel: jax.Array, moe_coeffs: jax.Array
) -> jax.Array:
    moe_coeffs = _standardize_contraction_coeffs(moe_coeffs)
    expert_kernel = jnp.asarray(expert_kernel)

    if expert_kernel.ndim != 3:
        raise ValueError(
            "Expected expert_kernel to have shape "
            "[num_experts, in_features, out_features]. "
            f"Received shape {expert_kernel.shape}."
        )
    if expert_kernel.shape[0] != moe_coeffs.shape[0]:
        raise ValueError(
            "Coefficient count must match num_experts. "
            f"Received moe_coeffs.shape={moe_coeffs.shape}, "
            f"expert_kernel.shape={expert_kernel.shape}."
        )

    return jnp.einsum("e,eio->io", moe_coeffs, expert_kernel)


def contract_moe_params(
    params: Mapping[str, Any], moe_coeffs: jax.Array
) -> dict[str, Any]:
    """Contract expert kernels in a full parameter tree into standard dense kernels.

    Walks the parameter tree and replaces every `expert_kernel` with a single
    `kernel` obtained by linearly combining experts using `moe_coeffs`.
    """
    contracted = {}
    for key, value in params.items():
        if isinstance(value, Mapping):
            if "expert_kernel" in value:
                contracted_value = {
                    nested_key: nested_value
                    for nested_key, nested_value in value.items()
                    if nested_key != "expert_kernel"
                }
                contracted_value["kernel"] = _contract_expert_kernel(
                    value["expert_kernel"], moe_coeffs
                )
                contracted[key] = contracted_value
            else:
                contracted[key] = contract_moe_params(value, moe_coeffs)
        else:
            contracted[key] = value
    return contracted


class GlobalsEmbedding(nn.Module):
    """Embeds graph-level routing globals into a fixed-size vector.

    Produces one embedding per global and concatenates them, yielding a
    `[n_graphs, embed_dim * len(routing_globals)]` output.

    Attributes:
        embed_dim: Size of each per-global embedding.
        routing_globals: Names of the graph-level globals to embed.
        embedding_type: Which embedding scheme to use.
        scale: Multiplicative scale for the embedding output.
        rand_emb_num_classes: Size of the lookup table used by
            `"rand_emb"`. Together with `rand_emb_value_offset` this
            defines the supported value range
            `[-rand_emb_value_offset, rand_emb_num_classes - 1 -
            rand_emb_value_offset]`. Defaults to 201 (covers values in
            `[-100, 100]` with the default offset).
        rand_emb_value_offset: Integer added to the rounded global value
            to produce the table index; acts as the zero point of the
            lookup table. Must be non-negative. Defaults to 100.
    """

    embed_dim: int
    routing_globals: tuple[MoERoutingGlobal, ...] = ("charge",)
    embedding_type: Literal["pos_emb", "lin_emb", "rand_emb"] = "pos_emb"
    scale: float = 1.0
    rand_emb_num_classes: int = 201
    rand_emb_value_offset: int = 100

    def setup(self) -> None:
        if self.embedding_type == "pos_emb" and self.embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even when embedding_type='pos_emb'.")
        if self.rand_emb_value_offset < 0:
            raise ValueError("rand_emb_value_offset must be non-negative.")
        if self.rand_emb_num_classes <= self.rand_emb_value_offset:
            raise ValueError(
                "rand_emb_num_classes must be greater than rand_emb_value_offset."
            )

    @nn.compact
    def __call__(self, **globals_values: jax.Array) -> jax.Array:
        embeddings = [
            self._embed_one(
                jnp.asarray(globals_values[global_name]),
                global_name,
            )
            for global_name in self.routing_globals
        ]
        return jnp.concatenate(embeddings, axis=-1)

    def _embed_one(self, values: jax.Array, name: MoERoutingGlobal) -> jax.Array:
        if self.embedding_type == "pos_emb":
            return self._positional_embedding(values, name)
        if self.embedding_type == "lin_emb":
            return self._linear_embedding(values, name)
        if self.embedding_type == "rand_emb":
            return self._random_embedding(values, name)

        raise ValueError(f"Unknown embedding_type {self.embedding_type!r}.")

    def _positional_embedding(
        self, values: jax.Array, name: MoERoutingGlobal
    ) -> jax.Array:
        frequencies = self.param(
            f"frequencies_{name}",
            lambda rng, shape: jax.random.normal(rng, shape) * self.scale,
            (self.embed_dim // 2,),
        )
        projected = values[:, None].astype(jnp.float32) * frequencies[None, :]
        projected = projected * (2 * jnp.pi)
        return jnp.concatenate([jnp.sin(projected), jnp.cos(projected)], axis=-1)

    def _linear_embedding(self, values: jax.Array, name: MoERoutingGlobal) -> jax.Array:
        return nn.Dense(self.embed_dim, name=f"linear_embedding_{name}")(
            values[:, None].astype(jnp.float32)
        )

    def _random_embedding(self, values: jax.Array, name: MoERoutingGlobal) -> jax.Array:
        indices = jnp.clip(
            jnp.round(values).astype(jnp.int32) + self.rand_emb_value_offset,
            0,
            self.rand_emb_num_classes - 1,
        )
        table = self.param(
            f"embedding_table_{name}",
            nn.initializers.normal(0.02),
            (self.rand_emb_num_classes, self.embed_dim),
        )
        return table[indices]


class MoERouter(nn.Module):
    """MLP router that maps graph-level embeddings to expert weights.

    Produces one normalized coefficient vector per graph. The returned
    coefficients are used to blend expert-specific parameters in MoE layers.

    Attributes:
        num_experts: Number of experts to route between.
        hidden_dims: Hidden layer widths for the routing MLP.
        activation: Activation applied after each hidden layer.
    """

    num_experts: int
    hidden_dims: list[int]
    activation: Activation | str = Activation.SILU

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Return softmax routing coefficients with shape `[n_graphs, num_experts]`.

        Args:
            x: Graph-level embedding array with shape `[n_graphs, embed_dim]`.
        """
        activation = parse_activation(self.activation)
        for layer_idx, dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                dim,
                name=f"hidden_{layer_idx}",
            )(x)
            x = activation(x)
        x = nn.Dense(self.num_experts, name="logits")(x)
        return jax.nn.softmax(x, axis=-1)


class MoEDense(nn.Module):
    """Dense layer with optional Mixture-of-Experts parameterisation.

    When `num_experts` is `None`, behaves as a standard dense layer.
    Otherwise stores one kernel per expert and blends them at runtime
    using the provided MoE coefficients.
    """

    features: int
    num_experts: int | None = None
    use_bias: bool = True
    kernel_init: Callable[..., Any] = _fan_in_uniform_init
    bias_init: Callable[..., Any] = nn.initializers.zeros_init()

    def _plain_dense(self, x: jax.Array) -> jax.Array:
        kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features))
        out = jnp.einsum("...i,io->...o", x, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            out = out + bias
        return out

    def _moe_dense(self, x: jax.Array, moe_coeffs: jax.Array | None) -> jax.Array:
        in_features = x.shape[-1]

        expert_kernel = self.param(
            "expert_kernel",
            self.kernel_init,
            (self.num_experts, in_features, self.features),
        )

        if moe_coeffs is None:
            raise ValueError(
                "MoEDense received moe_coeffs=None while num_experts is set. "
                "MoE-enabled graph paths must populate "
                "graph.globals.features['moe_coefficients'], and contracted "
                "inference should use a plain dense model."
            )
        if x.shape[0] != moe_coeffs.shape[0]:
            raise ValueError(
                "MoEDense expects the leading input dimension to match the number "
                "of graph-level MoE coefficient rows. "
                f"Received x.shape={x.shape} and moe_coeffs.shape={moe_coeffs.shape}."
            )
        out = jnp.einsum("ne,eio,n...i->n...o", moe_coeffs, expert_kernel, x)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            out = out + bias

        return out

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        moe_coeffs: jax.Array | None = None,
    ) -> jax.Array:
        if self.num_experts is None:
            return self._plain_dense(x)
        return self._moe_dense(x, moe_coeffs)
