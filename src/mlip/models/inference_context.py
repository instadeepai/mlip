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

from dataclasses import dataclass

import jax.numpy as jnp

from mlip.data.dataset_info import DatasetInfo
from mlip.graph import Graph


@dataclass(frozen=True)
class InferenceContext:
    """Graph-level metadata applied to a graph before inference.

    Used to populate routing globals (charge, spin multiplicity, dataset index)
    that MoE-enabled models need for expert selection or multi-head models.

    Attributes:
        dataset_name: Human-readable dataset identifier. Must match an entry
            in `DatasetInfo.dataset_name`. Resolved to a `dataset_idx` via
            :meth:`resolve`.
        dataset_idx: Numeric index of the dataset. Equivalent to
            `dataset_name`; if both are given they must agree. `resolve()`
            fills whichever is missing from the other.
        charge: Total system charge to tag on each graph's `globals.charge`.
            Only applied if set.
        spin_multiplicity: Spin multiplicity to tag on each graph's
            `globals.spin_multiplicity`. Only applied if set.
    """

    dataset_name: str | None = None
    dataset_idx: int | None = None
    charge: int | None = None
    spin_multiplicity: int | None = None

    def resolve(self, dataset_info: DatasetInfo) -> "InferenceContext":
        """Return a new `InferenceContext` with `dataset_name` and
        `dataset_idx` cross-filled against the model's known datasets.

        Raises `ValueError` when the name is unknown, the index is
        out-of-range, or the two disagree. Single-dataset models (where
        `dataset_info.dataset_name` is `None`) are pass-through.
        """
        dataset_name = self.dataset_name
        dataset_idx = self.dataset_idx
        available = dataset_info.dataset_name

        if available is not None and isinstance(available, str):
            available = [available]

        if dataset_name is not None:
            if available is None:
                raise ValueError(
                    "InferenceContext.dataset_name was provided, but this model does "
                    "not have dataset names recorded in dataset_info."
                )
            if dataset_name not in available:
                raise ValueError(
                    f"Unknown dataset_name {dataset_name!r}. "
                    f"Expected one of {available!r}."
                )
            resolved_idx = available.index(dataset_name)
            if dataset_idx is not None and dataset_idx != resolved_idx:
                raise ValueError(
                    "InferenceContext.dataset_name and dataset_idx disagree. "
                    f"{dataset_name!r} resolves to {resolved_idx}, "
                    f"but dataset_idx={dataset_idx} was provided."
                )
            dataset_idx = resolved_idx
        elif dataset_idx is not None and available is not None:
            if not 0 <= dataset_idx < len(available):
                raise ValueError(
                    f"dataset_idx={dataset_idx} is out of range"
                    f" for datasets {available!r}."
                )
            dataset_name = available[dataset_idx]

        return InferenceContext(
            dataset_name=dataset_name,
            dataset_idx=dataset_idx,
            charge=self.charge,
            spin_multiplicity=self.spin_multiplicity,
        )


def apply_inference_context_to_graph(
    graph: Graph,
    inference_context: InferenceContext,
) -> Graph:
    """Populate graph globals from an inference context.

    Returns a new graph with the relevant global fields (`dataset_idx`,
    `charge`, `spin_multiplicity`) broadcast to `[n_graphs]`.
    """
    globals_update = {}
    num_graphs = graph.num_graphs

    if inference_context.dataset_idx is not None:
        globals_update["dataset_idx"] = jnp.full(
            num_graphs, inference_context.dataset_idx, dtype=jnp.int32
        )
    if inference_context.charge is not None:
        globals_update["charge"] = jnp.full(
            num_graphs, inference_context.charge, dtype=jnp.int32
        )
    if inference_context.spin_multiplicity is not None:
        globals_update["spin_multiplicity"] = jnp.full(
            num_graphs, inference_context.spin_multiplicity, dtype=jnp.int32
        )

    if not globals_update:
        return graph
    return graph.replace_globals(**globals_update)
