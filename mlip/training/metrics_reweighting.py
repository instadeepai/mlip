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

import jax.numpy as jnp
from jraph import GraphsTuple, get_graph_padding_mask


def reweight_metrics_by_number_of_graphs(
    metrics: dict[str, jnp.ndarray],
    batch: GraphsTuple,
    avg_n_graphs_per_batch: float,
) -> dict[str, jnp.ndarray]:
    """Reweights the metrics dictionary to account for different number of real
    graphs per batch.

    Multiplies each metric by a factor of `n_graphs / avg_n_graphs_per_batch`. Metrics
    that contain the word "weight" in its name will be skipped.

    Args:
        metrics: The metrics dictionary.
        batch: The batch from which we extract the number of real graphs
               contained in it.
        avg_n_graphs_per_batch: The average number of graphs per batch
                                over the whole dataset.

    Returns:
        The scaled metrics dictionary.
    """
    n_graphs = jnp.sum(get_graph_padding_mask(batch))
    scaling_factor = n_graphs / avg_n_graphs_per_batch

    def _apply_factor(value: jnp.array, metric_name: str) -> jnp.array:
        return scaling_factor * value if "weight" not in metric_name else value

    return {
        metric_name: _apply_factor(value, metric_name)
        for metric_name, value in metrics.items()
    }
