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

import jax
import jax.numpy as jnp

from mlip.simulation.jax_md.helpers import is_neighbor_list


def pad_neighbor_lists(neighbor_lists: list, padding_value: int) -> list:
    """Pad all neighbor lists to the same maximum size.

    Finds the maximum edge capacity across all neighbor lists and pads them to match.
    Used to enable reusing a single compiled step_fun across multiple engines.
    """

    max_edges = max(
        nl.idx.shape[1]
        for neighbors in neighbor_lists
        for nl in jax.tree_util.tree_leaves(neighbors, is_leaf=is_neighbor_list)
        if is_neighbor_list(nl)
    )

    def _pad_nl(nl):
        pad_len = max_edges - nl.idx.shape[1]
        if pad_len > 0:
            new_idx = jnp.pad(
                nl.idx, ((0, 0), (0, pad_len)), constant_values=padding_value
            )
            return nl.set(idx=new_idx, max_occupancy=max_edges)
        return nl

    # Apply the padding to every neighbor list
    return [jax.tree.map(_pad_nl, n, is_leaf=is_neighbor_list) for n in neighbor_lists]
