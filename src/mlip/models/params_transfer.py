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

import copy
import re

import jax

from mlip.typing import ModelParameters

# Matches flax-style readout block names, e.g. `ReadoutBlock_0`,
# `NequipReadoutBlock_1`. Group 1 is the block-type prefix, group 2 the head
# index. Used to warm-start new heads from head 0 during fine-tuning.
_READOUT_BLOCK_RE = re.compile(r"^(.*)ReadoutBlock_(\d+)$")


class ParameterTransferImpossibleError(Exception):
    """Exception to be raised if the destination and source parameters deviate more
    in their structures than just having some missing blocks in the source.
    """


def _params_transfer_helper(
    dict_src: dict, dict_dst: dict, scale_factor: float
) -> dict:
    for key, val_dst in dict_dst.items():
        if key in dict_src:
            val_src = dict_src[key]
            if isinstance(val_src, dict):
                if not isinstance(val_dst, dict):
                    raise ParameterTransferImpossibleError(
                        "Destination and source parameters have "
                        "incompatible structures."
                    )
                dict_dst[key] = _params_transfer_helper(val_src, val_dst, scale_factor)
            else:
                dict_dst[key] = val_src
            continue

        # Warm-start additional readout heads (`*ReadoutBlock_N` with N >= 1)
        # from the pretrained head-0 weights rather than leaving them at their
        # random init. Scaling by `scale_factor=0` is kept as an escape hatch
        # for tests / callers that explicitly want the zeroed-init behaviour.
        match = _READOUT_BLOCK_RE.match(key)
        if match is not None and scale_factor != 0.0:
            source_key = f"{match.group(1)}ReadoutBlock_0"
            if source_key in dict_src:
                dict_dst[key] = copy.deepcopy(dict_src[source_key])
                continue

        dict_dst[key] = jax.tree.map(lambda x: x * scale_factor, val_dst)

    return dict_dst


def transfer_params(
    params_source: ModelParameters,
    params_destination: ModelParameters,
    scale_factor: float = 1.0,
) -> ModelParameters:
    """Transfer parameters from a source to a destination.

    Typically, the destination will be some newly initialized parameters that have some
    additional blocks in them compared to a source, which is an already trained model.
    This function will raise an exception if the two parameters deviate more than this
    from one another.

    For new readout heads (`*ReadoutBlock_N` with N >= 1 that don't exist in
    `params_source`), the params are warm-started by deep-copying
    `*ReadoutBlock_0` from the source, so fine-tuning begins from the
    pretrained readout weights rather than a random init. Pass
    `scale_factor=0.0` to restore the original "scaled-init" behaviour
    (useful for tests that want reproducible zero-initialised new blocks).

    Args:
        params_source: The parameters to transfer into the destination.
        params_destination: The destination parameters that may contain additional
                            blocks compared to the source.
        scale_factor: Scale factor applied to the destination's random init for
                      any new block that's neither in the source nor an added
                      readout head. Default is 1.0.

    Returns:
        The updated destination parameters.

    Raises:
        ParameterTransferImpossibleError: if the source and destination parameters are
                                          incompatible with each other.

    """
    return _params_transfer_helper(params_source, params_destination, scale_factor)


def count_readout_heads(params: ModelParameters) -> int:
    """Count the number of distinct readout heads in a parameters tree.

    A readout head is any `*ReadoutBlock_N` entry (e.g. `LinearReadoutBlock_0`,
    `NonLinearReadoutBlock_1`); two blocks with the same `N` but different
    prefixes (e.g. linear + non-linear at different layers) are one head.

    Args:
        params: A parameters pytree, typically nested dicts of arrays.

    Returns:
        The number of distinct head indices found anywhere in `params`.
    """
    indices: set[int] = set()

    def _walk(node: dict) -> None:
        for key, val in node.items():
            match = _READOUT_BLOCK_RE.match(key)
            if match is not None:
                indices.add(int(match.group(2)))
            if isinstance(val, dict):
                _walk(val)

    _walk(params)
    return len(indices)
