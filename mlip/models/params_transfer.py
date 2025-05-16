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

from mlip.typing import ModelParameters


class ParameterTransferImpossibleError(Exception):
    """Exception to be raised if the destination and source parameters deviate more
    in their structures than just having some missing blocks in the source."""


def _params_transfer_helper(
    dict_src: dict, dict_dst: dict, scale_factor: float, missing_keys: list[str]
) -> dict:
    for key in dict_dst:
        if key in dict_src:
            if isinstance(dict_src[key], dict):
                if not isinstance(dict_dst[key], dict):
                    raise ParameterTransferImpossibleError(
                        "Destination and source parameters have "
                        "incompatible structures."
                    )
                dict_dst[key] = _params_transfer_helper(
                    dict_src[key], dict_dst[key], scale_factor, missing_keys
                )
            else:
                dict_dst[key] = dict_src[key]
        else:
            missing_keys.append(key)
            dict_dst[key] = jax.tree.map(lambda x: x * scale_factor, dict_dst[key])

    return dict_dst


def transfer_params(
    params_source: ModelParameters,
    params_destination: ModelParameters,
    scale_factor: float = 1.0,
) -> tuple[ModelParameters, list[str]]:
    """Transfer parameters from a source to a destination.

    Typically, the destination will be some newly initialized parameters that have some
    additional blocks in them compared to a source, which is an already trained model.
    This function will raise an exception if the two parameters deviate more than this
    from one another.

    Args:
        params_source: The parameters to transfer into the destination.
        params_destination: The destination parameters that may contain additional
                            blocks compared to the source.
        scale_factor: Scale factor to multiply the new parameters by. Default is 1.0.


    Returns:
        A tuple of the updated destination parameters and a list of strings, which
        represent the key names that were missing in the source parameters.

    Raises:
        ParameterTransferImpossibleError: if the source and destination parameters are
                                          incompatible with each other.

    """
    missing_keys_in_source = []

    params_destination = _params_transfer_helper(
        params_source, params_destination, scale_factor, missing_keys_in_source
    )

    return params_destination, missing_keys_in_source
