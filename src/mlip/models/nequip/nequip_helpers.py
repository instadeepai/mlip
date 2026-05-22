# Copyright (c) 2021 The President and Fellows of Harvard College
# Copyright (c) 2025 The NequIP Developers
#
# Licensed under the MIT License (https://opensource.org/licenses.MIT)
#
# Copyright (c) 2026 InstaDeep Ltd
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

import e3nn_jax as e3nn


def split_target_node_irreps(
    source_node_irreps: e3nn.Irreps,
    spherical_embedding_irreps: e3nn.Irreps,
    target_irreps: e3nn.Irreps,
) -> tuple[e3nn.Irreps, e3nn.Irreps, e3nn.Irreps]:
    """Split target node irreps into the three components required by the NequIP gate.

    Splits `target_irreps` into scalars (l=0) and non-scalars (l>0), retaining
    only those reachable via a tensor product path from `source_node_irreps` ×
    `spherical_embedding_irreps`.

    Args:
        source_node_irreps: Irreps of the current node feature representations.
        spherical_embedding_irreps: Irreps of the spherical harmonic edge embeddings.
        target_irreps: The desired output irreps of the layer.

    Returns:
        A tuple `(irreps_scalars, irreps_gate_scalars, irreps_nonscalars)`, where:
            - `irreps_scalars`: reachable l=0 irreps from target_irreps.
            - `irreps_gate_scalars`: one scalar per non-scalar channel, used to gate
              the non-scalar outputs. e3nn.gate requires these between scalars and
              non-scalars: `irreps_scalars + irreps_gate_scalars + irreps_nonscalars`.
            - `irreps_nonscalars`: reachable l>0 irreps from target_irreps.
    """
    irreps_scalars = []
    irreps_nonscalars = []

    for multiplicity, irrep in e3nn.Irreps(target_irreps):
        if not tp_path_exists(source_node_irreps, spherical_embedding_irreps, irrep):
            continue
        # NOTE: e3nn.Irrep(irrep) is needed here although irrep is already `Irrep`.
        if e3nn.Irrep(irrep).l == 0:
            irreps_scalars.append((multiplicity, irrep))
        else:
            irreps_nonscalars.append((multiplicity, irrep))

    irreps_scalars = e3nn.Irreps(irreps_scalars)
    irreps_nonscalars = e3nn.Irreps(irreps_nonscalars)

    gate_scalar_irreps_type = (
        "0e"
        if tp_path_exists(source_node_irreps, spherical_embedding_irreps, "0e")
        else "0o"
    )
    irreps_gate_scalars = e3nn.Irreps([
        (mul, gate_scalar_irreps_type) for mul, _ in irreps_nonscalars
    ])

    return irreps_scalars, irreps_gate_scalars, irreps_nonscalars


def tp_path_exists(
    arg_in1: e3nn.Irreps | str,
    arg_in2: e3nn.Irreps | str,
    arg_out: e3nn.Irrep | str,
) -> bool:
    """Determine whether a tensor product path exists.

    A path is allowed by the Clebsch-Gordan (CG) selection rule if `arg_out`
    appears in the CG decomposition of at least one pair of irreps drawn from
    `arg_in1` and `arg_in2`, i.e. if `|l_in1 - l_in2| <= l_out <= l_in1 + l_in2`
    holds for some pair. Similar to the helper in https://github.com/e3nn/e3nn.
    """
    arg_in1 = e3nn.Irreps(arg_in1).simplify()
    arg_in2 = e3nn.Irreps(arg_in2).simplify()
    arg_out = e3nn.Irrep(arg_out)

    for _multiplicity_1, irreps_1 in arg_in1:
        for _multiplicity_2, irreps_2 in arg_in2:
            if arg_out in irreps_1 * irreps_2:
                return True
    return False
