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

import operator
import pickle
from pathlib import Path

import jax.tree_util
import pytest

from mlip.models.params_transfer import transfer_params

DATA_DIR = Path(__file__).parent.parent / "data"
MACE_PARAMS_PICKLE_FILE = DATA_DIR / "mace_test_params.pkl"
MACE_PARAMS_3_HEADS_PICKLE_FILE = DATA_DIR / "mace_test_params_3_heads.pkl"


@pytest.mark.parametrize("scale_factor", [0.0, 1.0])
def test_transfer_of_parameters_works_correctly(scale_factor):
    with MACE_PARAMS_PICKLE_FILE.open("rb") as pkl_file:
        mace_params_1_head = pickle.load(pkl_file)

    with MACE_PARAMS_3_HEADS_PICKLE_FILE.open("rb") as pkl_file:
        mace_params_3_heads = pickle.load(pkl_file)

    example_indexes = [(0, 0), (1, 2), (3, 2)]
    for i, j in example_indexes:
        assert float(
            mace_params_1_head["params"]["mlip_network"]["MaceBlock_0"][
                "LinearNodeEmbeddingBlock_0"
            ]["embeddings"][i][j]
        ) != float(
            mace_params_3_heads["params"]["mlip_network"]["MaceBlock_0"][
                "LinearNodeEmbeddingBlock_0"
            ]["embeddings"][i][j]
        )

    transferred, missing_keys = transfer_params(
        mace_params_1_head, mace_params_3_heads, scale_factor=scale_factor
    )

    for i, j in example_indexes:
        assert float(
            mace_params_1_head["params"]["mlip_network"]["MaceBlock_0"][
                "LinearNodeEmbeddingBlock_0"
            ]["embeddings"][i][j]
        ) == float(
            transferred["params"]["mlip_network"]["MaceBlock_0"][
                "LinearNodeEmbeddingBlock_0"
            ]["embeddings"][i][j]
        )

    assert missing_keys == [
        "LinearReadoutBlock_1",
        "LinearReadoutBlock_2",
        "NonLinearReadoutBlock_1",
        "NonLinearReadoutBlock_2",
    ]

    mace_params_block = transferred["params"]["mlip_network"]["MaceBlock_0"]
    for block in [
        mace_params_block["layer_0"]["LinearReadoutBlock_1"],
        mace_params_block["layer_0"]["LinearReadoutBlock_2"],
        mace_params_block["layer_1"]["NonLinearReadoutBlock_1"],
        mace_params_block["layer_1"]["NonLinearReadoutBlock_2"],
    ]:
        if scale_factor == 0.0:
            assert jax.tree.reduce(operator.add, block).min() == 0.0
            assert jax.tree.reduce(operator.add, block).max() == 0.0
        else:
            assert jax.tree.reduce(operator.add, block).min() < 0.0
            assert jax.tree.reduce(operator.add, block).max() > 0.0
