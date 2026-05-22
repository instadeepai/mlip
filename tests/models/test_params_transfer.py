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

import jax
import pytest

from mlip.models.params_transfer import count_readout_heads, transfer_params

DATA_DIR = Path(__file__).parent.parent / "sample_data"
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

    transferred = transfer_params(
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

    assert count_readout_heads(mace_params_1_head) == 1
    assert count_readout_heads(mace_params_3_heads) == 3
    assert count_readout_heads(transferred) == 3

    mace_params_block = transferred["params"]["mlip_network"]["MaceBlock_0"]
    source_mace_block = mace_params_1_head["params"]["mlip_network"]["MaceBlock_0"]
    new_head_blocks = [
        ("layer_0", "LinearReadoutBlock_1"),
        ("layer_0", "LinearReadoutBlock_2"),
        ("layer_1", "NonLinearReadoutBlock_1"),
        ("layer_1", "NonLinearReadoutBlock_2"),
    ]
    for layer, name in new_head_blocks:
        prefix = name.rsplit("_", 1)[0]  # strip "_1" / "_2"
        dst = mace_params_block[layer][name]
        if scale_factor == 0.0:
            # Zero-init escape hatch: every parameter in the new head is 0.
            assert jax.tree.reduce(operator.add, dst).min() == 0.0
            assert jax.tree.reduce(operator.add, dst).max() == 0.0
        else:
            # Warm start: the new head's params were deep-copied from head 0
            # of the pretrained model — so they must match `<prefix>_0`
            # leaf-for-leaf (and not be identically zero, which would mean
            # the warm start silently fell back to the scale-factor branch).
            source_head_0 = source_mace_block[layer][f"{prefix}_0"]
            match_tree = jax.tree.map(
                lambda s, d: bool((s == d).all()), source_head_0, dst
            )
            assert all(jax.tree.leaves(match_tree))
            assert jax.tree.reduce(operator.add, dst).min() != 0.0
            assert jax.tree.reduce(operator.add, dst).max() != 0.0
