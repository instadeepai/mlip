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

import jraph
import numpy as np
import pytest

from mlip.data.helpers.dynamically_batch import dynamically_batch
from mlip.data.helpers.hessian_utils import (
    _concat_hessian_rows,  # noqa
    _sample_hessian_rows,  # noqa
)

# fmt: off
EXPECTED_QUADRATIC_MLIP_HESSIAN = np.array([
    [[[3.79467e-2, 3.75544e-6, -6.35845e-6],
      [-3.79467e-02, -3.75544e-6, 6.35845e-6]],

      [[3.75576e-06, 3.79584e-02, -2.83611e-06],
       [-3.75576e-06, -3.79584e-02, 2.83611e-06]],

      [[-6.35761e-06, -2.83631e-06, 3.76432e-02],
       [6.35761e-06, 2.83631e-06, -3.76432e-02]],
    ],
    [[[-3.79467e-2, -3.75544e-6, 6.35845e-6],
      [3.79467e-02, 3.75544e-6, -6.35845e-6]],

      [[-3.75576e-06, -3.79584e-02, 2.83611e-06],
       [3.75576e-06, 3.79584e-02, -2.83611e-06]],

      [[6.35761e-06, 2.83631e-06, -3.76432e-02],
       [-6.35761e-06, -2.83631e-06, 3.76432e-02]],
    ],
])
# fmt: on


@pytest.mark.parametrize(
    "hessian_rows",
    [None, np.array(True), np.array([[0, 2]])],  # arbitrary indices
)
def test_quadratic_mlip_hessian_predictor(
    salt_graph, quadratic_hessian_force_field, hessian_rows
):
    graph = salt_graph
    graph = next(
        dynamically_batch(
            [graph],
            n_node=graph.nodes.positions.shape[0] + 1,
            n_edge=graph.senders.shape[0] + 1,
            n_graph=2,
        )
    )
    graph = graph.replace_globals(sample_hessian_rows=hessian_rows)

    padding_mask = jraph.get_graph_padding_mask(graph)
    n_real_node = sum(graph.n_node[padding_mask])

    hessian = quadratic_hessian_force_field(graph).hessian
    if hessian_rows is not None:
        if hessian_rows.ndim == 0:
            # crop padding nodes
            hessian = hessian[:n_real_node, :, :n_real_node, :]
            assert hessian.shape == (n_real_node, 3, n_real_node, 3)
            np.testing.assert_allclose(
                EXPECTED_QUADRATIC_MLIP_HESSIAN, hessian, rtol=1e-3
            )

        else:
            # mimic how Hessians are processed in GraphDataset
            # [n_rows, n_graph, n_node, 3]
            n_graph, num_hessian_rows = hessian_rows.shape
            true_hessian_sample = EXPECTED_QUADRATIC_MLIP_HESSIAN.reshape(
                n_real_node * 3, n_real_node, 3
            )[hessian_rows].transpose(1, 0, 2, 3)

            true_hessian_sample = true_hessian_sample.reshape(
                num_hessian_rows, n_graph * n_real_node, 3
            ).transpose(1, 0, -1)

            hessian = hessian[:n_real_node]

            assert hessian.shape == (n_real_node, num_hessian_rows, 3)
            assert hessian.shape == true_hessian_sample.shape
            np.testing.assert_allclose(true_hessian_sample, hessian, rtol=1e-3)

    else:
        assert hessian is None


def test_hessian_distillation(salt_graph, quadratic_hessian_force_field):
    graph = salt_graph
    graph = next(
        dynamically_batch(
            [graph],
            n_node=graph.nodes.positions.shape[0] + 1,
            n_edge=graph.senders.shape[0] + 1,
            n_graph=2,
        )
    )
    num_hessian_rows = 2

    n_node, _ = graph.nodes.positions.shape
    padding_mask = jraph.get_graph_padding_mask(graph)
    n_real_node = sum(graph.n_node[padding_mask])

    # full Hessian returned when graph.globals.hessian_rows is not None
    # and graph.globals.hessian_rows.ndims == 0
    graph = graph.replace_globals(sample_hessian_rows=np.array(True))
    teacher_hessian = quadratic_hessian_force_field(graph).hessian
    teacher_hessian = teacher_hessian[:n_real_node, :, :n_real_node, :]
    assert teacher_hessian.shape == (n_real_node, 3, n_real_node, 3)

    # remove padding and reshape to (n_node * 3, max_system_size, 3)
    teacher_hessian = teacher_hessian.reshape(n_real_node * 3, max(graph.n_node), 3)
    teacher_hessian_sample, sampled_rows = _sample_hessian_rows(
        teacher_hessian,
        num_hessian_rows,
        graph.n_node[padding_mask],
        len(graph.n_node[padding_mask]),
    )
    teacher_hessian_sample = _concat_hessian_rows(
        hessian_samples=teacher_hessian_sample,
        n_node=graph.n_node[padding_mask],
        target_len=n_node,
    )
    teacher_hessian_sample = teacher_hessian_sample.transpose(1, 0, -1)[:-1]

    # Hessian subsample returned when graphs.globals.hessian_rows is not None
    # and graph.globals.hessian_rows.ndims > 0
    graph = graph.replace_nodes(hessian=teacher_hessian_sample)
    graph = graph.replace_globals(sample_hessian_rows=sampled_rows)
    student_hessian = quadratic_hessian_force_field(graph).hessian[:-1, :, :]

    assert student_hessian.shape == (n_real_node, num_hessian_rows, 3)
    assert teacher_hessian_sample.shape == student_hessian.shape
    np.testing.assert_allclose(teacher_hessian_sample, student_hessian, rtol=1e-4)
