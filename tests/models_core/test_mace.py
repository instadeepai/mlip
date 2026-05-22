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

import e3nn_jax as e3nn
import jax
import pytest

from mlip.models.mace.models import MaceBlock
from mlip.models.radial_embedding import bessel_basis, soft_envelope


@pytest.fixture
def graph_sizes():
    return 128, 512


@pytest.fixture
def graph_edges(graph_sizes):
    rng = jax.random.PRNGKey(12)
    n0, n1 = graph_sizes
    senders, receivers = jax.random.randint(rng, (2, n1), 0, n0)
    return senders, receivers


@pytest.fixture
def mace_hparams():
    layers, scalars_out = 3, 8
    return dict(
        output_irreps=f"{scalars_out}x0e",
        num_channels=16,
        r_max=10,
        num_interactions=layers,
        readout_mlp_irreps="16x0e",
        avg_num_neighbors=4.0,
        num_species=6,
        num_bessel=8,
        radial_basis=bessel_basis,
        radial_envelope=soft_envelope,
        l_max=3,
        node_symmetry=3,
        correlation=3,
    )


@pytest.fixture
def mace_inputs(graph_sizes, graph_edges, mace_hparams):
    rng = jax.random.PRNGKey(12)
    # graph topology
    n0, n1 = graph_sizes
    senders, receivers = graph_edges
    # graph features
    num_species = mace_hparams["num_species"]
    node_specie = jax.random.randint(rng, (n0,), 0, num_species - 1)
    vectors = jax.random.normal(rng, (n1, 3))
    return (vectors, node_specie, senders, receivers)


def test_mace_shape(graph_sizes, mace_hparams, mace_inputs):
    rng = jax.random.PRNGKey(12)
    # forward pass
    mace = MaceBlock(**mace_hparams)
    params = mace.init(rng, *mace_inputs)
    out = mace.apply(params, *mace_inputs)
    # check output shape
    n0, n1 = graph_sizes
    layers = mace_hparams["num_interactions"]
    scalars_out = e3nn.Irreps(mace_hparams["output_irreps"]).num_irreps
    assert tuple(out.array.shape) == (n0, layers, 1, scalars_out)
