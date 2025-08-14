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
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import pytest

from mlip.models.mace.models import MaceBlock
from mlip.models.radial_embedding import bessel_basis, soft_envelope


class _TestMaceBlock:

    # test parameters
    graph_sizes: tuple[int, int] = 128, 512
    key: jax.Array = random.key(42)
    # module arguments
    num_species: int = 6
    num_channels: int = 16
    num_interactions: int = 2
    correlation: int = 3
    l_max: int = 3
    output_irreps: e3nn.Irreps = "1x0e"
    gate_nodes: bool
    species_embedding_dim: int | None

    @pytest.fixture(scope="class")
    def inputs(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        # graph topology
        n0, n1 = self.graph_sizes
        senders, receivers = jax.random.randint(self.key, (2, n1), 0, n0)
        # graph features
        num_species = self.num_species
        node_specie = jax.random.randint(self.key, (n0,), 0, num_species - 1)
        vectors = jax.random.normal(self.key, (n1, 3))
        return (vectors, node_specie, senders, receivers)

    @pytest.fixture(scope="class")
    def mace_block_kwargs(self) -> dict:
        mace_block_kwargs = dict(
            output_irreps=self.output_irreps,
            r_max=10.0,
            num_channels=self.num_channels,
            avg_num_neighbors=4.0,
            num_interactions=self.num_interactions,
            avg_r_min=None,
            num_species=self.num_species,
            num_bessel=8,
            radial_basis=bessel_basis,
            radial_envelope=soft_envelope,
            symmetric_tensor_product_basis=False,
            off_diagonal=False,
            l_max=self.l_max,
            node_symmetry=1,
            include_pseudotensors=False,
            num_readout_heads=1,
            readout_mlp_irreps="16x0e",
            correlation=self.correlation,
            gate=jax.nn.silu,
            gate_nodes=self.gate_nodes,
            species_embedding_dim=self.species_embedding_dim,
        )
        return mace_block_kwargs

    @pytest.fixture(scope="class")
    def module(self, mace_block_kwargs) -> nn.Module:
        return MaceBlock(**mace_block_kwargs)

    def test_output_shape(self, module, inputs):
        params = module.init(self.key, *inputs)
        out = module.apply(params, *inputs)
        n0, n1 = self.graph_sizes
        layers = self.num_interactions
        scalars_out = e3nn.Irreps(self.output_irreps).num_irreps
        assert tuple(out.array.shape) == (n0, layers, 1, scalars_out)


class TestMaceBlockNodesGating(_TestMaceBlock):

    gate_nodes = True
    species_embedding_dim = None


class TestMaceBlockSpeciesEmbedding(_TestMaceBlock):

    gate_nodes = False
    species_embedding_dim = 8
