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

from typing import TypeAlias

import e3nn_jax as e3nn
import flax
import jax
import jax.numpy as np
import jax.random as random
from e3j.utils.options import Layout

from mlip.models.mace.symmetric_contraction import SymmetricContraction

# is there a jax/flax PyTree type?
Params: TypeAlias = dict


class TestSymmetricContraction:
    # test parameters
    key = random.key(123)
    batch_size = 32
    num_channels = 2
    source_irreps = "0e + 1o"
    irreps_in = f"{num_channels}x0e + {num_channels}x1o"
    # module arguments
    correlation = 3
    keep_irrep_out = "0e + 1o"
    num_species = 4
    layout: Layout = Layout.TRAILING_CHANNELS

    def module_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Prepare module inputs: (node_feats, species)."""
        nb = self.batch_size
        rep_in = e3nn.Irreps(self.irreps_in)
        node_feats = random.normal(self.key, (nb, rep_in.dim))
        node_feats = e3nn.IrrepsArray(rep_in, node_feats).mul_to_axis()
        species = random.randint(self.key, (nb,), 0, self.num_species)
        return (node_feats, species)

    def module_params(self) -> Params:
        """Prepare parameters."""
        module = self.module()
        inputs = self.module_inputs()
        return module.init(self.key, *inputs)

    def module(self) -> flax.linen.Module:
        """Prepare module."""
        return SymmetricContraction(
            source_irreps=self.source_irreps,
            correlation=self.correlation,
            keep_irrep_out=self.keep_irrep_out,
            num_species=self.num_species,
            num_channels=self.num_channels,
            layout=self.layout,
        )

    def test_symmetric_contraction(self):
        """Check that module runs without error."""
        module = self.module()
        inputs = self.module_inputs()
        params = self.module_params()
        out = jax.jit(module.apply)(params, *inputs)
        assert out.array.shape[0] == self.batch_size
