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
import jax.numpy as np
import jax.random as random

from mlip.models.mace.symmetric_contraction import SymmetricContraction

# is there a jax/flax PyTree type?
Params: TypeAlias = dict


class TestSymmetricContraction:

    # test parameters
    key = random.key(123)
    batch_size = 32
    irreps_in = "2x0e + 2x1o + 2x2e"
    # module arguments
    correlation = 3
    keep_irrep_out = "0e + 1o + 2e"
    num_species = 4

    def module_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Prepare module inputs: (node_feats, species)."""
        nb = self.batch_size
        rep_in = e3nn.Irreps(self.irreps_in)
        node_feats = random.normal(self.key, (nb, rep_in.dim))
        species = random.randint(self.key, (nb,), 0, self.num_species)
        return (e3nn.IrrepsArray(rep_in, node_feats), species)

    def module_params(self) -> Params:
        """Prepare parameters."""
        module = self.module()
        inputs = self.module_inputs()
        return module.init(self.key, *inputs)

    def module(self) -> flax.linen.Module:
        """Prepare module."""
        return SymmetricContraction(
            self.correlation,
            self.keep_irrep_out,
            self.num_species,
        )

    def test_symmetric_contraction(self):
        """Check that module runs without error."""
        module = self.module()
        inputs = self.module_inputs()
        params = self.module_params()
        out = module.apply(params, *inputs)
        print(out.irreps)
        assert out.array.shape[0] == self.batch_size
