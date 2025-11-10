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

import numpy as np
from flax.struct import dataclass


@dataclass
class Prediction:
    """Holds data that is a prediction of an MLIP force field model.

    As default, everything is ``None``, such that this class can be initialized with
    any property missing. Units will always be eV for energies and Angstrom for
    length.

    Attributes:
        energy: The energy or energies (if multiple graphs in a batch).
                Can be just a single float or array of shape ``(n_graphs,)``.
        forces: The forces. Will be of shape ``(n_nodes, 3)``.
        stress: The stress tensor. Will be of shape ``(n_graphs, 3, 3)``.
        pressure: The microsopic 0K pressure term. Will be of shape ``(n_graphs,)``.
    """

    energy: float | np.ndarray | None = None
    forces: np.ndarray | None = None

    stress: np.ndarray | None = None
    pressure: np.ndarray | None = None
