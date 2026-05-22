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

DEFAULT_WEIGHT = 1.0
DEFAULT_PBC = (False, False, False)
DEFAULT_CELL = np.zeros((3, 3))
PROPERTY_NAMES = [
    "forces",
    "energy",
    "stress",
    "hessian",
    "partial_charges",
    "charge",
    "dipole_moment",
    "spin_multiplicity",
]
DEFAULT_PARTIAL_CHARGE_PROPERTY = "partial_charges"
DEFAULT_PROPERTY_KEY_MAPPING = {
    property_name: property_name for property_name in PROPERTY_NAMES
} | {"partial_charges": DEFAULT_PARTIAL_CHARGE_PROPERTY}
