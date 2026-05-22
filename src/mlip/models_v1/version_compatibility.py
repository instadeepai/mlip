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


import os
from typing import Literal

# NOTE: MLIP_BACKWARD_V1_COMPATIBLE must be true (the default) to load V1 models.
#       Only set MLIP_BACKWARD_V1_COMPATIBLE=false for v1 blocks to match v2 exactly.
#       This option is only introduced for testing purposes, as it changes e.g. the
#       ordering of edge scalars, some normalization factors or padding behaviours.
VERSION: Literal[1, 2] = (
    1 if os.environ.get("MLIP_BACKWARD_V1_COMPATIBLE", "true").lower() != "false" else 2
)
