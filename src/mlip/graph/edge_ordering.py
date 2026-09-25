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

from e3j.utils import options

# Ordering of graph edges: SENDER | RECEIVER | NONE
EdgeOrdering: TypeAlias = options.GraphOrdering

# Default = SENDER required for e3j fused CUDA / Mosaic TPU convolution kernels.
# Note: SENDER makes symmetry assumptions on the graph (edge ij = ji) and edge features
# (ylm[ij] = (-1)**l ylm[ji] and s[ij] = s[ji] for scalars) that RECEIVER does not.
# Note: other backends (e.g. OpenEquivariance) may be faster end-to-end with RECEIVER.
DEFAULT_EDGE_ORDERING = EdgeOrdering.SENDER
