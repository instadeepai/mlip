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

import math
from enum import Enum

from typing_extensions import Self


class FEPStage(Enum):
    """Enum for the stage of the FEP workflow.

    We define the states:
        State A: Connected graph containing solute-solvent edges.
        State R: Graph containing repulsive solute-solvent interactions.
        State B: Disjoint graph without solute-solvent edges.

    This enum is used to prevent unnecessary computation inside the models.

    Attributes:
        A: Simulation running only in state A.
        AR: Simulation running between A and R.
        RB: Simulation running between R and B, or in state B.
    """

    A = "A"
    AR = "AR"
    RB = "RB"

    @classmethod
    def from_edge_weight(cls, edge_weight: float) -> Self:
        """Returns the appropriate FEP stage based on the edge weight."""
        if math.isclose(edge_weight, 1.0):
            return cls.A
        elif math.isclose(edge_weight, 0.0, abs_tol=1e-9):
            return cls.RB
        elif 0.0 < edge_weight < 1.0:
            return cls.AR
        else:
            raise ValueError(
                f"Invalid edge weight {edge_weight}. Must be in [0.0, 1.0]."
            )


class EpisodeStatus(Enum):
    """Enum for the status of an episode after completion.

    Attributes:
        SUCCESS: Episode completed successfully without overflowing or exploding.
        OVERFLOW: Overflow in neighbor list was encountered during episode.
        EXPLODED: Simulation exploded during episode.
    """

    SUCCESS = "SUCCESS"
    OVERFLOW = "OVERFLOW"
    EXPLODED = "EXPLODED"
