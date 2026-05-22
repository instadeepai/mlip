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

import dataclasses


@dataclasses.dataclass(frozen=True)
class Properties:
    """Holds the required properties of a force field model.

    Attributes:
        energy: Whether the model should predict total energy. Default is True.
        forces: Whether the model should predict atomic forces. Default is True.
        partial_charges: Whether the model should predict partial atomic charges.
            Default is False.
        hessian: Whether the model should predict the Hessian (second derivative of
            energy). Default is False.
        stress: Whether the model should predict the stress tensor. Default is False.
    """

    energy: bool = True
    forces: bool = True
    partial_charges: bool = False
    hessian: bool = False
    stress: bool = False

    def true_fields(self) -> list[str]:
        """Return a list of all properties that are true."""
        return [
            field.name
            for field in dataclasses.fields(self)
            if getattr(self, field.name) is True
        ]
