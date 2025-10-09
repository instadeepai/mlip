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

import abc
from typing import Any, Callable

import ase
import pydantic

from mlip.models import ForceField
from mlip.simulation.configs.simulation_config import SimulationConfig
from mlip.simulation.state import SimulationState


class SimulationEngine(abc.ABC):
    """Abstract base class of a simulation engine that can be implemented by different
    backends and can, in principle, run many types of
    simulations (e.g., MD or energy minimizations).
    """

    Config = pydantic.BaseModel  # must be overridden by child classes

    def __init__(
        self,
        atoms: ase.Atoms,
        force_field: ForceField,
        config: SimulationConfig,
    ) -> None:
        """Constructor that initializes the simulation state and
        an empty list of loggers. Engine-specific initialization is then
        delegated to ``._initialize()``

        Args:
            atoms: The atoms of the system to simulate.
            force_field: The force field to use in the simulation.
            config: The configuration/settings of the simulation.

        Raises:
            ValueError: if input system has only one atom.
        """
        if len(atoms) == 1:
            raise ValueError("Single atom systems are not supported yet.")

        self.state = SimulationState()
        self.loggers: list[Callable[[SimulationState], None]] = []
        self._initialize(atoms, force_field, config)

    @abc.abstractmethod
    def _initialize(
        self,
        atoms: ase.Atoms,
        force_field: ForceField,
        config: SimulationConfig,
    ) -> None:
        """Subclasses should implement this method to handle their
        specific initialization.

        Args:
            atoms: The atoms of the system to simulate.
            force_field: The force field to use in the simulation.
            config: The configuration/settings of the simulation.
        """
        pass

    @abc.abstractmethod
    def run(self) -> None:
        """Runs the simulation and populates the simulation state during the run.
        Note that this method should only be called once and its behaviour will not
        be defined if called a second time."""
        pass

    def attach_logger(self, logger: Callable[[SimulationState], None]) -> None:
        """Adds a logger to the list of loggers of the simulation engine.

        The logger function must only take in a single argument, the simulation state,
        and it shall not return anything.

        Args:
            logger: The logger to add.
        """
        self.loggers.append(logger)

    def __init_subclass__(cls, **kwargs: Any):
        """This enforces that child classes will
        need to override the `Config` attribute.
        """
        super().__init_subclass__(**kwargs)
        if cls.Config is pydantic.BaseModel:
            raise NotImplementedError(
                f"{cls.__name__} must override the `Config` attribute."
            )
