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

from dataclasses import dataclass

import jax
import jraph
from pydantic import BaseModel
from typing_extensions import Self

from mlip.data import DatasetInfo
from mlip.data.helpers.dummy_init_graph import get_dummy_graph_for_model_init
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.predictor import ForceFieldPredictor
from mlip.typing import ModelParameters, Prediction


@dataclass(frozen=True)
class ForceField:
    """An initialized force field, wrapping a
    :class:`~mlip.models.predictor.ForceFieldPredictor` with parameters.

    `ForceField` instances can be called on an input graph to obtain the
    predictions of the force field. Internally, this means we compose the
    `predictor.apply` method inherited from `flax.linen.Module` with the current
    dictionary of learnable parameters.

    Only the `cutoff_distance` and `allowed_atomic_numbers` properties are subject
    to duck-typing in the simulation engine. Users are therefore free to provide
    any other force field callable that provides this simple interface.

    Attributes:
        predictor: The :class:`~mlip.models.predictor.ForceFieldPredictor`
                   which derives forces and stress from the
                   underlying :class:`~mlip.models.mlip_network.MLIPNetwork`
                   energy model.
        params: The dictionary of learnable parameters. If an integer is passed
                instead, it will be used as seed for the random number generator
                to initialize model parameters.
    """

    predictor: ForceFieldPredictor
    params: ModelParameters

    @classmethod
    def from_mlip_network(
        cls, mlip_network: MLIPNetwork, predict_stress: bool = False, seed: int = 42
    ) -> Self:
        """Initializes a force field from an
        :class:`~mlip.models.mlip_network.MLIPNetwork` instance with random parameters.

        This is an alternative constructor to this dataclass, but the preferred one in
        a typical MLIP pipeline.

        Args:
            mlip_network: The MLIP network to use in this force field.
            predict_stress: Whether to predict stress properties with this force field.
                            Default is `False`.
            seed: The initialization seed for the parameters. Default is 42.

        Returns:
            The initialized instance of the force field.
        """
        predictor = ForceFieldPredictor(
            mlip_network=mlip_network, predict_stress=predict_stress
        )
        return cls.init(predictor, seed)

    def __call__(self, graph: jraph.GraphsTuple) -> Prediction:
        """Predict physical properties of an input graph from current parameters.

        See documentation of the
        :meth:`~mlip.models.predictor.ForceFieldPredictor.__call__` method of
        :class:`~mlip.models.predictor.ForceFieldPredictor` for more
        information on the returned object.
        """
        return self.predictor.apply(self.params, graph)

    @classmethod
    def init(cls, predictor: ForceFieldPredictor, seed: int = 42):
        """Initialize force field parameters from random number generator seed.

        Args:
            predictor: The force field predictor.
            seed: The integer seed to generate initial random parameters from.

        Returns:
            The initialized instance of the force field (with random parameters).
        """
        params = predictor.init(
            jax.random.key(seed),
            get_dummy_graph_for_model_init(),
        )
        return cls(predictor, params)

    @property
    def cutoff_distance(self) -> float:
        """Cutoff distance in Angstrom the model was built for."""
        dataset_info = self.predictor.mlip_network.dataset_info
        return dataset_info.cutoff_distance_angstrom

    @property
    def allowed_atomic_numbers(self) -> set[int]:
        """Set of atomic numbers supported by the model."""
        dataset_info = self.predictor.mlip_network.dataset_info
        return set(dataset_info.atomic_energies_map.keys())

    @property
    def config(self) -> BaseModel:
        """Return configuration of the underlying MLIP model."""
        return self.predictor.mlip_network.config

    @property
    def dataset_info(self) -> DatasetInfo:
        """Return dataset info stored in the MLIP network."""
        return self.predictor.mlip_network.dataset_info

    def __hash__(self):
        """Simple hashing function to allow for jitting `self.__call__` directly."""
        return id(self)

    def __eq__(self, other):
        """Simple comparison based on IDs to allow for
        jitting `self.__call__` directly.
        """
        return id(other) == id(self)
