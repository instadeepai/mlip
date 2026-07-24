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

import logging
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from typing_extensions import Self

from mlip.data.helpers.dummy_init_graph import get_dummy_graph_for_model_init
from mlip.graph import Graph, GraphEdges
from mlip.models.config import MLIPNetworkConfig
from mlip.models.force_field import ForceField
from mlip.models.inference_context import InferenceContext
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.predictors.energy_heads import (
    coulomb_energy_computation_head,
    standard_energy_computation_head,
)
from mlip.models_v1.mlip_network_v1 import MLIPNetworkV1
from mlip.simulation.fep.alchemical_graph import AlchemicalGraph
from mlip.simulation.fep.enums import FEPStage
from mlip.simulation.fep.models.alchemical_models import AlchemicalMLIPNetwork
from mlip.simulation.fep.models.alchemical_predictor import (
    AlchemicalPredictor,
    LinearAlchemicalPredictor,
    LinearAlchemicalPredictorV1,
)
from mlip.simulation.fep.models.repulsive_potentials import (
    SoftcoreLennardJonesPotential,
    SoftcoreRepulsivePotential,
)
from mlip.typing import ModelParameters
from mlip.typing.properties import Properties

logger = logging.getLogger("mlip")


def _get_dummy_alchemical_graph_for_model_init() -> AlchemicalGraph:
    """Creates a dummy graph for running `AlchemicalForceField.init`

    Replaces fields in the dummy graph to include alchemical edges.
    """
    dummy_graph = AlchemicalGraph.from_graph(get_dummy_graph_for_model_init())
    dummy_graph = dummy_graph.replace_nodes(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        atomic_numbers=np.array([1, 1, 1]),
        forces=np.zeros((3, 3)),
        partial_charges=np.zeros(3),
    )
    dummy_graph = dummy_graph.replace(
        senders=jnp.array([0, 0, 1]),
        receivers=jnp.array([1, 2, 2]),
        n_node=jnp.array([3]),
        n_edge=jnp.array([3]),
        edges=GraphEdges(shifts=jnp.zeros((3, 3)), displ_fun=None),
    )
    return dummy_graph.replace_globals(
        alchemical_lambda=jnp.array([[1.0, 0.0]]),
        alchemical_atom_indices=jnp.array([0]),
    )


@dataclass(frozen=True)
class AlchemicalForceField(ForceField):
    """An initialized alchemical force field, wrapping a
    :class:`~mlip.fep.models.alchemical_predictor.AlchemicalPredictor`.

    Attributes:
        predictor: The AlchemicalPredictor which derives properties from
            the underlying MLIP network.
        params: The dictionary of learnable parameters.
        inference_context: Optional context used by MoE multi-head models to configure
            graph-level routing globals (e.g. charge, dataset index) at inference time.
    """

    predictor: AlchemicalPredictor
    params: ModelParameters
    inference_context: InferenceContext | None = None

    @classmethod
    def from_mlip_network(
        cls,
        mlip_network: AlchemicalMLIPNetwork | MLIPNetwork,
        fep_stage: FEPStage | None = None,
        required_properties: Properties | None = None,
        seed: int = 42,
        repulsive_potential: SoftcoreRepulsivePotential | None = None,
        inference_context: InferenceContext | None = None,
    ) -> Self:
        """Initializes from an `MLIPNetwork` with random parameters.

        Args:
            mlip_network: The MLIP network to use in this force field. If an
                `AlchemicalMLIPNetwork` is provided this is wrapped by an
                `AlchemicalPredictor`. Otherwise, if a basic `MLIPNetwork` is
                provided, this is wrapped by a `LinearAlchemicalPredictor`.
            fep_stage: The FEP stage for which a potential is being computed.
            required_properties: Properties to be computed by the force field.
            seed: The initialization seed for the parameters. Default is 42.
            repulsive_potential: Softcore repulsion potential applied to
                alchemical edges. Defaults to SoftcoreLennardJonesPotential().
            inference_context: Additional context that will be placed on the graph
                during inference, such as dataset_idx.

        Returns:
            The initialized `AlchemicalForceField` instance, with random parameters.
        """
        if required_properties is None:
            required_properties = Properties()
        if repulsive_potential is None:
            repulsive_potential = SoftcoreLennardJonesPotential()

        cls.validate_properties(required_properties, mlip_network.available_properties)
        energy_head = cls.get_energy_head(
            mlip_network.config, required_properties, mlip_network
        )

        kwargs = dict(
            mlip_network=mlip_network,
            required_properties=required_properties,
            energy_head=energy_head,
            repulsive_potential=repulsive_potential,
        )

        if isinstance(mlip_network, AlchemicalMLIPNetwork):
            predictor_cls = AlchemicalPredictor
        else:
            if fep_stage is None:
                raise ValueError(
                    "`fep_stage` is required for `LinearAlchemicalPredictor`."
                )
            kwargs["fep_stage"] = fep_stage

            predictor_cls = (
                LinearAlchemicalPredictorV1
                if isinstance(mlip_network, MLIPNetworkV1)
                else LinearAlchemicalPredictor
            )

        predictor = predictor_cls(**kwargs)
        return cls.init(predictor, seed, inference_context)

    @classmethod
    def validate_properties(
        cls, required_properties: Properties, mlip_available_properties: Properties
    ) -> None:
        """Validates that the required properties can be computed by this force field.

        Args:
            required_properties: The set of properties needed.
            mlip_available_properties: The set of properties the mlip network supports.

        Raises:
            ValueError: If a required property is unavailable.
        """
        if required_properties.hessian:
            raise ValueError(
                "Hessian prediction is not supported by AlchemicalForceField."
            )
        super().validate_properties(required_properties, mlip_available_properties)

    @classmethod
    def get_energy_head(
        cls,
        config: MLIPNetworkConfig,
        required_properties: Properties,
        mlip_network: AlchemicalMLIPNetwork | MLIPNetwork | None = None,
    ) -> Callable[[Graph], Array]:
        """Returns the appropriate energy computation head function.

        Args:
            config: The configuration of the model.
            required_properties: The properties required by the predictor.
            mlip_network: The MLIP network the head will be used with.

        Returns:
            The selected function for computing the energy from a graph object.

        Raises:
            NotImplementedError: If `config.use_coulomb_term` is True, as we do not yet
                support alchemical energy prediction with a Coulomb term.
        """
        energy_head = super().get_energy_head(config, required_properties)

        if energy_head is coulomb_energy_computation_head:
            raise NotImplementedError(
                "Cannot create an `AlchemicalForceField` using a force field with "
                "`use_coulomb_term=True`. Please use a different force field."
            )

        if not isinstance(mlip_network, AlchemicalMLIPNetwork):
            return energy_head

        if energy_head is not standard_energy_computation_head:
            logger.warning(
                f"Using an Alchemical MLIP with energy head '{energy_head.__name__}'. "
                "If this head computes terms using edges that cross the alchemical "
                "boundary, it must handle these edges correctly. The alchemical edge "
                "weight of both standard and long-range edges are available via the "
                "respective edge features e.g. `graph.edges.features['edge_scale']`.",
            )
        return energy_head

    @classmethod
    def init(
        cls,
        predictor: AlchemicalPredictor,
        seed: int = 42,
        inference_context: InferenceContext | None = None,
    ) -> Self:
        """Initialize force field parameters using a random seed.

        Args:
            predictor: The force field predictor.
            seed: The seed to use for generating initial random parameters.
            inference_context: Additional context that will be placed on the graph
                during inference, such as dataset_idx.

        Returns:
            The initialized instance of the force field with random parameters.
        """
        dummy_graph = _get_dummy_alchemical_graph_for_model_init()
        params = predictor.init(jax.random.key(seed), dummy_graph)
        return cls(
            predictor=predictor, params=params, inference_context=inference_context
        )
