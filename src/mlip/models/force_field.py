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

from dataclasses import dataclass, replace

import jax
from jax import Array
from typing_extensions import Callable, Self

from mlip.data import DatasetInfo
from mlip.data.helpers.dummy_init_graph import get_dummy_graph_for_model_init
from mlip.graph import Graph
from mlip.models.config import MLIPNetworkConfig
from mlip.models.inference_context import (
    InferenceContext,
    apply_inference_context_to_graph,
)
from mlip.models.mlip_network import MLIPNetwork
from mlip.models.predictors import (
    ConservativePredictor,
    ForceFieldPredictor,
    HessianPredictor,
)
from mlip.models.predictors.energy_heads import (
    coulomb_energy_computation_head,
    standard_energy_computation_head,
)
from mlip.typing import ModelParameters, Prediction
from mlip.typing.properties import Properties


@dataclass(frozen=True)
class ForceField:
    """An initialized force field, wrapping a
    :class:`~mlip.models.predictor.ForceFieldPredictor` with parameters.

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
        inference_context: Optional context used by MoE multi-head models to configure
                          graph-level routing globals (e.g. charge, dataset index)
                          at inference time. When set,
                          :meth:`prepare_experts_for_inference` uses it to contract
                          MOE expert parameters into a single-expert model.
    """

    predictor: ForceFieldPredictor
    params: ModelParameters
    inference_context: InferenceContext | None = None

    @classmethod
    def from_mlip_network(
        cls,
        mlip_network: MLIPNetwork,
        required_properties: Properties | None = None,
        seed: int = 42,
        inference_context: InferenceContext | None = None,
    ) -> Self:
        """Initializes a force field from an
        :class:`~mlip.models.mlip_network.MLIPNetwork` instance with random parameters.

        This is an alternative constructor to this dataclass, but the preferred one in
        a typical MLIP pipeline.

        Args:
            mlip_network: The MLIP network to use in this force field.
            required_properties: The properties that are required from this model, which
                                 means these will end up in the `Prediction` objects
                                 that are returned from the `__call__` method. Error
                                 is raised if the `MLIPNetwork` is not able to compute
                                 these required properties. Default is `None` which
                                 means that a default `Properties` object will be
                                 constructed (i.e., energy and forces).
            seed: The initialization seed for the parameters. Default is 42.
            inference_context: Additional context that will be placed on the graph
                during inference, such as dataset_idx.


        Returns:
            The initialized instance of the force field.
        """
        if required_properties is None:
            required_properties = Properties()

        # Check that the selected mlip network is suited to the required properties.
        cls.validate_properties(required_properties, mlip_network.available_properties)

        if inference_context is not None:
            inference_context = inference_context.resolve(mlip_network.dataset_info)

        predictor_cls = cls.get_predictor_class(
            required_properties, mlip_network.config
        )
        energy_head = cls.get_energy_head(mlip_network.config, required_properties)

        predictor = predictor_cls(
            mlip_network=mlip_network,
            required_properties=required_properties,
            energy_head=energy_head,
        )
        return cls.init(
            predictor=predictor, seed=seed, inference_context=inference_context
        )

    @classmethod
    def get_energy_head(
        cls, config: MLIPNetworkConfig, required_properties: Properties
    ) -> Callable[[Graph], Array]:
        """Returns the appropriate energy computation head function.

        A model trained with `use_coulomb_term=True` uses the Coulomb head,
        otherwise the standard energy head is used.

        Args:
            config: The configuration of the model.
            required_properties: The properties required by the predictor.

        Returns:
            The selected function for computing the energy from a graph object.
        """
        if getattr(config, "use_coulomb_term", False):
            return coulomb_energy_computation_head
        return standard_energy_computation_head

    @classmethod
    def validate_properties(
        cls, required_properties: Properties, mlip_available_properties: Properties
    ) -> None:
        """Validates that the mlip network provides all properties required by the user.

        Raises an error if any required property is not available in the mlip network.

        Args:
            required_properties: The set of properties needed.
            mlip_available_properties: The set of properties the mlip network supports.

        Raises:
            ValueError: If a required property is unavailable.
        """
        required_properties_list = required_properties.true_fields()
        mlip_properties_list = mlip_available_properties.true_fields()
        non_compatible_properties = []
        for prop in required_properties_list:
            if prop not in mlip_properties_list:
                non_compatible_properties.append(prop)

        if len(non_compatible_properties) > 0:
            raise ValueError(
                "The mlip network is not compatible with the required properties. "
                "The following properties are not available: "
                f"{non_compatible_properties}."
            )

    @classmethod
    def get_predictor_class(
        cls,
        required_properties: Properties,
        config: MLIPNetworkConfig,
    ) -> type[ForceFieldPredictor]:
        """Returns the appropriate predictor class based on the required properties.

        This method can be used to customize which predictor class is instantiated,
        depending on the properties needed for the force field.

        Args:
            required_properties: The properties required by the predictor.
            config: The configuration of the model.

        Returns:
            The selected predictor class.
        """
        if required_properties.hessian:
            return HessianPredictor
        return ConservativePredictor

    def calculate(self, graph: Graph) -> Graph:
        """Computes a forward pass of the predictor and returns the updated graph.

        See documentation of the
        :meth:`~mlip.models.predictors.predictor.ForceFieldPredictor.__call__` method of
        :class:`~mlip.models.predictors.predictor.ForceFieldPredictor` for more
        information on the returned object.

        Args:
            graph: The input graph.

        Returns:
            The graph updated with the calculated properties.
        """
        if self.inference_context is not None:
            inference_context = self.inference_context.resolve(self.dataset_info)
            graph = apply_inference_context_to_graph(
                graph, inference_context=inference_context
            )

        return self.predictor.apply(self.params, graph)

    def predict(self, graph: Graph) -> Prediction:
        """Computes a forward pass of the predictor and returns predicted properties.

        Args:
            graph: The input graph.

        Returns:
            The predicted properties as a `Prediction` object extracted from the graph.
        """
        graph = self.calculate(graph)
        return graph.to_prediction()

    def __call__(self, graph: Graph) -> Prediction:
        """Predict physical properties of an input graph from current parameters.

        See documentation of the
        :meth:`~mlip.models.predictors.force_field.ForceField.predict` method for more
        information.

        Args:
            graph: The input graph.

        Returns:
            The predicted properties as a `Prediction` object extracted from the graph.
        """
        return self.predict(graph)

    @classmethod
    def init(
        cls,
        predictor: ForceFieldPredictor,
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
        params = predictor.init(
            jax.random.key(seed),
            get_dummy_graph_for_model_init(),
        )
        return cls(
            predictor=predictor, params=params, inference_context=inference_context
        )

    def prepare_experts_for_inference(self) -> Self:
        """Contract MoE expert parameters for a fixed inference context.

        Only models that advertise `is_moe_model` support this operation.
        Routes through the model's MoE router using the context's globals and
        linearly combines expert kernels into standard dense kernels. The
        returned force field has `config.moe=None`, no `expert_kernel`
        parameters, and routing coefficients baked in.

        `load_model_from_zip` and `load_trained_force_field` call
        this method for MoE models when an `inference_context` is provided.
        """
        if not self.predictor.mlip_network.is_moe_model:
            raise ValueError(
                "Cannot prepare experts for inference because this model does not "
                "use mixture-of-experts parameters."
            )
        if self.inference_context is None:
            raise ValueError(
                "Cannot prepare experts for inference without an inference context."
            )
        inference_context = self.inference_context.resolve(self.dataset_info)

        mlip_network, mlip_params = (
            self.predictor.mlip_network.prepare_experts_for_inference(
                self.params["params"]["mlip_network"],
                inference_context,
            )
        )

        root_params = dict(self.params)
        predictor_params = dict(root_params["params"])
        predictor_params["mlip_network"] = mlip_params
        root_params["params"] = predictor_params

        return replace(
            self,
            predictor=replace(self.predictor, mlip_network=mlip_network),
            params=root_params,
            inference_context=inference_context,
        )

    def replace_config(self, **kwargs) -> Self:
        """Returns an updated force field object with updated fields in
        the model config.

        Updates fields in the config according to the passed kwargs.

        Returns:
            An updated force field instance.
        """
        updated_network = replace(
            self.predictor.mlip_network, config=self.config.model_copy(update=kwargs)
        )
        updated_predictor = replace(self.predictor, mlip_network=updated_network)
        return replace(self, predictor=updated_predictor)

    def replace_required_properties(self, required_properties: Properties) -> Self:
        """Returns an updated force field object with updated required properties.

        This function re-evaluates the energy head and predictor class based on the
        new required properties, so be aware that these might have changed for the
        returned force field instance.

        Args:
            required_properties: The required properties to set.

        Returns:
            An updated force field instance.
        """
        predictor_cls = self.get_predictor_class(required_properties, self.config)
        energy_head = self.get_energy_head(self.config, required_properties)

        updated_predictor = predictor_cls(
            mlip_network=self.predictor.mlip_network,
            required_properties=required_properties,
            energy_head=energy_head,
        )
        return replace(self, predictor=updated_predictor)

    def replace_inference_context(self, inference_context: InferenceContext) -> Self:
        """Return a copy of this force field with `inference_context` attached.

        The context is resolved against the model's
        :class:`~mlip.data.dataset_info.DatasetInfo` so callers can pass a
        partial spec (e.g. just `dataset_name`) and have complementary fields
        populated.

        Args:
            inference_context: The context to attach.

        Returns:
            A new force field instance with the resolved context attached.
        """
        resolved = inference_context.resolve(self.dataset_info)
        return replace(self, inference_context=resolved)

    @property
    def cutoff_distance(self) -> float:
        """Cutoff distance in Angstrom the model was built for."""
        dataset_info = self.predictor.mlip_network.dataset_info
        return dataset_info.graph_cutoff_angstrom

    @property
    def long_range_cutoff_distance(self) -> float | None:
        """Long-range cutoff in Angstrom, or `None` if the model has no
        long-range interactions."""
        dataset_info = self.predictor.mlip_network.dataset_info
        return dataset_info.long_range_cutoff_angstrom

    @property
    def allowed_atomic_numbers(self) -> set[int]:
        """Set of atomic numbers supported by the model."""
        return set(self.predictor.mlip_network.dataset_info.allowed_atomic_numbers)

    @property
    def config(self) -> MLIPNetworkConfig:
        """Return configuration of the underlying MLIP model."""
        return self.predictor.mlip_network.config

    @property
    def dataset_info(self) -> DatasetInfo:
        """Return dataset info stored in the MLIP network."""
        return self.predictor.mlip_network.dataset_info

    def __hash__(self):
        """Simple hashing function to allow for jitting `self.__call__`."""
        return id(self)

    def __eq__(self, other):
        """Simple comparison based on IDs to allow for jitting `self.__call__`."""
        return id(other) == id(self)
