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

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import pytest

from mlip.data.helpers.dummy_init_graph import get_dummy_graph_for_model_init
from mlip.graph.batching_helpers import batch_graphs
from mlip.models.esen.config import EsenConfig, EsenMoEConfig, MoERoutingGlobal
from mlip.models.esen.network import Esen
from mlip.models.force_field import ForceField
from mlip.models.inference_context import InferenceContext
from mlip.models.model_io import load_model_from_zip, save_model_to_zip
from mlip.typing.properties import Properties


def _contains_key(tree, key: str) -> bool:
    if isinstance(tree, Mapping):
        if key in tree:
            return True
        return any(_contains_key(value, key) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return any(_contains_key(value, key) for value in tree)
    return False


@pytest.fixture()
def make_moe_force_field(dataset_info):
    """Factory fixture: build a ForceField with MoE-enabled eSEN."""

    def _build(
        routing_globals: tuple[MoERoutingGlobal, ...] = ("charge",),
        num_experts: int = 4,
        **moe_kwargs,
    ) -> ForceField:
        model = Esen(
            EsenConfig(
                moe=EsenMoEConfig(
                    num_experts=num_experts,
                    routing_globals=routing_globals,
                    **moe_kwargs,
                )
            ),
            dataset_info,
        )
        return ForceField.from_mlip_network(
            model,
            required_properties=Properties(stress=True),
            seed=0,
        )

    return _build


@pytest.fixture()
def make_moe_graph():
    """Factory fixture: build a dummy graph with routing globals populated."""

    def _build(
        charge: int = 0,
        spin_multiplicity: int = 1,
        dataset_idx: int = 0,
    ):
        return get_dummy_graph_for_model_init().replace_globals(
            charge=jnp.array([charge], dtype=jnp.int32),
            spin_multiplicity=jnp.array([spin_multiplicity], dtype=jnp.int32),
            dataset_idx=jnp.array([dataset_idx], dtype=jnp.int32),
        )

    return _build


@pytest.fixture()
def moe_force_field(make_moe_force_field) -> ForceField:
    return make_moe_force_field()


@pytest.fixture()
def moe_graph(make_moe_graph):
    return make_moe_graph(charge=1)


@pytest.fixture()
def get_moe_coefficients():
    """Returns a helper that extracts softmax expert weights for a graph."""

    def _extract(force_field: ForceField, graph):
        mlip_network = force_field.predictor.mlip_network
        mlip_network_params = force_field.params["params"]["mlip_network"]
        return mlip_network.apply(
            {"params": mlip_network_params},
            graph,
            method=type(mlip_network).get_moe_coefficients,
        )

    return _extract


def test_esen_moe_runs_on_dummy_graph(make_moe_force_field, make_moe_graph):
    force_field = make_moe_force_field(
        routing_globals=("charge", "spin_multiplicity", "dataset_idx"),
    )

    prediction = jax.jit(force_field)(
        make_moe_graph(charge=1, spin_multiplicity=2, dataset_idx=0)
    )

    assert jnp.all(jnp.isfinite(jnp.asarray(prediction.energy)))
    assert jnp.all(jnp.isfinite(jnp.asarray(prediction.forces)))
    assert jnp.all(jnp.isfinite(jnp.asarray(prediction.stress)))


@pytest.mark.parametrize("embedding_type", ["pos_emb", "lin_emb", "rand_emb"])
def test_esen_moe_runs_for_all_embedding_types(
    make_moe_force_field, make_moe_graph, get_moe_coefficients, embedding_type
):
    force_field = make_moe_force_field(embedding_type=embedding_type)
    graph = make_moe_graph(charge=1)

    prediction = force_field(graph)
    coeffs = get_moe_coefficients(force_field, graph)
    num_experts = force_field.predictor.mlip_network.config.moe.num_experts

    assert coeffs.shape == (1, num_experts)
    assert jnp.allclose(jnp.sum(coeffs, axis=-1), jnp.ones((1,)), atol=1e-6)
    assert jnp.all(jnp.isfinite(jnp.asarray(prediction.energy)))
    assert jnp.all(jnp.isfinite(jnp.asarray(prediction.forces)))


def test_moe_config_rejects_unsupported_routing_globals():
    with pytest.raises(ValueError, match="Input should be"):
        EsenMoEConfig(
            num_experts=2,
            routing_globals=("charge", "not_a_supported_global"),
        )


def test_moe_coefficients_are_well_formed_on_dummy_graph(
    moe_force_field, moe_graph, get_moe_coefficients
):
    num_experts = moe_force_field.predictor.mlip_network.config.moe.num_experts

    coeffs = get_moe_coefficients(moe_force_field, moe_graph)

    assert coeffs.shape == (1, num_experts)
    assert jnp.allclose(jnp.sum(coeffs, axis=-1), jnp.ones((1,)), atol=1e-6)
    assert jnp.all(coeffs >= 0.0)
    assert jnp.all(coeffs <= 1.0)


def test_esen_moe_runs_on_batched_dummy_graphs(
    moe_force_field, make_moe_graph, get_moe_coefficients
):
    batched_graph = batch_graphs([
        make_moe_graph(charge=0),
        make_moe_graph(charge=1),
        make_moe_graph(charge=-1),
    ])
    num_experts = moe_force_field.predictor.mlip_network.config.moe.num_experts

    prediction = jax.jit(moe_force_field)(batched_graph)
    coeffs = get_moe_coefficients(moe_force_field, batched_graph)

    assert jnp.asarray(prediction.energy).shape == (3,)
    assert jnp.asarray(prediction.forces).shape == (3, 3)
    assert jnp.asarray(prediction.stress).shape == (3, 3, 3)
    assert jnp.all(jnp.isfinite(jnp.asarray(prediction.energy)))
    assert jnp.all(jnp.isfinite(jnp.asarray(prediction.forces)))
    assert jnp.all(jnp.isfinite(jnp.asarray(prediction.stress)))
    assert coeffs.shape == (3, num_experts)
    assert jnp.allclose(jnp.sum(coeffs, axis=-1), jnp.ones((3,)), atol=1e-6)


def test_moe_coefficients_change_across_graphs_in_batch(
    moe_force_field, make_moe_graph, get_moe_coefficients
):
    batched_graph = batch_graphs([
        make_moe_graph(charge=-100),
        make_moe_graph(charge=100),
    ])

    coeffs = get_moe_coefficients(moe_force_field, batched_graph)

    assert jnp.max(jnp.abs(coeffs[0] - coeffs[1])) > 1e-3


def test_prepare_experts_for_inference_contracts_moe_model(moe_force_field):
    prepared_force_field = ForceField(
        moe_force_field.predictor,
        moe_force_field.params,
        inference_context=InferenceContext(charge=1),
    ).prepare_experts_for_inference()

    assert isinstance(prepared_force_field.predictor.mlip_network, Esen)
    assert prepared_force_field.predictor.mlip_network.config.moe is None
    assert not _contains_key(
        prepared_force_field.params["params"]["mlip_network"],
        "expert_kernel",
    )


def test_prepare_experts_for_inference_rejects_missing_context(moe_force_field):
    with pytest.raises(ValueError, match="without an inference context"):
        moe_force_field.prepare_experts_for_inference()


def test_in_script_flow_matches_loaded_path(moe_force_field, tmp_path):
    """In-script `replace_inference_context().prepare_experts_for_inference()`
    should produce the same contracted force field as the load-from-zip path.
    """
    in_script_force_field = moe_force_field.replace_inference_context(
        InferenceContext(charge=1)
    ).prepare_experts_for_inference()

    filepath = tmp_path / "moe_model.zip"
    save_model_to_zip(filepath, moe_force_field)
    loaded_force_field = load_model_from_zip(
        Esen,
        filepath,
        inference_context=InferenceContext(charge=1),
    )

    graph = get_dummy_graph_for_model_init()
    in_script_pred = in_script_force_field(graph)
    loaded_pred = loaded_force_field(graph)

    assert jnp.allclose(
        jnp.asarray(in_script_pred.energy),
        jnp.asarray(loaded_pred.energy),
        atol=1e-6,
    )
    assert jnp.allclose(
        jnp.asarray(in_script_pred.forces),
        jnp.asarray(loaded_pred.forces),
        atol=1e-6,
    )


def test_load_model_from_zip_with_context_contracts_moe_model(
    moe_force_field,
    tmp_path,
):
    filepath = tmp_path / "moe_model.zip"

    save_model_to_zip(filepath, moe_force_field)
    loaded_force_field = load_model_from_zip(
        Esen,
        filepath,
        inference_context=InferenceContext(charge=1),
    )

    assert isinstance(loaded_force_field.predictor.mlip_network, Esen)
    assert loaded_force_field.predictor.mlip_network.config.moe is None
    assert loaded_force_field.inference_context == InferenceContext(charge=1)
    assert not _contains_key(
        loaded_force_field.params["params"]["mlip_network"],
        "expert_kernel",
    )


def test_prepare_experts_for_inference_matches_uncontracted_output(
    moe_force_field, moe_graph
):
    expected = moe_force_field(moe_graph)
    prepared_force_field = ForceField(
        moe_force_field.predictor,
        moe_force_field.params,
        inference_context=InferenceContext(charge=1),
    ).prepare_experts_for_inference()
    actual = prepared_force_field(get_dummy_graph_for_model_init())

    assert jnp.allclose(
        jnp.asarray(expected.energy),
        jnp.asarray(actual.energy),
        atol=1e-6,
    )
    assert jnp.allclose(
        jnp.asarray(expected.forces),
        jnp.asarray(actual.forces),
        atol=1e-6,
    )
    assert jnp.allclose(
        jnp.asarray(expected.stress),
        jnp.asarray(actual.stress),
        atol=1e-6,
    )


def test_esen_moe_jit_matches_eager_output(moe_force_field, moe_graph):
    eager = moe_force_field(moe_graph)
    jitted = jax.jit(moe_force_field)(moe_graph)

    assert jnp.allclose(
        jnp.asarray(eager.energy),
        jnp.asarray(jitted.energy),
        atol=1e-6,
    )
    assert jnp.allclose(
        jnp.asarray(eager.forces),
        jnp.asarray(jitted.forces),
        atol=1e-6,
    )
    assert jnp.allclose(
        jnp.asarray(eager.stress),
        jnp.asarray(jitted.stress),
        atol=1e-6,
    )


def test_esen_moe_jit_cache_stable_across_calls(moe_force_field, moe_graph):
    jitted_force_field = jax.jit(moe_force_field)
    first = jitted_force_field(moe_graph)
    jax.block_until_ready(jnp.asarray(first.energy))
    cache_size = jitted_force_field._cache_size()

    second = jitted_force_field(moe_graph)
    jax.block_until_ready(jnp.asarray(second.energy))

    assert jitted_force_field._cache_size() == cache_size


def test_prepare_experts_for_inference_matches_uncontracted_output_on_batched_graphs(
    moe_force_field, make_moe_graph
):
    raw_graph = batch_graphs([
        make_moe_graph(charge=1),
        make_moe_graph(charge=1),
    ])
    prepared_graph = batch_graphs([
        get_dummy_graph_for_model_init(),
        get_dummy_graph_for_model_init(),
    ])

    expected = jax.jit(moe_force_field)(raw_graph)
    prepared_force_field = ForceField(
        moe_force_field.predictor,
        moe_force_field.params,
        inference_context=InferenceContext(charge=1),
    ).prepare_experts_for_inference()
    actual = jax.jit(prepared_force_field)(prepared_graph)

    assert jnp.allclose(
        jnp.asarray(expected.energy),
        jnp.asarray(actual.energy),
        atol=1e-6,
    )
    assert jnp.allclose(
        jnp.asarray(expected.forces),
        jnp.asarray(actual.forces),
        atol=1e-6,
    )
    assert jnp.allclose(
        jnp.asarray(expected.stress),
        jnp.asarray(actual.stress),
        atol=1e-6,
    )
