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

import jax
import jax.numpy as jnp
import pytest

from mlip.data.dataset_info import DatasetInfo
from mlip.graph.batching_helpers import batch_graphs, pad_with_graphs
from mlip.models.blocks import (
    PERIODIC_TABLE_SIZE,
    SPECIES_PLACEHOLDER,
    AtomicEnergiesBlock,
    ChargeIndexAssignmentBlock,
    MaskPaddedNodeOutputsBlock,
    RadialEmbeddingBlock,
    SpeciesAssignmentBlock,
)
from mlip.models.options import RadialBasis, RadialEnvelope

UNKNOWN_ATOMIC_NUMBER = 22


@pytest.fixture(scope="module")
def dataset_info_hco():
    return DatasetInfo(
        atomic_energies_map={6: -4.0, 1: -0.5, 8: -2.0},
        total_charge_set={1, 0, -1},
        graph_cutoff_angstrom=2.0,
    )


def test_species_assignment_block(setup_system, dataset_info_hco):
    _, graph = setup_system
    # Padding adds 0s at end of atomic_numbers
    atomic_numbers = jnp.array([1, 6, 8, 1, 6, 6, 0])
    graph = graph.replace_nodes(atomic_numbers=atomic_numbers)

    updated_graph = SpeciesAssignmentBlock(dataset_info_hco)(graph)
    atomic_species = updated_graph.nodes.features["species"]
    assert jnp.all(atomic_species == jnp.array([0, 1, 2, 0, 1, 1, SPECIES_PLACEHOLDER]))


def test_charge_index_assignment_block(setup_system, dataset_info_hco):
    _, graph = setup_system
    n_nodes = graph.nodes.positions.shape[0]
    charge_array = jnp.array([1.0])
    charge_unseen_array = jnp.array([-2.0])
    graph = graph.replace_globals(charge=charge_array)
    graph_unseen = graph.replace_globals(charge=charge_unseen_array)
    idx_block: ChargeIndexAssignmentBlock = ChargeIndexAssignmentBlock(dataset_info_hco)
    updated_graph = idx_block(graph)

    charge_indices = updated_graph.nodes.features["charge_indices"]
    assert jnp.all(charge_indices == jnp.array([3] * n_nodes))

    # test that we are throwing an error when we have unseen charges
    with pytest.raises(ValueError):
        _ = idx_block(graph_unseen)


def test_atomic_energies_block_raises_key_error(setup_system, dataset_info):
    """Check that we catch the key error when features['energy'] does not exist."""
    _, graph = setup_system
    with pytest.raises(KeyError):
        graph = AtomicEnergiesBlock(dataset_info).apply({}, graph)


@pytest.mark.parametrize("skip_addition", [True, False])
@pytest.mark.parametrize("in_bounds", [True, False])
def test_atomic_energies_block(setup_system, dataset_info, in_bounds, skip_addition):
    """Check atomic energy lookup works, and returns NaN when a number is OOB.

    We don't really check intermediate 'holes' in this test, however note that
    it is important to pad the atomic energies lookup table for its last value
    to be NaN, as JAX truncates indices and would return the last atomic energy
    when the atomic number is OOB.
    """
    atomic_energies_map = dataset_info.atomic_energies_map
    _, graph = setup_system
    n_node = graph.nodes.positions.shape[0]
    graph = graph.update_node_features(energy=jnp.zeros(n_node))
    expect = jnp.array([
        atomic_energies_map[int(z)] for z in graph.nodes.atomic_numbers
    ])
    if skip_addition:
        expect = jnp.zeros(n_node)

    if not in_bounds:
        numbers_oob = graph.nodes.atomic_numbers.copy()
        numbers_oob[-1] = UNKNOWN_ATOMIC_NUMBER
        graph = graph.replace_nodes(atomic_numbers=numbers_oob)
        expect = expect.at[-1].set(jnp.nan)

    graph = AtomicEnergiesBlock(
        dataset_info, skip_atomic_energies_addition=skip_addition
    ).apply({}, graph)
    result = graph.nodes.features["energy"]

    if in_bounds:
        assert jnp.allclose(expect, result)
    else:
        assert jnp.allclose(expect[:-1], result[:-1])
        if not skip_addition:
            assert jnp.isnan(result[-1])


def test_atomic_energies_block_multi_dataset(setup_system, multi_head_dataset_info):
    """Check that per-dataset e0s are selected via dataset_idx."""
    ds_info = multi_head_dataset_info
    e0_map_0, e0_map_1 = ds_info.atomic_energies_map

    _, graph = setup_system
    n_node = graph.n_node[0]
    graph = graph.update_node_features(energy=jnp.zeros(n_node))

    # Single graph from dataset 0
    graph_d0 = graph.replace_globals(dataset_idx=jnp.array([0]))
    result_d0 = (
        AtomicEnergiesBlock(ds_info).apply({}, graph_d0).nodes.features["energy"]
    )
    expect_d0 = jnp.array([e0_map_0[int(z)] for z in graph.nodes.atomic_numbers])
    assert jnp.allclose(result_d0, expect_d0)

    # Single graph from dataset 1
    graph_d1 = graph.replace_globals(dataset_idx=jnp.array([1]))
    result_d1 = (
        AtomicEnergiesBlock(ds_info).apply({}, graph_d1).nodes.features["energy"]
    )
    expect_d1 = jnp.array([e0_map_1[int(z)] for z in graph.nodes.atomic_numbers])
    assert jnp.allclose(result_d1, expect_d1)

    # The two datasets should give different results
    assert not jnp.allclose(result_d0, result_d1)


def test_atomic_energies_block_multi_dataset_batched(
    setup_system, salt_graph, multi_head_dataset_info
):
    """Check correct e0 selection in a batch with mixed dataset_idx."""
    ds_info = multi_head_dataset_info
    e0_map_0, e0_map_1 = ds_info.atomic_energies_map

    _, graph_a = setup_system  # dimethyl sulfoxide
    graph_b = salt_graph  # NaCl

    n_a = graph_a.n_node[0]
    n_b = graph_b.n_node[0]

    # Assign graph_a -> dataset 0, graph_b -> dataset 1
    graph_a = graph_a.update_node_features(energy=jnp.zeros(n_a))
    graph_a = graph_a.replace_globals(dataset_idx=jnp.array([0]))
    graph_b = graph_b.update_node_features(energy=jnp.zeros(n_b))
    graph_b = graph_b.replace_globals(dataset_idx=jnp.array([1]))

    batched = batch_graphs([graph_a, graph_b])
    result = AtomicEnergiesBlock(ds_info).apply({}, batched).nodes.features["energy"]

    # First n_a nodes should use map 0, next n_b nodes should use map 1
    expect_a = jnp.array([e0_map_0[int(z)] for z in graph_a.nodes.atomic_numbers])
    expect_b = jnp.array([e0_map_1[int(z)] for z in graph_b.nodes.atomic_numbers])
    assert jnp.allclose(result[:n_a], expect_a)
    assert jnp.allclose(result[n_a : n_a + n_b], expect_b)


def test_atomic_energies_block_learnable(setup_system, dataset_info):
    """Check that learnable E0s are initialized properly and read from parameters."""
    block = AtomicEnergiesBlock(dataset_info, learnable=True)

    _, graph = setup_system
    graph = graph.replace_nodes(
        features={"energy": jnp.zeros(graph.nodes.positions.shape[0])}
    )

    atomic_energies_map = dataset_info.atomic_energies_map

    # Check initial parameters
    params = block.init(jax.random.key(123), graph)
    for z, e0 in atomic_energies_map.items():
        assert params["params"]["atomic_energies"][z] == e0

    # Check parameters are used when passed from the outside
    graph_out = block.apply(
        {"params": {"atomic_energies": jnp.ones(PERIODIC_TABLE_SIZE)}}, graph
    )
    result = graph_out.nodes.features["energy"]
    expect = jnp.ones(graph.nodes.positions.shape[0])
    assert jnp.allclose(expect, result)


def test_shift_and_rescale_works_inside_atomic_energies_block(
    setup_system, dataset_info
):
    """Check that the shifting and rescaling that is optionally applied in the
    `AtomicEnergiesBlock` works correctly."""
    dataset_info = dataset_info.model_copy(
        update={
            "scaling_mean": 1.5,
            "scaling_stdev": 0.1,
            "atomic_energies_map": {
                k: 0.0 for k in dataset_info.allowed_atomic_numbers
            },
        }
    )

    _, graph = setup_system
    n_node = graph.nodes.positions.shape[0]
    graph = graph.update_node_features(energy=jnp.full(n_node, 2.0))

    graph = AtomicEnergiesBlock(dataset_info).apply({}, graph)
    result = graph.nodes.features["energy"]

    assert result.shape == (n_node,)
    for val in result:
        assert val == 1.7


@pytest.mark.parametrize(
    "radial_basis",
    [
        RadialBasis.GAUSS,
        RadialBasis.EXPNORM,
        RadialBasis.BESSEL,
    ],
)
@pytest.mark.parametrize("learnable", [True, False])
@pytest.mark.parametrize(
    "radial_envelope",
    [
        None,
        RadialEnvelope.COSINE_CUTOFF,
        RadialEnvelope.POLYNOMIAL,
        RadialEnvelope.SOFT,
    ],
)
@pytest.mark.parametrize("return_as_irreps", [True, False])
def test_radial_embedding_block_shape_and_cutoff(
    radial_basis,
    learnable,
    radial_envelope,
    return_as_irreps,
):
    key = jax.random.PRNGKey(0)

    num_edges = 8
    num_rbf = 6
    cutoff = 5.0

    # Distances: mix of below and above cutoff
    distances = jnp.array(
        [0.1, 1.0, 2.5, cutoff, cutoff + 1.0, 10.0, 0.0, cutoff + 0.5],
        dtype=jnp.float32,
    )

    model = RadialEmbeddingBlock(
        radial_basis=radial_basis,
        num_rbf=num_rbf,
        graph_cutoff_angstrom=cutoff,
        learnable=learnable,
        radial_envelope=radial_envelope,
        return_as_irreps=return_as_irreps,
    )

    params = model.init(key, distances)
    output = model.apply(params, distances)

    # Handle IrrepsArray case
    if return_as_irreps:
        output_array = output.array
    else:
        output_array = output

    # (1) Shape check
    assert output_array.shape == (num_edges, num_rbf)

    # (2) Values should be zero where distance >= cutoff OR distance == 0
    enforces_cutoff = radial_basis in [RadialBasis.EXPNORM, RadialBasis.BESSEL]
    if enforces_cutoff:
        mask = (distances >= cutoff) | (distances == 0.0)
        mask = mask[:, None]  # (8, 1)

        # Compare only masked positions
        masked_values = jnp.where(mask, output_array, 0.0)

        assert jnp.allclose(masked_values, 0.0, atol=1e-6)


def test_mask_padded_node_outputs_handles_multiple_features(setup_system):
    _, graph = setup_system
    graph = pad_with_graphs(graph, n_node=15, n_edge=70, n_graph=2)
    real_n = int(graph.n_node[0])
    n_total = graph.nodes.positions.shape[0]

    energy = jnp.ones(n_total, dtype=jnp.float32)
    charges = jnp.full(n_total, 2.0, dtype=jnp.float32)
    graph = graph.update_node_features(energy=energy, partial_charges=charges)

    block = MaskPaddedNodeOutputsBlock(feature_names=("energy", "partial_charges"))
    out = block.apply({}, graph)

    for name, original in [("energy", energy), ("partial_charges", charges)]:
        result = out.nodes.features[name]
        assert jnp.all(result[:real_n] == original[:real_n])
        assert jnp.all(result[real_n:] == 0.0)


def test_mask_padded_node_outputs_under_jit(setup_system):
    _, graph = setup_system
    graph = pad_with_graphs(graph, n_node=15, n_edge=70, n_graph=2)
    real_n = int(graph.n_node[0])

    energy = jnp.arange(graph.nodes.positions.shape[0], dtype=jnp.float32) + 1.0
    graph = graph.update_node_features(energy=energy)

    block = MaskPaddedNodeOutputsBlock(feature_names=("energy",))

    @jax.jit
    def run(g):
        return block.apply({}, g).nodes.features["energy"]

    masked_energy = run(graph)
    assert jnp.all(masked_energy[:real_n] == energy[:real_n])
    assert jnp.all(masked_energy[real_n:] == 0.0)
