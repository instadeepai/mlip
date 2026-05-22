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

import jax.numpy as jnp
import numpy as np
import pytest

from mlip.models.loss.eval_metrics import compute_eval_metrics


def _fully_populated_graph(make_customizable_graph):
    """Build a graph with every optional field present on both nodes and globals."""
    rng = np.random.default_rng(0)
    n_nodes, n_edges = 3, 3
    g = make_customizable_graph(n_nodes, n_edges)
    g = g.replace_nodes(
        forces=rng.random((n_nodes, 3)),
        partial_charges=rng.random((n_nodes,)),
        hessian=rng.random((n_nodes, 4, 3)),
    )
    # `compute_eval_metrics` compares ref.charge vs pred.non_corrected_charge,
    # so keep them equal here for the zero-case sanity test to be exactly zero.
    g = g.replace_globals(
        energy=np.array([1.25]),
        stress=rng.random((1, 3, 3)),
        charge=np.array([1.0]),
        non_corrected_charge=np.array([1.0]),
        dipole_moment=rng.random((1, 3)),
    )
    return g


def test_compute_eval_metrics_survives_when_pred_has_no_charges_but_ref_does(
    make_customizable_graph,
):
    """Reproducer for Christoph's 2026-04-22 NequIP/SPICE2 run failure."""
    ref = _fully_populated_graph(make_customizable_graph)
    pred = ref.replace_nodes(partial_charges=None)
    pred = pred.replace_globals(
        charge=None, non_corrected_charge=None, dipole_moment=None
    )

    metrics = compute_eval_metrics(pred, ref, extended_metrics=True)

    for key in [
        "mae_partial_charges",
        "mse_partial_charges",
        "mae_charge",
        "mse_charge",
        "mae_dipole_moment",
        "mse_dipole_moment",
    ]:
        assert jnp.isnan(metrics[key]), f"{key} should be NaN when pred is missing"

    assert jnp.isfinite(metrics["mae_e"])
    assert jnp.isfinite(metrics["mae_f"])


@pytest.mark.parametrize(
    ("drop_node_fields", "drop_global_fields", "nan_metric_keys"),
    [
        pytest.param(
            (),
            ("energy",),
            ("mae_e", "mse_e", "mae_e_per_atom", "mse_e_per_atom"),
            id="pred-missing-energy",
        ),
        pytest.param(
            ("forces",),
            (),
            ("mae_f", "mse_f"),
            id="pred-missing-forces",
        ),
        pytest.param(
            (),
            ("stress",),
            (
                "mae_stress",
                "mse_stress",
                "mae_stress_per_atom",
                "mse_stress_per_atom",
            ),
            id="pred-missing-stress",
        ),
        pytest.param(
            ("partial_charges",),
            (),
            ("mae_partial_charges", "mse_partial_charges"),
            id="pred-missing-partial-charges",
        ),
        pytest.param(
            (),
            ("non_corrected_charge",),
            (
                "mae_charge",
                "mse_charge",
                "mae_charge_per_atom",
                "mse_charge_per_atom",
            ),
            id="pred-missing-non-corrected-charge",
        ),
        pytest.param(
            (),
            ("dipole_moment",),
            (
                "mae_dipole_moment",
                "mse_dipole_moment",
                "mae_dipole_moment_per_atom",
                "mse_dipole_moment_per_atom",
            ),
            id="pred-missing-dipole-moment",
        ),
    ],
)
def test_compute_eval_metrics_tolerates_any_single_pred_field_missing(
    make_customizable_graph,
    drop_node_fields,
    drop_global_fields,
    nan_metric_keys,
):
    """Dropping any single field on the pred side must leave the metric at its
    NaN default rather than crashing on `Array - None`."""
    ref = _fully_populated_graph(make_customizable_graph)

    pred = ref
    if drop_node_fields:
        pred = pred.replace_nodes(**{f: None for f in drop_node_fields})
    if drop_global_fields:
        pred = pred.replace_globals(**{f: None for f in drop_global_fields})

    metrics = compute_eval_metrics(pred, ref, extended_metrics=True)

    for key in nan_metric_keys:
        assert jnp.isnan(metrics[key]), f"{key} should be NaN when pred is missing"


def test_compute_eval_metrics_pred_predicts_partial_charges_only(
    make_customizable_graph,
):
    """Partial charges, total charge and dipole moment must be independently
    guarded — predicting one but not the others should not crash."""
    ref = _fully_populated_graph(make_customizable_graph)
    pred = ref.replace_globals(
        charge=None, non_corrected_charge=None, dipole_moment=None
    )

    metrics = compute_eval_metrics(pred, ref, extended_metrics=True)

    assert jnp.isfinite(metrics["mae_partial_charges"])
    assert jnp.isnan(metrics["mae_charge"])
    assert jnp.isnan(metrics["mae_dipole_moment"])


def test_compute_eval_metrics_zero_when_pred_matches_ref(make_customizable_graph):
    g = _fully_populated_graph(make_customizable_graph)
    metrics = compute_eval_metrics(g, g, extended_metrics=True)
    for key in [
        "mae_e",
        "mae_f",
        "mae_stress",
        "mae_partial_charges",
        "mae_charge",
        "mae_dipole_moment",
    ]:
        assert np.isclose(float(metrics[key]), 0.0), f"{key} should be zero"


def test_compute_eval_metrics_hessian_zero_when_pred_missing(make_customizable_graph):
    """Hessian intentionally defaults to a zero delta when pred is missing,
    rather than the NaN-default semantics of the other fields. Pin this."""
    ref = _fully_populated_graph(make_customizable_graph)
    pred = ref.replace_nodes(hessian=None)

    metrics = compute_eval_metrics(pred, ref, extended_metrics=True)

    assert float(metrics["mae_hes"]) == 0.0
    assert float(metrics["mse_hes"]) == 0.0


def test_compute_eval_metrics_nan_default_when_both_sides_missing(
    make_customizable_graph,
):
    ref = _fully_populated_graph(make_customizable_graph)
    ref = ref.replace_nodes(partial_charges=None)
    ref = ref.replace_globals(
        charge=None, non_corrected_charge=None, dipole_moment=None
    )
    pred = ref

    metrics = compute_eval_metrics(pred, ref, extended_metrics=True)

    for key in [
        "mae_partial_charges",
        "mae_charge",
        "mae_dipole_moment",
    ]:
        assert jnp.isnan(metrics[key])
