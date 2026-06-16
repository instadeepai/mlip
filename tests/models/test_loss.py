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
import numpy as np
import optax
import pytest

from mlip.graph.batching_helpers import batch_graphs, homogenize_graph_fields
from mlip.models.loss import HuberLoss, Loss, MSELoss
from mlip.models.loss.loss_term import (
    HuberChargeLoss,
    HuberDipoleMomentLoss,
    HuberEnergyLoss,
    HuberForcesLoss,
    HuberHessianLoss,
    HuberPartialChargesLoss,
    HuberStressLoss,
    MSEChargeLoss,
    MSEDipoleMomentLoss,
    MSEEnergyLoss,
    MSEForcesLoss,
    MSEHessianLoss,
    MSEPartialChargesLoss,
    MSEStressLoss,
)


def test_loss_will_be_zero_if_input_output_match(setup_system):
    _, graph = setup_system

    rng = np.random.default_rng(seed=0)
    forces = rng.random(graph.nodes.positions.shape)
    stress = rng.random((1, 3, 3))
    hessian = rng.random((len(graph.nodes.positions), 8, 3))  # (n, R, 3)
    graph = graph.replace_nodes(forces=forces, hessian=hessian)
    graph = graph.replace_globals(energy=np.array(3.2), stress=stress)

    partial_charges = rng.random(graph.nodes.positions.shape[0])
    charge = np.array(1.0)
    non_corrected_charge = charge
    dipole_moment = rng.random((1, 3))
    graph = graph.replace_nodes(partial_charges=partial_charges)
    graph = graph.replace_globals(
        charge=charge,
        non_corrected_charge=non_corrected_charge,
        dipole_moment=dipole_moment,
    )

    for _loss in [
        MSEEnergyLoss(),
        MSEForcesLoss(),
        MSEStressLoss(),
        MSEHessianLoss(),
        MSEPartialChargesLoss(),
        MSEChargeLoss(),
        MSEDipoleMomentLoss(),
        HuberEnergyLoss(),
        HuberForcesLoss(),
        HuberStressLoss(),
        HuberHessianLoss(),
        HuberPartialChargesLoss(),
        HuberChargeLoss(),
        HuberDipoleMomentLoss(),
    ]:
        assert _loss(graph, graph) == pytest.approx(0.0, abs=1e-12)


def test_weight_on_graph_is_respected_in_loss_computation(setup_system):
    _, graph = setup_system

    # Prediction graph
    rng = np.random.default_rng(seed=0)
    forces = rng.random(graph.nodes.positions.shape)
    stress = rng.random((1, 3, 3))
    hessian = rng.random((len(graph.nodes.positions), 8, 3))  # (n, R, 3)
    partial_charges = rng.random(graph.nodes.positions.shape[0])
    charge = np.array(1.0)
    non_corrected_charge = 0.9 * charge
    dipole_moment = rng.random((1, 3))
    pred_graph = graph.replace_nodes(
        forces=forces, partial_charges=partial_charges, hessian=hessian
    )
    pred_graph = pred_graph.replace_globals(
        energy=np.array(3.2),
        stress=stress,
        charge=charge,
        non_corrected_charge=non_corrected_charge,
        dipole_moment=dipole_moment,
    )

    # Reference graph
    rng = np.random.default_rng(seed=1)
    forces = rng.random(graph.nodes.positions.shape)
    stress = rng.random((1, 3, 3))
    hessian = rng.random((len(graph.nodes.positions), 8, 3))  # (n, R, 3)
    partial_charges = rng.random(graph.nodes.positions.shape[0])
    charge = np.array(1.0)
    dipole_moment = rng.random((1, 3))
    ref_graph = graph.replace_nodes(
        forces=forces, partial_charges=partial_charges, hessian=hessian
    )
    ref_graph = ref_graph.replace_globals(
        energy=np.array(2.1), stress=stress, charge=charge, dipole_moment=dipole_moment
    )

    losses_without_weight = []
    for _loss in [
        MSEEnergyLoss(),
        MSEForcesLoss(),
        MSEStressLoss(),
        MSEHessianLoss(),
        HuberEnergyLoss(),
        HuberForcesLoss(),
        HuberStressLoss(),
        HuberHessianLoss(),
        MSEPartialChargesLoss(),
        MSEChargeLoss(),
        MSEDipoleMomentLoss(),
        HuberPartialChargesLoss(),
        HuberChargeLoss(),
        HuberDipoleMomentLoss(),
    ]:
        losses_without_weight.append(_loss(pred_graph, ref_graph))

    # The one in ref graph should be used, hence 2.9
    pred_graph = pred_graph.replace_globals(weight=1.7)
    ref_graph = ref_graph.replace_globals(weight=2.9)

    losses_with_weight = []
    for _loss in [
        MSEEnergyLoss(),
        MSEForcesLoss(),
        MSEStressLoss(),
        MSEHessianLoss(),
        HuberEnergyLoss(),
        HuberForcesLoss(),
        HuberStressLoss(),
        HuberHessianLoss(),
        MSEPartialChargesLoss(),
        MSEChargeLoss(),
        MSEDipoleMomentLoss(),
        HuberPartialChargesLoss(),
        HuberChargeLoss(),
        HuberDipoleMomentLoss(),
    ]:
        losses_with_weight.append(_loss(pred_graph, ref_graph))

    for l_with_weight, l_without_weight in zip(
        losses_with_weight, losses_without_weight
    ):
        assert (l_with_weight / l_without_weight)[0] == pytest.approx(2.9)


def test_energy_losses(setup_system):
    _, graph = setup_system

    pred_graph = graph.replace_globals(energy=np.array(3.2))
    ref_graph = graph.replace_globals(energy=np.array(2.1))

    assert MSEEnergyLoss()(pred_graph, ref_graph)[0] == pytest.approx((1.1 / 10) ** 2)

    expected_huber_loss = optax.losses.huber_loss(3.2 / 10, 2.1 / 10, delta=0.01)
    assert HuberEnergyLoss()(pred_graph, ref_graph)[0] == pytest.approx(
        float(expected_huber_loss)
    )


def test_forces_losses(setup_system):
    _, graph = setup_system

    rng = np.random.default_rng(seed=0)
    forces = rng.random(graph.nodes.positions.shape)
    pred_graph = graph.replace_nodes(forces=forces)
    ref_graph = graph.replace_nodes(forces=forces + 0.3)

    assert MSEForcesLoss()(pred_graph, ref_graph) == pytest.approx(0.3**2)

    # For the Huber loss, this assert is just a signal to detect if anything in the
    # code has changed; the result is not a value easily checkable by hand
    assert HuberForcesLoss()(pred_graph, ref_graph) == pytest.approx(0.00295)


def test_stress_losses(setup_system):
    _, graph = setup_system

    rng = np.random.default_rng(seed=0)
    stress = rng.random((1, 3, 3))
    pred_graph = graph.replace_globals(stress=stress)
    ref_graph = graph.replace_globals(stress=stress + 0.11)

    assert MSEStressLoss()(pred_graph, ref_graph) == pytest.approx(0.11**2)

    expected_huber_loss = optax.losses.huber_loss(
        stress,
        stress + 0.11,
        delta=0.01,
    ).mean()
    assert HuberStressLoss()(pred_graph, ref_graph)[0] == pytest.approx(
        float(expected_huber_loss)
    )


def test_hessian_losses(setup_system):
    _, graph = setup_system
    rng = np.random.default_rng(seed=0)
    forces = rng.random(graph.nodes.positions.shape)
    hessian = rng.random((len(graph.nodes.positions), 8, 3))  # (n, R, 3)
    pred_graph = graph.replace_nodes(forces=forces, hessian=hessian)
    ref_graph = graph.replace_nodes(forces=forces + 0.3, hessian=hessian + 0.6)

    assert MSEHessianLoss()(pred_graph, ref_graph) == pytest.approx(0.36)
    assert HuberHessianLoss()(pred_graph, ref_graph) == pytest.approx(0.00595)


def test_charge_related_losses(setup_system):
    _, graph = setup_system

    rng = np.random.default_rng(seed=0)
    partial_charges = rng.random(graph.nodes.positions.shape[0])
    charge = np.array(1.0)
    non_corrected_charge = charge - 0.1
    dipole_moment = rng.random((1, 3))
    ref_graph = graph.replace_nodes(partial_charges=partial_charges)
    ref_graph = ref_graph.replace_globals(charge=charge, dipole_moment=dipole_moment)
    pred_graph = graph.replace_nodes(partial_charges=partial_charges + 0.1)
    pred_graph = pred_graph.replace_globals(
        charge=charge,
        non_corrected_charge=non_corrected_charge,
        dipole_moment=dipole_moment + 0.1,
    )

    assert MSEPartialChargesLoss()(pred_graph, ref_graph) == pytest.approx(0.1**2)
    assert HuberPartialChargesLoss()(pred_graph, ref_graph) == pytest.approx(0.00095)
    assert MSEChargeLoss()(pred_graph, ref_graph) == pytest.approx(
        0.1**2 / np.sum(graph.n_node)
    )
    assert HuberChargeLoss()(pred_graph, ref_graph) == pytest.approx(9.5e-5)
    assert MSEDipoleMomentLoss()(pred_graph, ref_graph) == pytest.approx(
        0.1**2 / np.sum(graph.n_node)
    )
    assert HuberDipoleMomentLoss()(pred_graph, ref_graph) == pytest.approx(9.5e-5)


def test_combined_losses_work_correctly(setup_system):
    _, graph = setup_system

    # Prediction graph
    rng = np.random.default_rng(seed=0)
    forces = rng.random(graph.nodes.positions.shape)
    stress = rng.random((1, 3, 3))
    hessian = rng.random((len(graph.nodes.positions), 8, 3))  # (n, R, 3)
    pred_graph = graph.replace_nodes(forces=forces, hessian=hessian)
    pred_graph = pred_graph.replace_globals(energy=np.array(3.2), stress=stress)

    # Reference graph
    rng = np.random.default_rng(seed=1)
    forces = rng.random(graph.nodes.positions.shape)
    stress = rng.random((1, 3, 3))
    hessian = rng.random((len(graph.nodes.positions), 8, 3))  # (n, R, 3)
    ref_graph = graph.replace_nodes(forces=forces, hessian=hessian)
    ref_graph = ref_graph.replace_globals(energy=np.array(2.1), stress=stress)

    # First, get the loss values for all the individual losses:
    single_losses = {
        "mse_e": MSEEnergyLoss()(pred_graph, ref_graph),
        "mse_f": MSEForcesLoss()(pred_graph, ref_graph),
        "mse_s": MSEStressLoss()(pred_graph, ref_graph),
        "mse_hes": MSEHessianLoss()(pred_graph, ref_graph),
        "huber_e": HuberEnergyLoss()(pred_graph, ref_graph),
        "huber_f": HuberForcesLoss()(pred_graph, ref_graph),
        "huber_s": HuberStressLoss()(pred_graph, ref_graph),
        "huber_hes": HuberHessianLoss()(pred_graph, ref_graph),
    }

    # For MSELoss
    default_mse_loss = MSELoss(
        lambda _: 1.7,
        lambda _: 2.9,
        lambda _: 0.3,
        lambda _: 0.5,
    )

    for epoch_num in [0, 315]:  # make sure no epoch dependence here
        _loss, metrics = default_mse_loss(pred_graph, ref_graph, epoch_num)
        assert (
            _loss
            == 1.7 * single_losses["mse_e"]
            + 2.9 * single_losses["mse_f"]
            + 0.3 * single_losses["mse_s"]
            + 0.5 * single_losses["mse_hes"]
        )
        assert metrics["loss"] == _loss

    # For HuberLoss
    default_huber_loss = HuberLoss(
        lambda _: 11.7,
        lambda _: 21.9,
        lambda _: 0.9,
        lambda _: 0.7,
    )
    for epoch_num in [0, 11]:  # make sure no epoch dependence here
        _loss, metrics = default_huber_loss(pred_graph, ref_graph, epoch_num)
        assert (
            _loss
            == 11.7 * single_losses["huber_e"]
            + 21.9 * single_losses["huber_f"]
            + 0.9 * single_losses["huber_s"]
            + 0.7 * single_losses["huber_hes"]
        )
        assert metrics["loss"] == _loss

    # For a custom loss
    custom_loss = Loss(
        losses=[
            MSEEnergyLoss(),
            HuberEnergyLoss(),
            MSEForcesLoss(),
            HuberStressLoss(),
            HuberHessianLoss(),
        ],
        schedules=[
            lambda _: 0.7,
            lambda _: 1.4,
            lambda _: 0.4,
            lambda _: 1.5,
            lambda _: 1,
        ],
    )
    for epoch_num in [0, 3]:  # make sure no epoch dependence here
        _loss, metrics = custom_loss(pred_graph, ref_graph, epoch_num)
        assert (
            _loss
            == 0.7 * single_losses["mse_e"]
            + 1.4 * single_losses["huber_e"]
            + 0.4 * single_losses["mse_f"]
            + 1.5 * single_losses["huber_s"]
            + 1 * single_losses["huber_hes"]
        )
        assert metrics["loss"] == _loss


def test_energy_loss_masks_nan_targets(setup_system):
    """NaN in the reference energy means "this sample's dataset did not provide
    energy" — the loss for that sample must be zero, and gradients wrt the
    prediction must be finite (no NaN leakage)."""
    _, graph = setup_system

    pred_graph = graph.replace_globals(energy=np.array([3.2]))
    ref_graph = graph.replace_globals(energy=np.array([np.nan]))

    for _loss in [MSEEnergyLoss(), HuberEnergyLoss()]:
        value = _loss(pred_graph, ref_graph)
        assert np.all(np.asarray(value) == 0.0)

        def loss_of_pred(pred, _loss=_loss):
            return jnp.sum(_loss(pred_graph.replace_globals(energy=pred), ref_graph))

        grad = jax.grad(loss_of_pred)(jnp.array([3.2]))
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.all(np.asarray(grad) == 0.0)


def test_forces_loss_masks_nan_targets(setup_system):
    """NaN in the reference forces means "this sample's dataset did not provide
    forces" — the per-graph loss must be zero, and gradients wrt the prediction
    must be finite."""
    _, graph = setup_system

    rng = np.random.default_rng(seed=0)
    pred_forces = rng.random(graph.nodes.positions.shape)
    ref_forces = np.full_like(pred_forces, np.nan)

    pred_graph = graph.replace_nodes(forces=pred_forces)
    ref_graph = graph.replace_nodes(forces=ref_forces)

    for _loss in [MSEForcesLoss(), HuberForcesLoss()]:
        value = _loss(pred_graph, ref_graph)
        assert np.all(np.asarray(value) == 0.0)

        def loss_of_pred(pred, _loss=_loss):
            return jnp.sum(_loss(pred_graph.replace_nodes(forces=pred), ref_graph))

        grad = jax.grad(loss_of_pred)(jnp.asarray(pred_forces))
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.all(np.asarray(grad) == 0.0)


def test_stress_loss_masks_nan_targets(setup_system):
    """NaN in the reference stress must zero the loss and keep gradients finite."""
    _, graph = setup_system

    rng = np.random.default_rng(seed=0)
    pred_stress = rng.random((1, 3, 3))
    ref_stress = np.full((1, 3, 3), np.nan)

    pred_graph = graph.replace_globals(stress=pred_stress)
    ref_graph = graph.replace_globals(stress=ref_stress)

    for _loss in [MSEStressLoss(), HuberStressLoss()]:
        value = _loss(pred_graph, ref_graph)
        assert np.all(np.asarray(value) == 0.0)

        def loss_of_pred(pred, _loss=_loss):
            return jnp.sum(_loss(pred_graph.replace_globals(stress=pred), ref_graph))

        grad = jax.grad(loss_of_pred)(jnp.asarray(pred_stress))
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.all(np.asarray(grad) == 0.0)


def test_partial_charges_loss_masks_nan_targets(setup_system):
    """NaN in the reference partial charges must zero the loss and keep
    gradients finite."""
    _, graph = setup_system

    rng = np.random.default_rng(seed=0)
    pred_partial_charges = rng.random(graph.nodes.positions.shape[0])
    ref_partial_charges = np.full_like(pred_partial_charges, np.nan)

    pred_graph = graph.replace_nodes(partial_charges=pred_partial_charges)
    ref_graph = graph.replace_nodes(partial_charges=ref_partial_charges)

    for _loss in [MSEPartialChargesLoss(), HuberPartialChargesLoss()]:
        value = _loss(pred_graph, ref_graph)
        assert np.all(np.asarray(value) == 0.0)

        def loss_of_pred(pred, _loss=_loss):
            return jnp.sum(
                _loss(pred_graph.replace_nodes(partial_charges=pred), ref_graph)
            )

        grad = jax.grad(loss_of_pred)(jnp.asarray(pred_partial_charges))
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.all(np.asarray(grad) == 0.0)


def test_charge_loss_masks_nan_targets(setup_system):
    """NaN in the reference total charge must zero the loss and keep
    gradients finite."""
    _, graph = setup_system

    pred_graph = graph.replace_globals(
        charge=np.array([1.0]), non_corrected_charge=np.array([1.1])
    )
    ref_graph = graph.replace_globals(charge=np.array([np.nan]))

    for _loss in [MSEChargeLoss(), HuberChargeLoss()]:
        value = _loss(pred_graph, ref_graph)
        assert np.all(np.asarray(value) == 0.0)

        def loss_of_pred(pred, _loss=_loss):
            return jnp.sum(
                _loss(pred_graph.replace_globals(non_corrected_charge=pred), ref_graph)
            )

        grad = jax.grad(loss_of_pred)(jnp.array([1.1]))
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.all(np.asarray(grad) == 0.0)


def test_dipole_moment_loss_masks_nan_targets(setup_system):
    """NaN in the reference dipole moment must zero the loss and keep
    gradients finite."""
    _, graph = setup_system

    rng = np.random.default_rng(seed=0)
    pred_dipole = rng.random((1, 3))
    ref_dipole = np.full((1, 3), np.nan)

    pred_graph = graph.replace_globals(dipole_moment=pred_dipole)
    ref_graph = graph.replace_globals(dipole_moment=ref_dipole)

    for _loss in [MSEDipoleMomentLoss(), HuberDipoleMomentLoss()]:
        value = _loss(pred_graph, ref_graph)
        assert np.all(np.asarray(value) == 0.0)

        def loss_of_pred(pred, _loss=_loss):
            return jnp.sum(
                _loss(pred_graph.replace_globals(dipole_moment=pred), ref_graph)
            )

        grad = jax.grad(loss_of_pred)(jnp.asarray(pred_dipole))
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.all(np.asarray(grad) == 0.0)


@pytest.mark.parametrize(
    "loss_cls", [MSEStressLoss, HuberStressLoss], ids=["mse", "huber"]
)
def test_nan_padded_batch_loss_matches_single_graph_loss(
    make_customizable_graph, loss_cls
):
    """End-to-end: after homogenization + batching, a NaN-padded sample must
    contribute zero to the stress loss, so the batch loss equals the
    single-graph loss on just the graph that has stress. Parameterised over
    MSE and Huber so both double-`jnp.where` maskings are exercised."""
    rng = np.random.default_rng(seed=0)
    pred_stress = rng.random((1, 3, 3))
    ref_stress = rng.random((1, 3, 3))

    g_with_pred = make_customizable_graph(3, 3).replace_globals(stress=pred_stress)
    g_with_ref = make_customizable_graph(3, 3).replace_globals(stress=ref_stress)

    # Second graph has no stress in either pred or ref — it will be
    # NaN-padded by homogenization and must not contribute.
    g_no_pred = make_customizable_graph(2, 2).replace_globals(stress=None)
    g_no_ref = make_customizable_graph(2, 2).replace_globals(stress=None)

    pred_batch = batch_graphs(homogenize_graph_fields([g_with_pred, g_no_pred]))
    ref_batch = batch_graphs(homogenize_graph_fields([g_with_ref, g_no_ref]))

    assert np.any(np.isnan(np.asarray(ref_batch.globals.stress)))

    batch_loss = np.asarray(loss_cls()(pred_batch, ref_batch))
    single_loss = np.asarray(loss_cls()(g_with_pred, g_with_ref))

    # First graph loss unchanged, second graph loss exactly zero (not NaN).
    assert np.isclose(batch_loss[0], single_loss[0])
    assert batch_loss[1] == 0.0


@pytest.mark.parametrize(
    "loss_cls", [MSEPartialChargesLoss, HuberPartialChargesLoss], ids=["mse", "huber"]
)
def test_nan_padded_batch_partial_charges_loss(make_customizable_graph, loss_cls):
    """Same end-to-end shape as the stress test, for the per-node
    `partial_charges` field."""
    rng = np.random.default_rng(seed=0)
    pred_pc = rng.random((3,))
    ref_pc = rng.random((3,))

    g_with_pred = make_customizable_graph(3, 3).replace_nodes(partial_charges=pred_pc)
    g_with_ref = make_customizable_graph(3, 3).replace_nodes(partial_charges=ref_pc)

    g_no_pred = make_customizable_graph(2, 2).replace_nodes(partial_charges=None)
    g_no_ref = make_customizable_graph(2, 2).replace_nodes(partial_charges=None)

    pred_batch = batch_graphs(homogenize_graph_fields([g_with_pred, g_no_pred]))
    ref_batch = batch_graphs(homogenize_graph_fields([g_with_ref, g_no_ref]))

    assert np.any(np.isnan(np.asarray(ref_batch.nodes.partial_charges)))

    batch_loss = np.asarray(loss_cls()(pred_batch, ref_batch))
    single_loss = np.asarray(loss_cls()(g_with_pred, g_with_ref))

    assert np.isclose(batch_loss[0], single_loss[0])
    assert batch_loss[1] == 0.0


@pytest.mark.parametrize(
    "loss_cls", [MSEChargeLoss, HuberChargeLoss], ids=["mse", "huber"]
)
def test_nan_padded_batch_charge_loss(make_customizable_graph, loss_cls):
    """Per-graph scalar: NaN-padded graphs must not poison the charge loss."""
    pred_charge = np.array([1.1])
    ref_charge = np.array([1.0])

    g_with_pred = make_customizable_graph(3, 3).replace_globals(
        charge=pred_charge, non_corrected_charge=pred_charge
    )
    g_with_ref = make_customizable_graph(3, 3).replace_globals(charge=ref_charge)

    g_no_pred = make_customizable_graph(2, 2).replace_globals(
        charge=None, non_corrected_charge=None
    )
    g_no_ref = make_customizable_graph(2, 2).replace_globals(charge=None)

    # Pred must carry `non_corrected_charge` on the batched graph — homogenize
    # only fills Prediction-targeted fields, not internal pred-only globals.
    g_no_pred = g_no_pred.replace_globals(
        non_corrected_charge=np.array([np.nan]),
    )
    g_with_pred = g_with_pred.replace_globals(non_corrected_charge=pred_charge)

    pred_batch = batch_graphs(homogenize_graph_fields([g_with_pred, g_no_pred]))
    ref_batch = batch_graphs(homogenize_graph_fields([g_with_ref, g_no_ref]))

    assert np.any(np.isnan(np.asarray(ref_batch.globals.charge)))

    batch_loss = np.asarray(loss_cls()(pred_batch, ref_batch))
    single_loss = np.asarray(loss_cls()(g_with_pred, g_with_ref))

    assert np.isclose(batch_loss[0], single_loss[0])
    assert batch_loss[1] == 0.0


@pytest.mark.parametrize(
    "loss_cls", [MSEDipoleMomentLoss, HuberDipoleMomentLoss], ids=["mse", "huber"]
)
def test_nan_padded_batch_dipole_moment_loss(make_customizable_graph, loss_cls):
    """Per-graph 3-vector: NaN-padded graphs must not poison the dipole loss."""
    rng = np.random.default_rng(seed=0)
    pred_dm = rng.random((1, 3))
    ref_dm = rng.random((1, 3))

    g_with_pred = make_customizable_graph(3, 3).replace_globals(dipole_moment=pred_dm)
    g_with_ref = make_customizable_graph(3, 3).replace_globals(dipole_moment=ref_dm)

    g_no_pred = make_customizable_graph(2, 2).replace_globals(dipole_moment=None)
    g_no_ref = make_customizable_graph(2, 2).replace_globals(dipole_moment=None)

    pred_batch = batch_graphs(homogenize_graph_fields([g_with_pred, g_no_pred]))
    ref_batch = batch_graphs(homogenize_graph_fields([g_with_ref, g_no_ref]))

    assert np.any(np.isnan(np.asarray(ref_batch.globals.dipole_moment)))

    batch_loss = np.asarray(loss_cls()(pred_batch, ref_batch))
    single_loss = np.asarray(loss_cls()(g_with_pred, g_with_ref))

    assert np.isclose(batch_loss[0], single_loss[0])
    assert batch_loss[1] == 0.0
