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

from mlip.models.loss.loss import Loss, WeightSchedule
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

DEFAULT_WEIGHTS = {
    "energy": 1.0,
    "forces": 25.0,
    "stress": 1.0,
    "partial_charges": 0.0,
    "charge": 0.0,
    "dipole_moment": 0.0,
    "hessian": 0.0,
}


class MSELoss(Loss):
    """Mean-squared error loss for energy, forces and stress.

    Weights are epoch-dependent. Custom schedules can be passed.
    """

    def __init__(
        self,
        energy_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS["energy"],
        forces_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS["forces"],
        stress_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS["stress"],
        hessian_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS["hessian"],
        partial_charges_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS[
            "partial_charges"
        ],
        charge_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS["charge"],
        dipole_moment_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS[
            "dipole_moment"
        ],
        extended_metrics: bool = False,
    ) -> None:
        """Constructor.

        Args:
            energy_weight_schedule: The schedule function for the energy weight.
                                    Default is a constant weight of 1.
            forces_weight_schedule: The schedule function for the energy weight.
                                    Default is a constant weight of 25.
            stress_weight_schedule: The schedule function for the energy weight.
                                    Default is a constant weight of 0.
            hessian_weight_schedule: The schedule function for the Hessian weight.
                                    Default is a constant weight of 0.
            partial_charges_weight_schedule: The schedule function for the partial
                                    charges weight. Default is a constant weight of 0.
            charge_weight_schedule: The schedule function for the charge weight.
                                    Default is a constant weight of 0.
            dipole_moment_weight_schedule: The schedule function for the dipole moment
                                    weight. Default is a constant weight of 0.
            extended_metrics: Whether to include an extended list of metrics.
                                    Defaults to `False`.
        """
        losses = [
            MSEEnergyLoss(),
            MSEForcesLoss(),
            MSEStressLoss(),
            MSEHessianLoss(),
            MSEPartialChargesLoss(),
            MSEChargeLoss(),
            MSEDipoleMomentLoss(),
        ]
        schedules = [
            energy_weight_schedule,
            forces_weight_schedule,
            stress_weight_schedule,
            hessian_weight_schedule,
            partial_charges_weight_schedule,
            charge_weight_schedule,
            dipole_moment_weight_schedule,
        ]
        super().__init__(losses, schedules, extended_metrics)


class HuberLoss(Loss):
    """Huber loss for energy, forces and stress.

    Weights are epoch-dependent. Custom schedules can be passed.
    """

    def __init__(
        self,
        energy_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS["energy"],
        forces_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS["forces"],
        stress_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS["stress"],
        hessian_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS["hessian"],
        partial_charges_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS[
            "partial_charges"
        ],
        charge_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS["charge"],
        dipole_moment_weight_schedule: WeightSchedule = lambda _: DEFAULT_WEIGHTS[
            "dipole_moment"
        ],
        extended_metrics: bool = False,
    ) -> None:
        """Constructor.

        Args:
            energy_weight_schedule: The schedule function for the energy weight.
                                    Default is a constant weight of 1.
            forces_weight_schedule: The schedule function for the forces weight.
                                    Default is a constant weight of 25.
            stress_weight_schedule: The schedule function for the stress weight.
                                    Default is a constant weight of 0.
            hessian_weight_schedule: The schedule function for the Hessian weight.
                                    Default is a constant weight of 0.
            partial_charges_weight_schedule: The schedule function for the partial
                                    charges weight. Default is a constant weight of 0.
            charge_weight_schedule: The schedule function for the charge weight.
                                    Default is a constant weight of 0.
            dipole_moment_weight_schedule: The schedule function for the dipole moment
                                    weight. Default is a constant weight of 0.
            extended_metrics: Whether to include an extended list of metrics.
                                    Defaults to `False`.
        """
        losses = [
            HuberEnergyLoss(),
            HuberForcesLoss(),
            HuberStressLoss(),
            HuberHessianLoss(),
            HuberPartialChargesLoss(),
            HuberChargeLoss(),
            HuberDipoleMomentLoss(),
        ]
        schedules = [
            energy_weight_schedule,
            forces_weight_schedule,
            stress_weight_schedule,
            hessian_weight_schedule,
            partial_charges_weight_schedule,
            charge_weight_schedule,
            dipole_moment_weight_schedule,
        ]
        super().__init__(losses, schedules, extended_metrics)
