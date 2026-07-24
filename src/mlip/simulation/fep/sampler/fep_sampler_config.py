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

from pathlib import Path

import pydantic
from jax.typing import ArrayLike

from mlip.simulation.configs.jax_md_config import JaxMDSimulationConfig
from mlip.simulation.fep.models.repulsive_potentials import (
    SoftcoreLennardJonesPotential,
    SoftcoreRepulsivePotential,
)


class FEPSimulationSamplerConfig(pydantic.BaseModel):
    """Configuration for FEP simulations managed by FEPSimulationSampler.

    Attributes:
        simulation_config: Config of the simulation to run for each lambda.
        alchemical_atom_indices: Indices of the atoms to treat as alchemical.
            Edges between the alchemical atoms and all other atoms are gradually
            switched off between adjacent simulations.
        lambda_edge_values: Alchemical edge weight to use for each simulation
            (1.0 = fully connected, 0.0 = fully decoupled). Must be the same length
            as `lambda_repulsion_values`, one entry per lambda window.
        lambda_repulsion_values: Repulsive potential weight between alchemical and
            non-alchemical atoms to use for each simulation. Must be the same length
            as `lambda_edge_values`, one entry per lambda window.
        use_alchemical_mlip: If True (default), uses an alchemical equivalent of the
            input force field, which weights alchemical edges in the message-passing
            equations. If False, uses the base model equations, and computes the
            alchemical potential by summing predictions on the two endstates.
        alchemical_energy_batch_size: Only used when `use_alchemical_mlip` is True.
            Controls the maximum batch size used when computing the per-lambda
            alchemical energy of each snapshot. Required by some models to prevent
            OOM errors. If `None` (default), computes all in a single batch.
        repulsive_potential: Softcore repulsive potential applied to alchemical
            edges. Defaults to `SoftcoreLennardJonesPotential()`.
        use_replica_exchange: Whether to use Hamiltonian Replica Exchange to swap
            adjacent lambdas between episodes, according to a Metropolis Criterion.
        num_equilibration_episodes: Number of episodes before Replica Exchange begins.
            Ignored if `use_replica_exchange` is `False`.
        checkpoint_dir: Optional local directory, If set, a checkpoint is saved to this
            directory every `checkpoint_interval_episodes` episodes, enabling
            restoration. To also upload to remote storage, pass a
            `checkpoint_dir_upload_fun` to the sampler's init.
            Note that to restore from a checkpoint, `sampler.restore_checkpoint(...)`
            must be used with a local directory before `sampler.run()`. If the
            checkpoint directory is in remote storage, it must be downloaded first.
        checkpoint_interval_episodes: Number of episodes between checkpoint saves.
            Can be used to reduce upload frequency when a `checkpoint_dir_upload_fun`
            is provided (see `checkpoint_dir` above). Default is 1.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    simulation_config: JaxMDSimulationConfig
    alchemical_atom_indices: ArrayLike | list[int]
    lambda_edge_values: ArrayLike | list[float]
    lambda_repulsion_values: ArrayLike | list[float]

    use_alchemical_mlip: bool = True
    alchemical_energy_batch_size: int | None = None
    repulsive_potential: SoftcoreRepulsivePotential = pydantic.Field(
        default_factory=SoftcoreLennardJonesPotential
    )

    use_replica_exchange: bool = True
    num_equilibration_episodes: int = 0

    checkpoint_dir: Path | None = None
    checkpoint_interval_episodes: int = 1

    @pydantic.model_validator(mode="after")
    def _check_lambda_values_same_length(self) -> "FEPSimulationSamplerConfig":
        if len(self.lambda_edge_values) != len(self.lambda_repulsion_values):
            raise ValueError(
                "`lambda_edge_values` and `lambda_repulsion_values` must have the "
                "same length."
            )
        return self
