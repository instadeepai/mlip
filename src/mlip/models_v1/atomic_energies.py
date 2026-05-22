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

from typing import Union

import jax.numpy as jnp

from mlip.data.dataset_info import DatasetInfo


def get_atomic_energies(
    dataset_info: DatasetInfo,
    atomic_energies_input: Union[str, dict[int, float]] | None = None,
    num_species: int | None = None,
) -> jnp.ndarray:
    """Converts an input description of atomic energies into the atomic energies array.

    The input description can be many different things. Some of them use the given
    dataset info object to extract information, other do not. See the description of the
    argument below for details.

    If only a dataset info is given, the other defaults make sure that the atomic
    energies are simply taken from the dataset info.

    Args:
        dataset_info: The dataset information object that holds the atomic energies
                      dictionary that is used if `atomic_energies_input='average'` or
                      `atomic_energies_input=None`, which is the default behavior
                      if only the dataset info is given to the function.
        atomic_energies_input: A description of the atomic energies strategy. This can
                               be a string `"average"` or `"zero"`, or it can be the
                               atomic energies dictionary. It can also be `None`,
                               which is the default.
                               If it is `"average"` or `None`, then the atomic energies
                               for each node are extracted from the dataset info.
                               If it is `"zero"`, all atomic energies will be just zero.
                               One can also provide a dictionary as input, in that case
                               that dictionary is used instead of the one in the dataset
                               info to extract the atomic energies for each node.
        num_species: Number of species for the model. It can be `None` (default), which
                     means that the number of species is inferred from the given
                     dataset info.

    Returns:
        The atomic energies as an array of size number of species.
    """
    if num_species is None:
        num_species = len(dataset_info.atomic_energies_map)

    # The two lines below replace the former AtomicNumberTable for this only-v1 function
    zs = sorted(dataset_info.atomic_energies_map.keys())
    z_map = {z: i for i, z in enumerate(zs)}

    if atomic_energies_input == "average" or atomic_energies_input is None:
        atomic_energies_dict = {
            z_map[z]: energy for z, energy in dataset_info.atomic_energies_map.items()
        }
        atomic_energies = jnp.array([atomic_energies_dict[i] for i in range(len(zs))])
    elif atomic_energies_input == "zero":
        atomic_energies = jnp.zeros(num_species)
    elif isinstance(atomic_energies_input, dict):
        atomic_energies_dict = atomic_energies_input
        atomic_energies = jnp.array([
            atomic_energies_dict.get(z, 0.0) for z in range(num_species)
        ])
    else:
        raise ValueError(
            f"The requested strategy for atomic energies "
            f"handling '{atomic_energies_input}' is not supported."
        )

    if len(zs) > num_species:
        raise ValueError(f"len(z_table.zs)={len(zs)} > num_species={num_species}")

    return atomic_energies
