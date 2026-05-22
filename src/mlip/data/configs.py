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


import pydantic
from typing_extensions import Annotated

PositiveInt = Annotated[int, pydantic.Field(gt=0)]
PositiveFloat = Annotated[float, pydantic.Field(gt=0)]


class GraphDatasetBuilderConfig(pydantic.BaseModel):
    """Pydantic-based config used by both `SingleGraphDatasetBuilder` and
    `GraphDatasetBuilder` to ensure the same graph construction and batching
    parameters are applied consistently across all datasets.

    Attributes:
        graph_cutoff_angstrom: Graph cutoff distance in Angstrom to apply when
                               creating the graphs. Default is 5.0.
        long_range_cutoff_angstrom: Long range cutoff distance in Ångström used to
            build the long range neighbor lists. Defaults to `None`, meaning no long
            range graph will be built.
        max_n_node: This value will be multiplied with the batch size to determine the
                    maximum number of nodes we allow in a batch.
                    Note that a batch will always contain max_n_node * batch_size
                    nodes, as the remaining ones are filled up with dummy nodes.
                    If set to `None`, a reasonable value will be automatically
                    computed. Default is `None`.
        max_n_edge: This value will be multiplied with the batch size to determine the
                    maximum number of edges we allow in a batch.
                    Note that a batch will always contain max_n_edge * batch_size
                    edges, as the remaining ones are filled up with dummy edges.
                    If set to `None`, a reasonable value will be automatically
                    computed. Default is `None`.
        max_n_edge_long_range: This value will be multiplied with the batch size to
            determine the maximum number of long range edges we allow in a batch.
            Note that a batch will always contain
            max_n_edge_long_range * batch_size long range edges, as the
            remaining ones are filled up with dummy long range edges.
            If set to `None`, a reasonable value will be automatically computed.
            Default is `None`.
        batch_size: The number of graphs in a batch. Will be filled up with dummy graphs
                    if either the maximum number of nodes or edges are reached before
                    the number of graphs is reached. Default is 16.
        num_batch_prefetch_host: Sets the depth of the inner (host) prefetch queue.
                                 Default is 1.
        num_batch_prefetch_device: sets the depth of the outer (device) prefetch queue:
                                   how many already-sharded global batches are kept
                                   queued on devices. Default is 1.
        use_formation_energies: Whether the energies in the dataset should already be
                                transformed to subtract the average atomic energies.
                                Default is `False`.
        avg_num_neighbors: The pre-computed average number of neighbors.
        avg_r_min_angstrom: The pre-computed average minimum distance between nodes.
        remove_systems_without_partial_charges: Whether to remove systems without
                                                partial charges from the dataset.
                                                Default is `False`.
        allowed_atomic_numbers: List of allowed atomic numbers to filter the dataset by
                                during preprocessing, will remove all systems with
                                elements not in the list. Default is `None` (no filter).
        excluded_atomic_numbers: List of excluded atomic numbers to filter the dataset
                                 by during preprocessing, will remove all systems with
                                 elements in the list. Default is `None` (no filter).
        allowed_charges: List of allowed total charges to filter the dataset by during
                         preprocessing, will remove all systems with charges not in
                         the list. Default is `None` (no filter).
        excluded_charges: List of excluded total charges to filter the dataset by during
                          preprocessing, will remove all systems with charges in
                          the list. Default is `None` (no filter).
        ensure_no_unseen_total_charges: Whether to ensure that no unseen total charges
                                        are present in the dataset based on the allowed
                                        charges provided by the `dataset_info`.
                                        Default is `False`.
        set_none_charges_to_zero: Whether to set None total charges to zero during
                                  preprocessing. Default is `False`.
        homogenize: If `True`, the resulting
                    :class:`GraphDataset` will pad any missing
                    `Prediction`-targeted optional fields (e.g. `stress`,
                    `forces`) with NaN so graphs from heterogeneous datasets
                    share the same pytree structure and can be batched. If
                    `False`, the dataset will instead validate that the
                    provided graphs are already batch-compatible and raise a
                    clear error otherwise. Multi-dataset merging typically
                    requires `True` because different subsets may not share
                    the same optional-field presence. Defaults to `False`.
    """

    graph_cutoff_angstrom: PositiveFloat = 5.0
    long_range_cutoff_angstrom: PositiveFloat | None = None
    max_n_node: PositiveInt | None = None
    max_n_edge: PositiveInt | None = None
    max_n_edge_long_range: PositiveInt | None = None
    batch_size: PositiveInt = 16

    num_batch_prefetch_host: PositiveInt = 1
    num_batch_prefetch_device: PositiveInt = 1

    use_formation_energies: bool = False
    avg_num_neighbors: float | None = None
    avg_r_min_angstrom: float | None = None

    remove_systems_without_partial_charges: bool = False

    # preprocessing :
    allowed_atomic_numbers: list[int] | None = None  # no filter or [ 1, 6, 7, 8 ]
    excluded_atomic_numbers: list[int] | None = None  # no filter or [ 1, 6, 7, 8 ]
    allowed_charges: list[int] | None = None  # no filter or [-1, 0, 1]
    excluded_charges: list[int] | None = None  # no filter or [-1, 0, 1]

    # Total charge preprocessing options:
    ensure_no_unseen_total_charges: bool = False
    set_none_charges_to_zero: bool = False

    homogenize: bool = False
