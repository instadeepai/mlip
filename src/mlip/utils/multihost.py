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

import contextlib
from typing import Any, Callable, List, Union
from unittest.mock import patch

import jax
import jax.numpy as jnp


def only_specific_processes(processes: Union[int, List[int]] = 0) -> Callable:
    """
    Decorator to execute a function only if the current process index
    matches the specified index/indices.

    Args:
        processes: The process index or a list of process indices for which the function
                   should be executed. Defaults to 0.

    Returns:
        The wrapped function, which executes based on the process index.

    Examples:
        @only_specific_processes()
        def function_for_zero():
            print("Function executed for process 0!")

        @only_specific_processes(1)
        def function_for_one():
            print("Function executed for process 1!")

        @only_specific_processes([2, 3])
        def function_for_two_and_three():
            print("Function executed for process 2 or 3!")

        # When executed in a multi-process JAX setup, functions will be
        # executed/skipped based on process index.
    """
    # If a single integer is passed, convert it to a list for uniformity
    if isinstance(processes, int):
        processes = [processes]

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if jax.process_index() in processes:
                return func(*args, **kwargs)
            else:
                return None

        return wrapper

    return decorator


@contextlib.contextmanager
def single_host_jax_and_orbax():
    """Context manager to mock JAX and Orbax functions.

    Makes JAX look as though it is running on a single host if in a multi-host setting.
    Additionally, skips Orbax's device sync checks (as they are not relevant in
    a "single-host" setting).

    In a true single-host setting, this context manager does nothing.

    Examples:
        print(jax.process_index())  # --> 0,...,num_hosts-1
            print(jax.process_count())  # --> num_hosts
            print(len(jax.devices()) == len(jax.local_devices())
        )  # --> False in multi-host setting

        with single_host_jax_and_orbax():
            print(jax.process_index())  # --> 0
            print(jax.process_count())  # --> 1
            print(len(jax.devices()) == len(jax.local_devices()))  # --> True
    """
    # fmt: off
    with patch("jax.process_index", return_value=0), \
            patch("jax.process_count", return_value=1), \
            patch("jax.devices", return_value=jax.local_devices()), \
            patch("orbax.checkpoint.utils.should_skip_device_sync", return_value=True):
        yield


def assert_pytrees_match_across_hosts(tree: Any) -> None:
    """
    Assert that the provided PyTree matches across multiple JAX hosts/processes.

    If there are multiple JAX processes, this function checks that the PyTree `A`
    on the current host matches the PyTree on the host with process index 0. If
    there's only one JAX process, the function returns immediately without any checks.

    Args:
        tree (Any): The PyTree to check for consistency across hosts.

    Raises:
        ValueError: If the PyTrees do not match across the hosts with a detailed path
        of differences.
    """

    if jax.process_count() == 1:
        return

    tree_src = jax.experimental.multihost_utils.broadcast_one_to_all(
        tree, is_source=jax.process_index() == 0
    )
    assert_pytrees_match(tree, tree_src)


def assert_pytrees_match(a: Any, b: Any) -> None:
    """
    Assert that the two provided PyTrees have matching structures and values.

    PyTrees are a way to flexibly handle nested data structures in the JAX library.

    Args:
        a (Any): The first PyTree.
        b (Any): The second PyTree.

    Raises:
        ValueError: If the PyTrees do not match with a detailed path of differences.
    """
    pytree_eq = jax.tree.map(lambda x, y: jnp.all(x == y), a, b)
    pytrees_match = all(jax.tree_util.tree_leaves(pytree_eq))
    if not pytrees_match:
        exp_str = (
            "Provided PyTree's do not match.  Difference(s) found at following paths:"
        )
        for path, do_match in jax.tree_util.tree_leaves_with_path(pytree_eq):
            if not do_match:
                exp_str += f" '{'/'.join([p.key for p in path])}'"
        raise ValueError(exp_str)
