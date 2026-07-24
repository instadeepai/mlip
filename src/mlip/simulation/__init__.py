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

import sys
import types


def _defer_jax_md_rigid_body_import() -> None:
    """Stub `jax_md.rigid_body` so importing it doesn't touch the XLA backend.

    Importing from `jax_md` automatically imports the `rigid_body` module which runs
    JAX code, initializing the XLA backend. This can cause problems for initializing
    JAX with requires settings, e.g. `jax.distributed.initialize()`. Overriding this
    module prevents XLA backend initialization when importing from `mlip.simulation`.
    """
    if "jax_md.rigid_body" in sys.modules:
        return

    stub = types.ModuleType("jax_md.rigid_body")

    def __getattr__(name: str):
        del sys.modules["jax_md.rigid_body"]
        import jax_md.rigid_body as real_module  # noqa: PLC0415

        return getattr(real_module, name)

    stub.__getattr__ = __getattr__
    sys.modules["jax_md.rigid_body"] = stub


_defer_jax_md_rigid_body_import()

from .enums import SimulationType  # noqa: E402
from .state import SimulationState  # noqa: E402
