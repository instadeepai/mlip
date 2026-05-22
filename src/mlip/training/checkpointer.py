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

"""Unified Orbax-based checkpointer for post-epoch and intra-epoch saves."""

import contextlib
import logging
import os
from typing import Any, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pydantic
from jax.sharding import Mesh, PartitionSpec
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from typing_extensions import Annotated

from mlip.data.graph_dataset import GraphDatasetState
from mlip.training.ema import get_debiased_params
from mlip.training.training_state import TrainingState
from mlip.utils.multihost import single_host_jax_and_orbax

PathLike: TypeAlias = str | os.PathLike
MODEL_SUBDIR_NAME = "model"
INTRA_EPOCH_SUBDIR_NAME = "intra_epoch"
PositiveInt = Annotated[int, pydantic.Field(gt=0)]
EMADecay = Annotated[float, pydantic.Field(gt=0.0, le=1.0)]

logger = logging.getLogger("mlip")


def _serialize_empty(ckpt: Any) -> Any:
    """Patch: serialize empty params as `jnp.array(False)`.

    Needed for skipped/padded zero blocks of `e3j.linen.LinearIndexwise`.
    We place `jnp.array(False)` onto the original leaf's sharding.
    """

    def _replace(t):
        if not hasattr(t, "size") or t.size:
            return t
        sentinel = jnp.array(False)
        sharding = getattr(t, "sharding", None)
        if sharding is not None:
            sentinel = jax.device_put(sentinel, sharding)
        return sentinel

    return jax.tree.map(_replace, ckpt)


def _deserialize_empty(ckpt: Any, reference: Any) -> Any:
    """Patch: deserialize empty params, restoring shape/dtype from reference."""

    def _is_empty(t):
        return hasattr(t, "size") and t.size == 1 and t.dtype == jnp.bool_ and not t

    return jax.tree.map(
        lambda t, ref: jnp.empty(ref.shape, dtype=ref.dtype) if _is_empty(t) else t,
        ckpt,
        reference,
    )


class CheckpointerConfig(pydantic.BaseModel):
    """Configuration for :class:`OrbaxCheckpointer`.

    Attributes:
        checkpoint_dir: Root directory for checkpoint storage.
        max_to_keep: Maximum number of checkpoints to retain. `None`
                     disables pruning entirely, which is the right setting for
                     read-only managers pointed at a foreign directory.
        save_interval_steps: Only save when the step key is a multiple of this
                             value. Use `1` for post-epoch checkpointers
                             (save every epoch).
        save_debiased_ema: Whether to persist debiased EMA parameters alongside
                           the training state.
        ema_decay: EMA decay rate used for debiasing.
        save_metrics: Whether to persist accumulated per-step metrics inside
                      the checkpoint. Typically `True` for intra-epoch and
                      `False` for post-epoch checkpointers.
        use_single_host_patch: Whether to wrap Orbax calls in
                               :func:`single_host_jax_and_orbax`. Defaults to
                               `True` so that Orbax writes without multi-host
                               coordination. Set to `False` to let Orbax
                               handle multi-host coordination natively.
        enable_async_checkpointing: Whether Orbax should write checkpoints
                                    asynchronously. Defaults to `False`.
                                    Incompatible with `use_single_host_patch`:
                                    if both are set to `True`, a warning is
                                    emitted and async checkpointing is disabled.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    checkpoint_dir: PathLike
    max_to_keep: PositiveInt | None = 5
    save_interval_steps: PositiveInt = 1
    save_debiased_ema: bool = True
    ema_decay: EMADecay = 0.99
    save_metrics: bool = False
    use_single_host_patch: bool = True
    enable_async_checkpointing: bool = False

    @pydantic.model_validator(mode="after")
    def _disable_async_with_single_host_patch(self) -> "CheckpointerConfig":
        if self.enable_async_checkpointing and self.use_single_host_patch:
            logger.warning(
                "enable_async_checkpointing=True is incompatible with "
                "use_single_host_patch=True; falling back to synchronous "
                "checkpointing."
            )
            self.enable_async_checkpointing = False
        return self


class CheckpointMetadata(NamedTuple):
    """Lightweight metadata stored alongside each checkpoint."""

    num_steps: int
    epoch_number: int


class RestoreResult(NamedTuple):
    """Value returned by :meth:`OrbaxCheckpointer.restore`."""

    training_state: TrainingState
    metadata: CheckpointMetadata
    accumulated_metrics: list
    dataset_state: GraphDatasetState | None = None


class OrbaxCheckpointer:
    """Unified Orbax :class:`CheckpointManager` wrapper.

    Handles both post-epoch and intra-epoch checkpoint use-cases through
    configuration (`save_interval_steps`, `save_metrics`).
    """

    def __init__(
        self,
        config: CheckpointerConfig,
    ) -> None:
        self._config = config
        self._checkpoint_dir = str(config.checkpoint_dir)

        item_handlers: dict[str, ocp.CheckpointHandler] = {
            "training_state": ocp.PyTreeCheckpointHandler(),
            "dataset_state": ocp.PyTreeCheckpointHandler(),
        }
        if config.save_debiased_ema:
            item_handlers["params_ema"] = ocp.PyTreeCheckpointHandler()
        if config.save_metrics:
            item_handlers["accumulated_metrics"] = ocp.JsonCheckpointHandler()

        options = CheckpointManagerOptions(
            save_interval_steps=config.save_interval_steps,
            max_to_keep=config.max_to_keep,
            create=True,
            cleanup_tmp_directories=True,
            enable_async_checkpointing=config.enable_async_checkpointing,
            best_fn=lambda m: m["eval_loss"],
            best_mode="min",
        )
        with self._context():
            self._ckpt_manager = CheckpointManager(
                self._checkpoint_dir,
                options=options,
                item_handlers=item_handlers,
            )

    def _context(self) -> contextlib.AbstractContextManager:
        """Return the appropriate context for Orbax calls."""
        if self._config.use_single_host_patch:
            return single_host_jax_and_orbax()
        return contextlib.nullcontext()

    def should_save(self, key: int) -> bool:
        """Return `True` if a checkpoint should be written for *key*."""
        with self._context():
            return self._ckpt_manager.should_save(key)

    def save(
        self,
        training_state: TrainingState,
        key: int,
        epoch_number: int,
        dataset_state: GraphDatasetState | None = None,
        accumulated_metrics: list | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Persist *training_state* under *key*.

        Args:
            training_state: Current training state (may be on-device or host).
            key: Step or epoch number used as the checkpoint key.
            epoch_number: Stored in custom metadata for later retrieval.
            dataset_state: Iterator state of the training dataset, persisted
                           as a sibling item so it can be restored separately.
                           In multi-host mode the caller must hand in a
                           globally-addressable pytree here; host-local arrays
                           cannot be serialised by Orbax.
            accumulated_metrics: Per-step metrics. Only serialised when
                                 `save_metrics` is enabled in the config.
            metrics: Scalar metrics (must contain `"eval_loss"`) used
                     for best-checkpoint ranking.
        """
        if self._config.use_single_host_patch and jax.process_index() != 0:
            return

        logger.debug("OrbaxCheckpointer: saving key=%s (epoch %s)", key, epoch_number)

        if self._config.use_single_host_patch:
            training_state = jax.device_get(training_state)
            if dataset_state is not None:
                dataset_state = jax.device_get(dataset_state)

        # Serialize empty arrays (e3j LinearIndexwise padded blocks) before
        # handing the pytree to Orbax.
        training_state_for_save = _serialize_empty(training_state)
        ckpt: dict[str, Any] = {
            "training_state": ocp.args.PyTreeSave(training_state_for_save),
        }
        if dataset_state is not None:
            ckpt["dataset_state"] = ocp.args.PyTreeSave(dataset_state)
        if self._config.save_debiased_ema:
            ckpt["params_ema"] = ocp.args.PyTreeSave(
                _serialize_empty(
                    get_debiased_params(
                        training_state.ema_state, self._config.ema_decay
                    )
                )
            )

        if self._config.save_metrics:
            # Convert numpy scalars to Python floats for JSON serialization.
            steps = [
                {k: float(v) for k, v in m.items()} for m in (accumulated_metrics or [])
            ]
            ckpt["accumulated_metrics"] = ocp.args.JsonSave({"steps": steps})

        custom_metadata = {
            "num_steps": int(training_state.num_steps),
            "epoch_number": epoch_number,
        }

        save_kwargs: dict[str, Any] = dict(
            args=ocp.args.Composite(**ckpt),
            custom_metadata=custom_metadata,
        )
        if metrics is not None:
            save_kwargs["metrics"] = metrics

        with self._context():
            self._ckpt_manager.save(key, **save_kwargs)

    @staticmethod
    def _restore_args_from_target(target: Any, mesh: Mesh) -> Any:
        """Build per-leaf `ArrayRestoreArgs` from a pytree template."""
        return jax.tree.map(
            lambda _: ocp.type_handlers.ArrayRestoreArgs(
                mesh=mesh, mesh_axes=PartitionSpec()
            ),
            target,
        )

    def restore(
        self,
        training_state: TrainingState,
        key: int | None = None,
        mesh: Mesh | None = None,
        dataset_state: GraphDatasetState | None = None,
    ) -> RestoreResult | None:
        """Restore a checkpoint.

        When `use_single_host_patch` is enabled, all hosts call this method
        and the :func:`single_host_jax_and_orbax` context lets each host
        restore independently from the shared filesystem.

        Args:
            training_state: Template state used to infer pytree structure.
            key: Checkpoint key to restore. When `None` (the default),
                 the most recent checkpoint is restored.
            mesh: Optional JAX :class:`Mesh` used for hardware-agnostic
                  restoration. When provided, each leaf is restored with
                  :class:`ArrayRestoreArgs` so that Orbax can re-shard
                  arrays onto the current device topology.
            dataset_state: Optional dataset state template. When provided,
                           the dataset state is restored alongside the
                           training state and returned in the result.

        Returns:
            A :class:`RestoreResult` or `None` when no checkpoint exists.
        """
        training_state_template = _serialize_empty(training_state)

        restore_kwargs: dict[str, Any] = {}
        if mesh is not None:
            restore_kwargs["restore_args"] = self._restore_args_from_target(
                training_state_template, mesh
            )

        with self._context():
            if key is None:
                key = self._ckpt_manager.latest_step()
                if key is None:
                    return None

            meta = self._ckpt_manager.metadata(key)
            num_steps = meta.custom_metadata.get("num_steps", 0)
            epoch_number = meta.custom_metadata.get("epoch_number", 1)

            restore_args: dict[str, Any] = {
                "training_state": ocp.args.PyTreeRestore(
                    training_state_template, **restore_kwargs
                ),
            }
            if dataset_state is not None:
                ds_restore_kwargs: dict[str, Any] = {}
                if mesh is not None:
                    ds_restore_kwargs["restore_args"] = self._restore_args_from_target(
                        dataset_state, mesh
                    )
                restore_args["dataset_state"] = ocp.args.PyTreeRestore(
                    dataset_state, **ds_restore_kwargs
                )
            if self._config.save_metrics:
                restore_args["accumulated_metrics"] = ocp.args.JsonRestore()

            ckpt = self._ckpt_manager.restore(
                key,
                args=ocp.args.Composite(**restore_args),
            )

        # Reconstruct empty e3j LinearIndexwise blocks using the original
        # training_state as the shape/dtype reference.
        restored_state = _deserialize_empty(ckpt["training_state"], training_state)

        if num_steps > 0 and int(restored_state.num_steps) != num_steps:
            raise ValueError(
                f"Checkpoint restore failed: metadata has num_steps={num_steps} "
                f"but restored state has num_steps={int(restored_state.num_steps)}. "
                f"This usually means the pytree structure of the template does not "
                f"match the checkpoint."
            )

        accumulated_metrics: list = []
        if self._config.save_metrics:
            accumulated_metrics = ckpt["accumulated_metrics"].get("steps", [])

        restored_dataset_state = ckpt.get("dataset_state") if dataset_state else None

        return RestoreResult(
            training_state=restored_state,
            metadata=CheckpointMetadata(num_steps=num_steps, epoch_number=epoch_number),
            accumulated_metrics=accumulated_metrics,
            dataset_state=restored_dataset_state,
        )

    def latest_key(self) -> int | None:
        """Return the key of the most recent checkpoint, or `None`."""
        with self._context():
            return self._ckpt_manager.latest_step()

    def best_key(self) -> int | None:
        """Return the key of the best checkpoint, or `None`.

        Returns `None` when no best checkpoint has been tracked yet.
        """
        with self._context():
            return self._ckpt_manager.best_step()

    def metadata(self, key: int) -> CheckpointMetadata | None:
        """Read metadata for checkpoint *key* without loading the pytree."""
        with self._context():
            meta = self._ckpt_manager.metadata(key)
        if meta is None or meta.custom_metadata is None:
            return None
        return CheckpointMetadata(
            num_steps=meta.custom_metadata.get("num_steps", 0),
            epoch_number=meta.custom_metadata.get("epoch_number", 0),
        )

    def latest_metadata(self) -> CheckpointMetadata | None:
        """Read metadata from the latest checkpoint without loading the pytree.

        Returns:
            A :class:`CheckpointMetadata` or `None` if no checkpoint exists.
        """
        with self._context():
            latest = self._ckpt_manager.latest_step()
        if latest is None:
            return None
        return self.metadata(latest)

    def wait_until_finished(self) -> None:
        """Block until any pending asynchronous saves complete."""
        with self._context():
            self._ckpt_manager.wait_until_finished()
