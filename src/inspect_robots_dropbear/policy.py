"""Offline Inspect policy descriptor for Dropbear-hosted DreamZero-YAM."""

from __future__ import annotations

import atexit
import threading
import time
from numbers import Integral
from typing import Any, Literal

import dropbear as _dropbear  # type: ignore[import-untyped]
import numpy as np
from dropbear import RegionPreference, RunStrategy  # type: ignore[import-untyped]
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_dropbear.dreamzero_yam import to_dreamzero_yam
from inspect_robots_dropbear.telemetry import (
    TrialContext,
    runtime_identity,
    telemetry_row,
    write_trial_sidecar,
)

# Kept module-visible so discovery tests can prove construction does not call connect().
dropbear: Any = _dropbear

YAM_DIM_LABELS = (
    "left_j0",
    "left_j1",
    "left_j2",
    "left_j3",
    "left_j4",
    "left_j5",
    "left_gripper",
    "right_j0",
    "right_j1",
    "right_j2",
    "right_j3",
    "right_j4",
    "right_j5",
    "right_gripper",
)


class DropbearPolicy(PolicyBase):
    """Describe DreamZero-YAM without reading config or opening a session."""

    def __init__(
        self,
        *,
        model: str = "dreamzero-yam",
        region: RegionPreference = "nearest",
        sampling: Literal["upstream_eval", "async_8"] = "async_8",
        timeout_s: float = 60.0,
    ) -> None:
        if model != "dreamzero-yam":
            raise ValueError("only dreamzero-yam is supported")
        if sampling not in {"upstream_eval", "async_8"}:
            raise ValueError("sampling must be upstream_eval or async_8")
        self.model = model
        self.region = region
        self.sampling = sampling
        self.timeout_s = timeout_s
        self._remote: Any | None = None
        self._episode_active = False
        self._closed = False
        self._trial_context: TrialContext | None = None
        self._trial_log_dir: str | None = None
        self._telemetry_rows: list[dict[str, object]] = []
        self.info = PolicyInfo(
            name="dropbear",
            action_space=Box(
                shape=(14,),
                semantics=ActionSemantics(
                    control_mode="joint_pos",
                    rotation_repr="none",
                    gripper="continuous",
                    frame="base",
                    dim_labels=YAM_DIM_LABELS,
                ),
            ),
            observation_space=ObservationSpace(
                cameras=(
                    CameraSpec("top_cam", 360, 640, 3),
                    CameraSpec("left_cam", 360, 640, 3),
                    CameraSpec("right_cam", 360, 640, 3),
                ),
                state=StateSpec(fields=(StateField("joint_pos", (14,), unit=""),)),
            ),
            control_hz=15.0,
        )
        self.config = PolicyConfig(action_horizon=24, replan_interval=1)
        self._atexit_handler = self._atexit_close
        atexit.register(self._atexit_handler)

    def _ensure_connected(self) -> Any:
        if self._closed:
            raise RuntimeError("DropbearPolicy is closed")
        if self._remote is None:
            self._remote = dropbear.connect(
                "dreamzero-yam",
                region=self.region,
                on_progress=None,
            )
        return self._remote

    def _end_episode(self) -> None:
        if not self._episode_active:
            return
        assert self._remote is not None
        try:
            self._remote.end_episode()
        finally:
            self._episode_active = False

    def reset(self, scene: Scene) -> None:
        """Start an isolated logical episode on the reusable connection."""
        remote = self._ensure_connected()
        self._end_episode()
        self._telemetry_rows.clear()
        remote.begin_episode(
            instruction=scene.instruction,
            strategy=RunStrategy(dreamzero_sampling=self.sampling),
        )
        self._episode_active = True

    def act(self, observation: Observation) -> ActionChunk:
        """Advance the externally clocked Dropbear episode by one Inspect step."""
        env_step = observation.extra.get("env_step")
        if isinstance(env_step, bool) or not isinstance(env_step, Integral) or env_step < 0:
            raise ValueError("extra['env_step'] must be a nonnegative integer")
        remote = self._remote
        if remote is None or not self._episode_active:
            raise RuntimeError("reset() must start an episode before act()")
        started = time.perf_counter()
        result = remote.step(
            to_dreamzero_yam(observation),
            action_index=int(env_step),
            timeout_s=self.timeout_s,
        )
        wall_s = time.perf_counter() - started
        if self._trial_context is not None:
            self._telemetry_rows.append(
                telemetry_row(result, self._trial_context, runtime_identity(remote))
            )
        join_key = f"{result.cache_generation}:{result.action_index}"
        action_meta = {
            "dropbear_action_source": "hold" if result.stalled else "model",
            "dropbear_cache_generation": result.cache_generation,
            "dropbear_chunk_id": result.source_chunk_id,
            "dropbear_join_key": join_key,
            "dropbear_observation_id": result.observation_id,
            "dropbear_step": result.action_index,
        }
        return ActionChunk(
            actions=[Action(data=np.asarray(result.action, dtype=np.float64), meta=action_meta)],
            control_hz=15.0,
            inference_latency_s=wall_s,
            meta={"dropbear_join_key": join_key},
        )

    def on_trial_start(self, scene_id: str, epoch: int, log_dir: str, run_id: str) -> None:
        """Capture immutable artifact identity before Inspect resets the policy."""
        self._trial_context = TrialContext(run_id=run_id, scene_id=scene_id, epoch=epoch)
        self._trial_log_dir = log_dir
        self._telemetry_rows.clear()

    def on_trial_end(self, record: TrialRecord, log_dir: str, run_id: str) -> None:
        """Persist partial diagnostics even when logical episode cleanup fails."""
        del log_dir, run_id
        cleanup_error: BaseException | None = None
        try:
            self._end_episode()
        except BaseException as error:
            cleanup_error = error
        try:
            if (
                self._telemetry_rows
                and self._trial_context is not None
                and self._trial_log_dir is not None
            ):
                pointer = write_trial_sidecar(
                    self._telemetry_rows,
                    log_dir=self._trial_log_dir,
                    run_id=self._trial_context.run_id,
                    scene_id=self._trial_context.scene_id,
                    epoch=self._trial_context.epoch,
                )
                record.metadata["dropbear_telemetry"] = pointer
        finally:
            if cleanup_error is not None:
                raise cleanup_error

    def _close_suppressing_errors(self) -> None:
        try:
            self.close()
        except BaseException:
            pass

    def _atexit_close(self) -> None:
        thread = threading.Thread(target=self._close_suppressing_errors, daemon=True)
        thread.start()
        thread.join(timeout=5.0)

    def close(self) -> None:
        """End the current episode and release the owned Dropbear session once."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self._atexit_handler)
        remote = self._remote
        if remote is None:
            return
        try:
            self._end_episode()
        finally:
            try:
                remote.close()
            finally:
                self._remote = None


def dropbear_policy(**kwargs: Any) -> DropbearPolicy:
    """Create the registry-discoverable Dropbear Inspect policy."""
    return DropbearPolicy(**kwargs)
