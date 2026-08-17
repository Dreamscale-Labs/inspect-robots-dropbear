"""The observation and action contract DreamZero-YAM requires.

A working skeleton, taken from an embodiment that a real run drove end to end.
Replace the marked sections with your hardware; everything else is contract and
should stay as it is.

The compatibility check runs before the first reset and compares field by field
against what the adapter declares. Cameras, state, and the action space are hard
errors, so a mismatch there fails the run rather than degrading it quietly.

`control_hz` is the exception, and worth understanding before you change it.
Inspect's rollout imposes no wall-clock rate of its own, so a rate disagreement
is only a *warning* and the run proceeds. Whatever you declare here, the rate the
robot actually runs at is however fast `step()` returns. See CONTROL_HZ below.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from inspect_robots import (
    ActionSemantics,
    CameraSpec,
    Observation,
    Scene,
    StateField,
    StateSpec,
    Task,
)
from inspect_robots.embodiment import (
    Box,
    EmbodimentBase,
    EmbodimentInfo,
    ObservationSpace,
    StepResult,
)
from inspect_robots.registry import embodiment as embodiment_decorator
from inspect_robots.registry import task as task_decorator

# --- replace with your hardware driver ---------------------------------
from your_robot import YamDriver  # must expose frames() and execute()

ACTION_DIM = 14
FRAME_HEIGHT = 360
FRAME_WIDTH = 640

# The rate this embodiment actually steps at. DreamZero-YAM's native rate is
# 30 Hz; the adapter accepts any whole rate from 5 to 30 via
# `-P control_hz=...`, which stretches the 24-action chunk in time rather than
# resampling it (0.8 s at 30 Hz, 1.6 s at 15 Hz).
#
# Nothing enforces agreement between the two, so keep them equal on purpose: run
# the policy with `control_hz` set to whatever rate your loop really achieves.
# The adapter measures the gap between steps and says so if they diverge, but it
# cannot fix it -- see `_pace` below for who is responsible for the clock.
CONTROL_HZ = 30.0
EMBODIMENT_NAME = "your-yam"
TASK_NAME = "your-task"


class YamEmbodiment(EmbodimentBase):
    """Your bimanual YAM, exposed to Inspect Robots."""

    def __init__(self, seed: int = 0) -> None:
        self._robot = YamDriver()
        self._env_step = 0
        self._next_deadline: float | None = None

    @property
    def info(self) -> EmbodimentInfo:
        # Mirrors what the adapter declares. Cameras, state, and the action
        # space are compared field by field before the first reset and a
        # mismatch is a hard error, so guessing here fails the run rather than
        # degrading it. `control_hz` only warns -- see the module docstring.
        limit = np.full(ACTION_DIM, np.pi, dtype=np.float64)
        return EmbodimentInfo(
            name=EMBODIMENT_NAME,
            action_space=Box(
                shape=(ACTION_DIM,),
                low=-limit,
                high=limit,
                semantics=ActionSemantics(
                    control_mode="joint_pos",
                    rotation_repr="none",
                    gripper="continuous",
                    frame="base",
                ),
            ),
            observation_space=ObservationSpace(
                cameras=(
                    CameraSpec("top_cam", FRAME_HEIGHT, FRAME_WIDTH, 3),
                    CameraSpec("left_cam", FRAME_HEIGHT, FRAME_WIDTH, 3),
                    CameraSpec("right_cam", FRAME_HEIGHT, FRAME_WIDTH, 3),
                ),
                state=StateSpec(fields=(StateField("joint_pos", (ACTION_DIM,), unit=""),)),
            ),
            control_hz=CONTROL_HZ,
            is_simulated=True,
        )

    def _observation(self, instruction: str) -> Observation:
        top, left, right, capture_times = self._robot.frames()
        # `state` is a mapping keyed by the declared StateField, not a bare
        # array, and the times are seconds rather than nanoseconds.
        return Observation(
            images={"top_cam": top, "left_cam": left, "right_cam": right},
            state={"joint_pos": self._robot.state.astype(np.float64)},
            instruction=instruction,
            image_times={
                "top_cam": capture_times[0] / 1e9,
                "left_cam": capture_times[1] / 1e9,
                "right_cam": capture_times[2] / 1e9,
            },
            state_time=time.monotonic_ns() / 1e9,
            # The adapter joins its telemetry on this. It must advance exactly
            # once per delivered action.
            extra={"env_step": self._env_step},
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self._robot = YamDriver()
        self._env_step = 0
        self._next_deadline = None
        return self._observation(scene.instruction)

    def step(self, action: Any) -> StepResult:
        # Inspect delivers an `Action` wrapper carrying `.data` plus metadata,
        # not a bare array.
        payload = getattr(action, "data", action)
        self._robot.execute(tuple(float(value) for value in np.asarray(payload).reshape(-1)))
        self._env_step += 1
        self._pace()
        return StepResult(
            observation=self._observation(""),
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
        )

    def _pace(self) -> None:
        """Hold the control period.

        **A physical YAM does not need this.** Real actuators take real time, so
        the loop paces itself and adding a sleep only steals headroom. Delete
        this method and its call in `step()` when driving hardware.

        It matters only when the embodiment can return faster than real time --
        a simulator, a replay, or a stub. Such a loop finishes the episode
        before any action chunk can arrive, so every step holds position and the
        model contributes nothing, while Inspect still reports success.
        """

        period = 1.0 / CONTROL_HZ
        now = time.monotonic()
        if self._next_deadline is None:
            self._next_deadline = now + period
            return
        remaining = self._next_deadline - now
        if remaining > 0:
            time.sleep(remaining)
        # Never chase a missed deadline by running fast afterwards.
        self._next_deadline = max(self._next_deadline + period, time.monotonic())

    def reset_clock(self) -> None:
        self._next_deadline = None

    def close(self) -> None:
        return None


def register() -> None:
    """Register the stub task and embodiment under their registry names."""

    embodiment_decorator(EMBODIMENT_NAME)(YamEmbodiment)

    @task_decorator(TASK_NAME)
    def _task() -> Task:
        return Task(
            name=TASK_NAME,
            scenes=[
                Scene(
                    id="dummy-yam-scene",
                    instruction="put everything into the box",
                    init_seed=0,
                )
            ],
            # A dummy arm has no goal state to reach, so score the only thing
            # that is meaningful here: how far the rollout got.
            scorer="episode_length",
            max_steps=120,
            epochs=1,
        )
