"""Offline Inspect policy descriptor for Dropbear-hosted DreamZero-YAM."""

from __future__ import annotations

from typing import Any

import dropbear as _dropbear  # type: ignore[import-untyped]
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import ActionChunk, Observation

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

    def __init__(self, *, model: str) -> None:
        if model != "dreamzero-yam":
            raise ValueError("only dreamzero-yam is supported")
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

    def act(self, observation: Observation) -> ActionChunk:
        """Reserve remote inference for the connection implementation in Task 6."""
        del observation
        raise RuntimeError("DropbearPolicy connection behavior is not implemented")


def dropbear_policy(**kwargs: Any) -> DropbearPolicy:
    """Create the registry-discoverable Dropbear Inspect policy."""
    return DropbearPolicy(**kwargs)
