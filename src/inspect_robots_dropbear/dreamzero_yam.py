"""Strict conversion from Inspect observations to Dropbear DreamZero-YAM input."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import dropbear as _dropbear  # type: ignore[import-untyped]
import numpy as np
from dropbear.dreamzero_yam import DreamZeroYamObservation  # type: ignore[import-untyped]
from inspect_robots.types import Observation

_CAMERA_NAMES = ("top_cam", "left_cam", "right_cam")
dropbear: Any = _dropbear


def _seconds_to_ns(value: object, *, camera: str) -> int:
    """Convert a finite, nonnegative capture time to integer nanoseconds."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"image_times[{camera!r}] must be finite monotonic seconds")
    try:
        seconds = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"image_times[{camera!r}] must be finite monotonic seconds") from error
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"image_times[{camera!r}] must be finite monotonic seconds")
    return round(seconds * 1_000_000_000)


def _mapping_value(mapping: Mapping[str, object], key: str, *, field: str) -> object:
    try:
        return mapping[key]
    except KeyError as error:
        raise ValueError(f"missing {field}[{key!r}]") from error


def _joint_positions(observation: Observation) -> np.ndarray:
    raw = _mapping_value(observation.state, "joint_pos", field="state")
    raw_values = np.asarray(raw, dtype=object).reshape(-1)
    if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise ValueError("joint_pos must contain exactly 14 finite values")
    try:
        joints = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("joint_pos must contain exactly 14 finite values") from error
    if joints.shape != (14,) or not np.isfinite(joints).all():
        raise ValueError("joint_pos must contain exactly 14 finite values")
    return joints


def to_dreamzero_yam(observation: Observation) -> DreamZeroYamObservation:
    """Map named Inspect cameras, times, and packed YAM state without synthesis."""
    joints = _joint_positions(observation)
    frames = tuple(
        _mapping_value(observation.images, name, field="images") for name in _CAMERA_NAMES
    )
    times = tuple(
        _seconds_to_ns(
            _mapping_value(observation.image_times, name, field="image_times"), camera=name
        )
        for name in _CAMERA_NAMES
    )
    return dropbear.dreamzero_yam.observe(
        top_frame=frames[0],
        left_frame=frames[1],
        right_frame=frames[2],
        camera_capture_times_ns=times,
        left_joint_positions=joints[0:6],
        left_gripper=float(joints[6]),
        right_joint_positions=joints[7:13],
        right_gripper=float(joints[13]),
    )
