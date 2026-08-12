import re

import numpy as np
import pytest
from inspect_robots.types import Observation

from inspect_robots_dropbear.dreamzero_yam import to_dreamzero_yam


def frame(value: int) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.uint8)


def valid_observation() -> Observation:
    return Observation(
        images={
            "top_cam": frame(1),
            "left_cam": frame(2),
            "right_cam": frame(3),
        },
        state={"joint_pos": np.arange(14, dtype=np.float64)},
        instruction="spell NEURIPS",
        image_times={"top_cam": 10.0, "left_cam": 10.001, "right_cam": 10.002},
        extra={"env_step": 8},
    )


def test_to_dreamzero_yam_maps_cameras_times_and_packed_state() -> None:
    """Catch a camera, timestamp, or joint-order mapping mismatch at the SDK boundary."""
    mapped = to_dreamzero_yam(valid_observation())

    assert mapped.camera_capture_times_ns == (10_000_000_000, 10_001_000_000, 10_002_000_000)
    assert mapped.left_joint_positions == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert mapped.left_gripper == 6.0
    assert mapped.right_joint_positions == (7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
    assert mapped.right_gripper == 13.0


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (
            Observation(
                images={"left_cam": frame(2), "right_cam": frame(3)},
                state={"joint_pos": np.arange(14, dtype=np.float64)},
                image_times={"top_cam": 10.0, "left_cam": 10.001, "right_cam": 10.002},
            ),
            "missing images['top_cam']",
        ),
        (
            Observation(
                images={"top_cam": frame(1), "left_cam": frame(2), "right_cam": frame(3)},
                state={"joint_pos": np.arange(14, dtype=np.float64)},
                image_times={"left_cam": 10.001, "right_cam": 10.002},
            ),
            "missing image_times['top_cam']",
        ),
        (
            Observation(
                images={"top_cam": frame(1), "left_cam": frame(2), "right_cam": frame(3)},
                state={},
                image_times={"top_cam": 10.0, "left_cam": 10.001, "right_cam": 10.002},
            ),
            "missing state['joint_pos']",
        ),
    ],
)
def test_to_dreamzero_yam_names_each_missing_required_observation_key(
    observation: Observation, message: str
) -> None:
    """Catch a KeyError or an ambiguous error when an Inspect mapping is incomplete."""
    with pytest.raises(ValueError, match=re.escape(message)):
        to_dreamzero_yam(observation)


@pytest.mark.parametrize("value", [True, np.bool_(True), -0.1, float("nan"), float("inf")])
def test_to_dreamzero_yam_rejects_invalid_camera_time(value: object) -> None:
    """Catch malformed wall-clock seconds before they are converted to nanoseconds."""
    observation = valid_observation()
    observation = Observation(
        images=observation.images,
        state=observation.state,
        image_times={**observation.image_times, "top_cam": value},
    )

    with pytest.raises(
        ValueError, match=r"image_times\['top_cam'\] must be finite monotonic seconds"
    ):
        to_dreamzero_yam(observation)


def test_to_dreamzero_yam_rejects_camera_skew_over_fifty_ms() -> None:
    """Catch a mapping that sends an incoherent camera triplet to DreamZero-YAM."""
    observation = valid_observation()
    observation = Observation(
        images=observation.images,
        state=observation.state,
        image_times={"top_cam": 10.0, "left_cam": 10.001, "right_cam": 10.050_000_001},
    )

    with pytest.raises(ValueError, match="camera skew exceeds the 50 ms DreamZero-YAM limit"):
        to_dreamzero_yam(observation)


@pytest.mark.parametrize(
    "joint_pos",
    [
        np.arange(13, dtype=np.float64),
        np.array([True] * 14),
        np.array([0.0] * 13 + [float("nan")]),
    ],
)
def test_to_dreamzero_yam_rejects_invalid_joint_positions(joint_pos: np.ndarray) -> None:
    """Catch malformed YAM state before joint values are packed by arm and gripper."""
    observation = valid_observation()
    observation = Observation(
        images=observation.images,
        state={"joint_pos": joint_pos},
        image_times=observation.image_times,
    )

    with pytest.raises(ValueError, match="joint_pos must contain exactly 14 finite values"):
        to_dreamzero_yam(observation)
