import atexit
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dropbear import PolicyStepResult
from dropbear.policy.config import ResolvedOptimizationConfig
from dropbear.policy.runtime.contract import RuntimeContract
from inspect_robots.policy import Policy, PolicyConfig
from inspect_robots.rollout import TrialRecord
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

from inspect_robots_dropbear import dropbear_policy


class FakeRemotePolicy:
    model = "dreamzero-yam"
    session_id = "session-123"
    region = "us-west-2"
    _target_key = "a3-ga"
    transport_mode = "quic"
    fallback_reason = None
    runtime_contract = RuntimeContract(
        action_hz=15.0,
        chunk_size=24,
        action_dim=14,
        action_space="raw_absolute_joint",
        hold_behavior="current_position",
    )
    resolved_optimization_config = ResolvedOptimizationConfig()

    def __init__(
        self,
        *,
        step_result: PolicyStepResult | None = None,
        begin_error: Exception | None = None,
        end_error: Exception | None = None,
    ) -> None:
        self.begin_calls: list[tuple[str, str]] = []
        self.end_calls = 0
        self.close_calls = 0
        self.step_result = step_result
        self.step_indices: list[int] = []
        self.step_timeouts: list[float] = []
        self.begin_error = begin_error
        self.end_error = end_error

    def begin_episode(self, *, instruction: str, strategy: Any) -> None:
        self.begin_calls.append((instruction, strategy.dreamzero_sampling))
        if self.begin_error is not None:
            raise self.begin_error

    def end_episode(self) -> None:
        self.end_calls += 1
        if self.end_error is not None:
            raise self.end_error

    def close(self) -> None:
        self.close_calls += 1

    def step(self, observation: object, *, action_index: int, timeout_s: float) -> PolicyStepResult:
        del observation
        self.step_indices.append(action_index)
        self.step_timeouts.append(timeout_s)
        assert self.step_result is not None
        return self.step_result


def inspect_observation(*, env_step: object = 8) -> Observation:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    return Observation(
        images={"top_cam": frame, "left_cam": frame, "right_cam": frame},
        state={"joint_pos": np.arange(14, dtype=np.float64)},
        instruction="spell NEURIPS",
        image_times={"top_cam": 10.0, "left_cam": 10.001, "right_cam": 10.002},
        extra={"env_step": env_step},
    )


def step_result(*, stalled: bool = False) -> PolicyStepResult:
    return PolicyStepResult(
        action=tuple(float(i) for i in range(14)),
        action_index=8,
        observation_id=11,
        stalled=stalled,
        actions_remaining=15,
        source_chunk_id=4,
        source_observation_id=9,
        source_control_tick=7,
        source_capture_to_execution_ms=81.25,
        replan_epoch=1,
        cache_generation=2,
    )


def test_factory_constructs_only_dreamzero_yam_without_connecting(monkeypatch) -> None:
    """Catch a discovery path that opens a Dropbear connection."""
    monkeypatch.setattr(
        "inspect_robots_dropbear.policy.dropbear.connect",
        lambda *_args, **_kwargs: pytest.fail("constructor touched the network"),
    )

    policy = dropbear_policy(model="dreamzero-yam")

    assert isinstance(policy, Policy)
    assert policy.info.name == "dropbear"
    assert policy.info.action_space.shape == (14,)
    assert policy.info.control_hz == 15.0
    assert policy.config == PolicyConfig(action_horizon=24, replan_interval=1)


def test_factory_rejects_every_other_model_offline() -> None:
    """Catch a factory that accepts a Dropbear model other than DreamZero-YAM."""
    with pytest.raises(ValueError, match="only dreamzero-yam is supported"):
        dropbear_policy(model="dreamzero-droid")


def test_factory_rejects_unknown_sampling_mode_offline() -> None:
    """Catch exposing an unsupported scheduler through the private adapter."""
    with pytest.raises(
        ValueError,
        match="sampling must be upstream_eval, async_8, or async_latest",
    ):
        dropbear_policy(model="dreamzero-yam", sampling="custom")


def test_factory_accepts_explicit_async_latest_without_connecting(monkeypatch) -> None:
    monkeypatch.setattr(
        "inspect_robots_dropbear.policy.dropbear.connect",
        lambda *_args, **_kwargs: pytest.fail("constructor touched the network"),
    )

    policy = dropbear_policy(model="dreamzero-yam", sampling="async_latest")

    assert policy.sampling == "async_latest"


def test_reset_passes_default_startup_timeout_to_dropbear_connect(monkeypatch) -> None:
    """Catch restoring the SDK's shorter default startup deadline for DreamZero-YAM."""
    remote = FakeRemotePolicy()
    startup_timeouts: list[float] = []

    def connect(
        _model: str,
        *,
        region: str,
        on_progress: object,
        startup_timeout: float,
    ) -> FakeRemotePolicy:
        del region, on_progress
        startup_timeouts.append(startup_timeout)
        return remote

    monkeypatch.setattr("inspect_robots_dropbear.policy.dropbear.connect", connect)
    policy = dropbear_policy(model="dreamzero-yam")

    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))

    assert startup_timeouts == [1800.0]


def test_custom_startup_timeout_does_not_change_per_step_timeout(monkeypatch) -> None:
    """Catch startup and action-step deadlines being conflated at the SDK boundary."""
    remote = FakeRemotePolicy(step_result=step_result())
    startup_timeouts: list[float] = []

    def connect(
        _model: str,
        *,
        region: str,
        on_progress: object,
        startup_timeout: float,
    ) -> FakeRemotePolicy:
        del region, on_progress
        startup_timeouts.append(startup_timeout)
        return remote

    monkeypatch.setattr("inspect_robots_dropbear.policy.dropbear.connect", connect)
    policy = dropbear_policy(
        model="dreamzero-yam",
        startup_timeout_s=2400.0,
        timeout_s=17.0,
    )

    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    policy.act(inspect_observation())

    assert startup_timeouts == [2400.0]
    assert remote.step_timeouts == [17.0]


@pytest.mark.parametrize(
    "startup_timeout_s",
    [True, False, None, "1800", 0, -1, float("nan"), float("inf"), float("-inf")],
)
def test_factory_rejects_invalid_startup_timeout_before_connect(
    monkeypatch, startup_timeout_s
) -> None:
    """Catch malformed startup budgets reaching config/session side effects."""
    monkeypatch.setattr(
        "inspect_robots_dropbear.policy.dropbear.connect",
        lambda *_args, **_kwargs: pytest.fail("invalid timeout opened a connection"),
    )

    with pytest.raises(ValueError, match="startup_timeout_s must be a finite positive number"):
        dropbear_policy(model="dreamzero-yam", startup_timeout_s=startup_timeout_s)


def test_reset_reuses_connection_but_isolates_same_instruction_episodes(monkeypatch) -> None:
    """Catch connection churn or reused episode state between identical trials."""
    remote = FakeRemotePolicy()
    connects: list[tuple[str, str, object]] = []

    def connect(
        model: str,
        *,
        region: str,
        on_progress: object,
        startup_timeout: float,
    ) -> FakeRemotePolicy:
        del startup_timeout
        connects.append((model, region, on_progress))
        return remote

    monkeypatch.setattr("inspect_robots_dropbear.policy.dropbear.connect", connect)
    policy = dropbear_policy(model="dreamzero-yam", sampling="upstream_eval")
    scene = Scene(id="spell", instruction="spell NEURIPS")

    policy.reset(scene)
    policy.reset(scene)
    policy.close()

    assert connects == [("dreamzero-yam", "nearest", None)]
    assert remote.begin_calls == [
        ("spell NEURIPS", "upstream_eval"),
        ("spell NEURIPS", "upstream_eval"),
    ]
    assert remote.end_calls == 2
    assert remote.close_calls == 1


def test_close_before_reset_is_offline_and_idempotent(monkeypatch) -> None:
    """Catch close accidentally connecting or releasing an owned remote twice."""
    monkeypatch.setattr(
        "inspect_robots_dropbear.policy.dropbear.connect",
        lambda *_args, **_kwargs: pytest.fail("close touched the network"),
    )
    policy = dropbear_policy(model="dreamzero-yam")

    policy.close()
    policy.close()


@pytest.mark.parametrize(("stalled", "source"), [(False, "model"), (True, "hold")])
def test_act_uses_env_step_and_returns_one_joinable_action(monkeypatch, stalled, source) -> None:
    """Catch a shifted control index, leaked chunk, or wrong action-source marker."""
    remote = FakeRemotePolicy(step_result=step_result(stalled=stalled))
    monkeypatch.setattr(
        "inspect_robots_dropbear.policy.dropbear.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dropbear_policy(model="dreamzero-yam", sampling="upstream_eval")
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))

    chunk = policy.act(inspect_observation(env_step=8))

    assert remote.step_indices == [8]
    assert len(chunk.actions) == 1
    assert chunk.actions[0].data.tolist() == [float(i) for i in range(14)]
    assert chunk.actions[0].meta == {
        "dropbear_action_source": source,
        "dropbear_cache_generation": 2,
        "dropbear_chunk_id": 4,
        "dropbear_join_key": "2:8",
        "dropbear_observation_id": 11,
        "dropbear_step": 8,
    }
    assert chunk.control_hz == 15.0
    assert chunk.inference_latency_s is not None
    assert chunk.inference_latency_s >= 0.0
    assert chunk.meta == {"dropbear_join_key": "2:8"}


@pytest.mark.parametrize("env_step", [None, True, 8.0, -1, "8"])
def test_act_rejects_malformed_env_step_before_remote_step(monkeypatch, env_step) -> None:
    """Catch an invalid Inspect clock value crossing the Dropbear boundary."""
    remote = FakeRemotePolicy(step_result=step_result())
    monkeypatch.setattr(
        "inspect_robots_dropbear.policy.dropbear.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dropbear_policy(model="dreamzero-yam")
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    observation = inspect_observation(env_step=env_step)
    if env_step is None:
        observation = Observation(
            images=observation.images,
            state=observation.state,
            image_times=observation.image_times,
            extra={},
        )

    with pytest.raises(ValueError, match=r"extra\['env_step'\] must be a nonnegative integer"):
        policy.act(observation)

    assert remote.step_indices == []


def test_trial_end_preserves_partial_row_and_pointer_before_reraising_cleanup_error(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch cleanup failure discarding the serving evidence needed to debug it."""
    remote = FakeRemotePolicy(step_result=step_result(), end_error=RuntimeError("end failed"))
    monkeypatch.setattr(
        "inspect_robots_dropbear.policy.dropbear.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dropbear_policy(model="dreamzero-yam")
    policy.on_trial_start("spell", 2, str(tmp_path), "run-123")
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    policy.act(inspect_observation())
    record = TrialRecord(scene_id="spell", epoch=2, seed=7, metadata={"existing": "kept"})

    with pytest.raises(RuntimeError, match="end failed"):
        policy.on_trial_end(record, str(tmp_path), "run-123")

    assert record.metadata == {
        "existing": "kept",
        "dropbear_telemetry": "dropbear/run-123/spell-e2.jsonl",
    }
    sidecar = tmp_path / record.metadata["dropbear_telemetry"]
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["join_key"] == "2:8"
    assert rows[0]["run_id"] == "run-123"


def test_failed_pre_episode_reset_writes_no_empty_sidecar_or_pointer(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch a failed reset being misrepresented as a delivered telemetry artifact."""
    remote = FakeRemotePolicy(begin_error=RuntimeError("begin failed"))
    monkeypatch.setattr(
        "inspect_robots_dropbear.policy.dropbear.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dropbear_policy(model="dreamzero-yam")
    policy.on_trial_start("spell", 0, str(tmp_path), "run-123")
    with pytest.raises(RuntimeError, match="begin failed"):
        policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    record = TrialRecord(scene_id="spell", epoch=0, seed=7)

    policy.on_trial_end(record, str(tmp_path), "run-123")

    assert "dropbear_telemetry" not in record.metadata
    assert not (tmp_path / "dropbear").exists()


def test_telemetry_records_runtime_transport_state_at_action_time(
    monkeypatch, tmp_path: Path
) -> None:
    """Catch a post-connect transport fallback being hidden by stale runtime identity."""
    remote = FakeRemotePolicy(step_result=step_result())
    monkeypatch.setattr(
        "inspect_robots_dropbear.policy.dropbear.connect",
        lambda *_args, **_kwargs: remote,
    )
    policy = dropbear_policy(model="dreamzero-yam")
    policy.on_trial_start("spell", 0, str(tmp_path), "run-123")
    policy.reset(Scene(id="spell", instruction="spell NEURIPS"))
    remote.transport_mode = "relay"
    remote.fallback_reason = "quic_result_timeout"
    policy.act(inspect_observation())
    record = TrialRecord(scene_id="spell", epoch=0, seed=7)

    policy.on_trial_end(record, str(tmp_path), "run-123")

    row = json.loads((tmp_path / record.metadata["dropbear_telemetry"]).read_text())
    assert row["runtime"]["transport_mode"] == "relay"
    assert row["runtime"]["fallback_reason"] == "quic_result_timeout"
    assert row["runtime"]["sampling"] == "async_8"


def test_explicit_close_unregisters_single_atexit_handler(monkeypatch) -> None:
    """Catch duplicate exit cleanup or an explicit close leaving fallback registered."""
    registered: list[object] = []
    unregistered: list[object] = []
    monkeypatch.setattr(atexit, "register", lambda handler: registered.append(handler) or handler)
    monkeypatch.setattr(atexit, "unregister", unregistered.append)

    policy = dropbear_policy(model="dreamzero-yam")
    policy.close()
    policy.close()

    assert len(registered) == 1
    assert unregistered == registered


def test_atexit_fallback_uses_bounded_daemon_thread(monkeypatch) -> None:
    """Catch process exit blocking indefinitely on remote cleanup."""
    created: list[object] = []
    joins: list[float] = []

    class FakeThread:
        def __init__(self, *, target, daemon: bool) -> None:
            self.target = target
            self.daemon = daemon
            created.append(self)

        def start(self) -> None:
            self.target()

        def join(self, *, timeout: float) -> None:
            joins.append(timeout)

    monkeypatch.setattr(threading, "Thread", FakeThread)
    policy = dropbear_policy(model="dreamzero-yam")

    policy._atexit_close()

    assert len(created) == 1
    assert created[0].daemon is True
    assert joins == [5.0]
