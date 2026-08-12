"""Bounded, joinable Dropbear serving telemetry for Inspect trial artifacts."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, cast

from dropbear import PolicyStepResult  # type: ignore[import-untyped]

_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_OMIT = object()
_TIMING_FIELDS = (
    "observation_id",
    "chunk_id",
    "obs_to_action_ms",
    "server_queue_ms",
    "server_inference_ms",
    "server_total_ms",
    "client_overhead_ms",
    "transport_send_ms",
    "wait_for_chunk_ms",
    "quic_rtt_ms",
    "transport_mode",
    "data_plane_rtt_ms",
    "worker_queue_ms",
    "worker_preprocess_ms",
    "worker_inference_ms",
    "worker_postprocess_ms",
)
_CHUNK_FIELDS = (
    "observation_id",
    "chunk_id",
    "base_action_index",
    "accepted_action_index",
    "action_indices",
)
_DREAMZERO_MERGE_FIELDS = (
    "observation_id",
    "chunk_id",
    "base_action_index",
    "accepted_action_index",
    "requested_mode",
    "resolved_mode",
    "new_action_weight",
    "correction_steps",
    "stale_steps",
    "overlap_steps",
    "resolved_steps",
    "rebase_offset_steps",
    "first_executable_source_offset",
)


@dataclass(frozen=True)
class TrialContext:
    """Immutable identity shared by all rows from one Inspect trial."""

    run_id: str
    scene_id: str
    epoch: int


def _clean_json(value: object) -> object:
    if value is None:
        return _OMIT
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else _OMIT
    if isinstance(value, Mapping):
        cleaned: dict[str, object] = {}
        for key, item in value.items():
            projected = _clean_json(item)
            if projected is not _OMIT:
                cleaned[str(key)] = projected
        return cleaned
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        cleaned_items: list[object] = []
        for item in value:
            projected = _clean_json(item)
            if projected is not _OMIT:
                cleaned_items.append(projected)
        return cleaned_items
    return value


def _event_fields(event: Any, fields: Sequence[str]) -> dict[str, object]:
    return {field: getattr(event, field) for field in fields}


def telemetry_row(
    result: PolicyStepResult,
    context: TrialContext,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    """Project one policy step into schema-v1 diagnostics without payload vectors."""
    row: dict[str, object] = {
        "schema_version": 1,
        "run_id": context.run_id,
        "scene_id": context.scene_id,
        "epoch": context.epoch,
        "env_step": result.action_index,
        "logical_action_index": result.action_index,
        "observation_id": result.observation_id,
        "source_observation_id": result.source_observation_id,
        "source_chunk_id": result.source_chunk_id,
        "replan_epoch": result.replan_epoch,
        "cache_generation": result.cache_generation,
        "join_key": f"{result.cache_generation}:{result.action_index}",
        "action_source": "hold" if result.stalled else "model",
        "actions_remaining": result.actions_remaining,
        "timing": [_event_fields(event, _TIMING_FIELDS) for event in result.timing_events],
        "chunks": [_event_fields(event, _CHUNK_FIELDS) for event in result.chunk_events],
        "merges": [
            _event_fields(event, _DREAMZERO_MERGE_FIELDS) for event in result.merge_events
        ],
        "runtime": dict(runtime),
    }
    return cast(dict[str, object], _clean_json(row))


def _remote_target_key(remote: Any) -> object:
    """Read the SDK's focused routing identity without serializing remote state."""
    return getattr(remote, "_target_key", None)


def runtime_identity(remote: Any) -> dict[str, object]:
    """Return the explicit serving-runtime whitelist for one connected policy."""
    contract = remote.runtime_contract
    optimization = remote.resolved_optimization_config
    identity: dict[str, object] = {
        "model": remote.model,
        "session_id": remote.session_id,
        "region": remote.region,
        "target_key": _remote_target_key(remote),
        "transport_mode": remote.transport_mode,
        "fallback_reason": remote.fallback_reason,
        "runtime_contract": dataclasses.asdict(contract),
        "resolved_optimization_config": optimization.to_wire(),
        "packages": {
            "dropbear": importlib.metadata.version("dropbear"),
            "inspect_robots": importlib.metadata.version("inspect-robots"),
            "inspect_robots_dropbear": importlib.metadata.version("inspect-robots-dropbear"),
        },
    }
    return cast(dict[str, object], _clean_json(identity))


def _safe_component(value: object, *, max_length: int = 120) -> str:
    sanitized = _UNSAFE_COMPONENT.sub("-", str(value)).strip(".-")
    return (sanitized or "unknown")[:max_length]


def write_trial_sidecar(
    rows: Sequence[Mapping[str, object]],
    *,
    log_dir: str | Path,
    run_id: str,
    scene_id: str,
    epoch: int,
) -> str:
    """Atomically persist strict JSONL and return its Inspect-log-relative pointer."""
    extension = ".jsonl"
    stem = _safe_component(
        f"{scene_id}-e{epoch}",
        max_length=120 - len(extension),
    )
    relative = Path("dropbear", _safe_component(run_id), f"{stem}{extension}")
    target = Path(log_dir) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                cleaned = _clean_json(row)
                handle.write(json.dumps(cleaned, allow_nan=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return relative.as_posix()
