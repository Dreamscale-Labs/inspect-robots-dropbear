# inspect-robots-dropbear

Private [Inspect Robots](https://github.com/robocurve/inspect-robots) policy adapter for
Dropbear-hosted DreamZero-YAM. Discovery and construction are offline; the first trial reset
opens one lazy Dropbear connection, and later trials reuse that connection while starting fresh
logical episodes.

This package is intentionally private and has no public license declaration yet.

## Install and discover

Install the immutable private Git release after DreamScale grants repository read access:

```bash
uv add "inspect-robots-dropbear @ git+ssh://git@github.com/Dreamscale-Labs/inspect-robots-dropbear.git@v0.1.0"
```

Verify that the entry point is available without opening a cloud session:

```bash
inspect-robots list policies
```

The output must contain `dropbear`. Robocurve keeps its existing registered task and embodiment;
do not replace or rename either. Change only the policy selection in the existing evaluation
command:

```bash
--policy dropbear -P model=dreamzero-yam -P sampling=async_8
```

Use `sampling=upstream_eval` only for the agreed open-loop dataset evaluation. The adapter exposes
no custom scheduling, buffering, horizon, or smoothing knobs.

## Observation and simulator contract

The existing task and embodiment must provide all of the following on every policy step:

- `top_cam`, `left_cam`, and `right_cam` uint8 images, each with a real monotonic capture time;
- finite packed `joint_pos` state with shape `(14,)` in YAM left-arm, left-gripper, right-arm,
  right-gripper order; and
- Inspect's integer `extra["env_step"]`, starting at zero and advancing once per delivered action.

The adapter declares a 14-dimensional raw absolute-joint action at 15 Hz. It returns exactly one
action per Inspect `act()` call while Dropbear owns DreamZero's managed action buffering. Simulator
compatibility means matching those camera, state, action, clock, and rate contracts; it does not by
itself establish physics parity, task success, or physical-robot safety.

## Artifact ownership and joining

Inspect remains canonical for the EvalLog, aggregate scores, post-approval commanded-action JSONL,
stored frames, Rerun recording, operator judgement, and trial termination/error state. The adapter
adds one atomic diagnostics sidecar and records its relative path at
`TrialRecord.metadata["dropbear_telemetry"]`:

```text
dropbear/<run_id>/<sanitized-scene-id>-e<epoch>.jsonl
```

Each sidecar row contains serving identity, timing, chunk/merge disposition, and the same Inspect
environment step. Join it to the EvalLog, action JSONL, or Rerun timeline using `env_step`; use
`join_key` (`<cache_generation>:<logical_action_index>`) for Dropbear chunk diagnostics. Sidecars do
not duplicate action vectors, images, credentials, authorization material, certificates, or
endpoints.

## Deterministic cleanup

Evaluation owners must call `policy.close()` in `finally` after the run, even when Inspect reports an
error or cancellation. Then verify that the just-used session is gone:

```bash
dropbear sessions list
```

Do not stop unrelated sessions. The adapter also registers a bounded process-exit fallback, but it
is not a substitute for explicit close.

## Physical YAM boundary

This integration does not authorize an unattended physical run. For any physical YAM test,
Robocurve owns and must supply its `inspect-robots-yam` package, validated limits, an attended
operator gate, a working e-stop, and a rehearsed termination procedure. DreamScale's adapter calls
neither the embodiment nor hardware directly, and compatibility or serving evidence must not be
reported as physical safety or effectiveness evidence.
