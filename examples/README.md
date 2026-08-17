# Examples

Reference code for running Dropbear-hosted **DreamZero-YAM** on a physical
bimanual YAM through Inspect Robots, using this adapter.

See the [Inspect Robots guide](https://docs.dropbear.dreamscalelabs.com/guides/inspect-robots)
for the full integration reference.

Everything here is adapted from a run that completed against production, not
from the docs: 120 steps, 51 model-sourced actions, GPU inference 324.9 ms p50,
`wss_tunnel` to `us-west-2`, `dropbear 0.1.0a9` / `inspect-robots 0.53.1` /
`inspect-robots-dropbear 0.1.7`.

> **These examples are deliberately incomplete, and none of it has run on a
> physical YAM.** The run behind them used a stub arm: synthetic frames and a
> perfect-tracking executor. What is proven is the *serving* path — install,
> auth, session lifecycle, observation upload, action delivery, timing — which
> is independent of the robot.
>
> The hardware half is yours, and we cannot anticipate your setup: camera
> drivers and their real capture clocks, CAN bus and joint ordering, calibration,
> limits, and the gripper convention on your arm. `yam_embodiment.py` is a
> skeleton of the *contract*, not a driver. Expect to write real code where it
> says so, and expect the first hardware bring-up to surface things this file
> does not mention.

| file | what it is |
| --- | --- |
| `run_eval.py` | A complete evaluation: resolve the policy, run, close, then check where the actions came from |
| `yam_embodiment.py` | The observation and action contract, as a working skeleton |

## Setup

```bash
uv add inspect-robots-dropbear
export DROPBEAR_API_KEY="<your key>"
dropbear status --model dreamzero-yam
```

Then edit `TASK_NAME` and `EMBODIMENT_NAME` in `run_eval.py` to your registered
names. Your task and embodiment do not change to use Dropbear — only the policy
does.

## The three things that actually bite

**1. A green run can contain no model actions.** Inspect reports success when a
trial completes without raising, and `episode_length` counts steps rather than
inference. An evaluation where the robot held position for all 120 steps looks
identical in the EvalLog to one the model drove. We shipped exactly that result
before checking. `run_eval.py` prints `action_source` after every run for this
reason; if you take one thing from these examples, take that check.

**2. Your control loop must pace itself.** DreamZero-YAM returns 24 actions
covering 0.8 s at 30 Hz, and the round trip is several hundred milliseconds. A
loop that steps as fast as the code allows finishes the episode before any chunk
can arrive, and every action is a hold. Physical hardware paces itself for free;
anything simulated or stubbed has to be told to.

**3. Keep `observe()` cheap.** Dropbear samples observations on the *capture*
clock, not once per action, because the sampler always wants the freshest frame.
An `observe()` that takes anything like a frame period starves the loop that
receives actions. A 31 ms observation callback on a 33 ms budget was enough to
stall a run completely, with no error — it simply stopped delivering.

## Reading the numbers

The adapter writes a sidecar per trial at
`logs/dropbear/<run_id>/<scene>-e<epoch>.jsonl`, and records its path at
`TrialRecord.metadata["dropbear_telemetry"]`. Server-stamped fields arrive with
the action, so they are measurements rather than client guesses:

- `server_inference_ms` — GPU time
- `obs_to_action_ms` — the full round trip
- `action_source` — `model` or `hold`, per step

`obs_to_action_ms − server_inference_ms` is network, encode, decode, and
scheduling combined. It is a residual, not a measurement, and it is dominated by
distance to `us-west-2`. From Sydney it was ~250 ms p50; co-locating the client
near the region moves it far more than any client-side tuning.

Sidecars carry no action vectors, images, credentials, or endpoints, so they are
safe to attach to a bug report.

## Safety

These examples exercise the software path. They do not authorize an unattended
physical run. Validated joint limits, an attended operator gate, a working
e-stop, and a rehearsed termination procedure remain yours. Serving evidence is
not safety evidence.
