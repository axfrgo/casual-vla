# Intel Physical AI Online Challenge Readiness

Source: `Online_Physical_AI_Challenge_Online (1).pdf` supplied with the project.

## Requirement comparison

| Challenge requirement | Current project | Status |
| --- | --- | --- |
| Natural-language task instruction | Typed lessons and commands are compiled into a validated skill contract | Implemented |
| Multi-modal reasoning from camera observations | Three RGB views, 21-value robot/task state, language features, bounded neural action decoding, and a native MuJoCo evaluator are connected; direct joint behavior cloning is `0/10`, while a separately labeled neural intent gate plus measured controller completes `10/10` | Partial |
| Dual SO-101 arms | Two namespaced official SO-101 MJCF copies are driven through MuJoCo position actuators; challenge hardware calibration is not verified | Partial |
| MuJoCo simulation | Reproducible MuJoCo 3.13.0 scene, task state, measured grasp/lift gates, post-contact retention, and policy evaluation are implemented | Partial |
| Dinner-table workflow | Drawer opening, native tray stow/release, utensil/plate/cup pick-place state, and final task verification run in MuJoCo; official challenge task binding remains open | Partial |
| Hand-off and complementary bimanual actions | Two-jaw source/receiver contact, post-contact retention switching, and a right-to-left cup hand-off are verified; hardware contact remains unverified | Partial |
| Perturbation across placement, weight, friction, lighting, shape, background | The closed-loop baseline completes a reproducible 10-seed evaluation spanning all six scene knobs | Partial |
| Policy training or fine-tuning with LeRobot-compatible tooling | Ten native expert episodes produce 232 RGB/state/action frames; direct camera/state/language behavior cloning and a separate camera/state/language neural intent checkpoint are trained, loaded, and evaluated through native task gates | Partial |
| OpenVINO optimization and Intel device benchmark | The exact compact checkpoint exports to static FP32 OpenVINO IR; CPU inference measures 2.39 ms mean, 3.18 ms p95, and 418 FPS, while the matching task smoke is reported separately at 0/1 | Partial |
| Intel Core Ultra demonstration | Not verified in this environment | Not implemented |
| Reproducible repository and evaluation | Source, tests, README, pinned MuJoCo package, closed-loop evaluator, and teach/retry/reset/remember harness exist | Partial |
| 10-seed demonstration video | H.264 MP4 overview recording of native MuJoCo baseline seeds `1001`–`1010`, with JSON sidecar and per-seed task metadata | Implemented (simulation) |
| Technical architecture summary | This document plus the existing handoff documentation | Implemented |

## What is genuinely demonstrated today

The strongest completed contribution is now a connected vertical slice: the browser still provides the competition-facing mission surface and the Aegis teaching workspace, while `sim/mujoco` provides a real observation-driven controller over two official SO-101 model copies. The controller opens a native drawer tray whose fork and spoon are carried by measured MuJoCo stow constraints until retrieval begins, then gates every grasp on measured two-jaw contact and lift before using an explicitly inactive-at-reset MuJoCo weld as validated post-contact retention for long transports and hand-off. It verifies releases and a bimanual hand-off, and completes the six-step task across a reproducible ten-seed perturbation sweep when the MuJoCo environment is installed. The apprenticeship harness also records a pre-teaching hand-off failure, compiles a human correction into `handoff_staging`, retries successfully, resets into a new seed, and succeeds from persisted memory.

The current evidence has three clearly separated tracks. The deterministic
baseline completes the native MuJoCo task across ten seeds. A direct compact
camera/state/language imitation checkpoint is a real native neural report but
is intentionally recorded as a failure (`0/10` complete: drawer-open reached,
fork retrieval not reached in any episode). Separately, a neural task-intent
checkpoint gates the same measured low-level controller on every decision and
completes the native task across ten seeds (`10/10`, zero intent rejections).
That guided score is a neural high-level policy plus verified control-stack
result, not an end-to-end neural joint-target score. None of these artifacts
establish physical manipulation, Intel Core Ultra performance, or compliance
with an official challenge SDK that has not been supplied.

The browser links the metadata and reports so a reviewer can inspect the exact
artifact chains: direct neural checkpoint → native MuJoCo evaluation, neural
intent checkpoint → gated native completion, and OpenVINO IR → inference-only
CPU benchmark. A fast benchmark is never presented as task success. The
ten-seed video is linked from the native evidence ledger and is labeled as
deterministic MuJoCo simulation evidence.

### Reproducible MuJoCo evidence

On 2026-09-11, the installed offline environment ran the following command
from the repository root:

```powershell
$env:UV_CACHE_DIR = (Join-Path (Get-Location) 'tmp\uv-cache')
uv run --offline --project sim/mujoco python -m fortifiers_mujoco.evaluate_policy --episodes 10 --start-seed 1001 --max-actions 180
```

Result: `10/10` completed, `1.0` success rate, seeds `1001` through `1010`,
`0` failed task actions in every episode, and `23–24` policy actions per
episode. The run includes literal drawer-tray retrieval: the drawer carries
the fork and spoon under native MuJoCo stow welds, the controller must reach
the measured grasp target, and the stow weld is released only after the
two-jaw contact/lift gate passes. This is a deterministic baseline result;
it must not be presented as the challenge's neural-policy or hardware result.

The compact per-seed projection of this run is checked in at
`public/evidence/mujoco-policy-eval.json` and linked from the browser's
baseline card. The full action/contact trace can be regenerated with the same
command using an output path under `sim/mujoco/results/`.

### Neural and OpenVINO evidence

The checked-in neural metadata and reports were produced from ten native baseline episodes
(232 frames, three RGB views, 21 state/task values, and 13 actuator targets).
The compact checkpoint is `tmp/imitation-policy-v7.pt`; its metadata is exposed
to the browser as `public/evidence/imitation-policy-checkpoint.json`.

A fresh 10-episode collection (230 frames, seeds `3001`–`3010`) was also used
for a reproducible CUDA training run on a temporary Google Cloud `g2-standard-8`
VM with one NVIDIA L4. The retrieved v10 checkpoint was measured through the
local native renderer on seed `4001`: `0/1`, stopping at `retrieve_fork` after
40 actions with zero rejected actions. The provenance sidecar is
`public/evidence/imitation-policy-gpu-v10.json`; it is GPU-training evidence,
not Core Ultra evidence and not a direct-policy task pass.

The native neural evaluation was run across seeds `1001`–`1010` with the same
camera rendering and MuJoCo task gates used by the baseline:

```powershell
uv run --offline --project sim/mujoco --extra vla fortifiers-mujoco-eval-imitation tmp/imitation-policy-v7.pt --episodes 10 --start-seed 1001 --max-actions 100 --patience-actions 24 --width 160 --height 120 --output tmp/imitation-evaluation-v7-10seed.json
```

It produced `0/10` completed episodes, `0` rejected actions, and reached
`drawer-open` before stopping at `retrieve-fork` after 40 actions in every
episode. The same checkpoint was
exported to static FP32 OpenVINO IR and evaluated through the native task; that
matching one-seed smoke also produced `0/1`, with `42.91 ms` mean and `49.73 ms` p95
per-step OpenVINO-connected inference in this CPU environment. The separate
100-iteration OpenVINO inference benchmark measured `2.39 ms` mean, `3.18 ms`
p95, and `417.7 FPS` on `CPU`; it contains no task-success claim. These reports
are linked from the browser's neural evidence ledger.

The neural-guided completion evaluation uses a separately trained intent head
and fails closed when its predicted step disagrees with the live task index or
falls below the confidence threshold:

```powershell
uv run --offline --project sim/mujoco --extra vla fortifiers-mujoco-train-neural-intent tmp/vla-demonstrations-10-phase-onehot/manifest.json --output tmp/neural-intent-v1.pt
uv run --offline --project sim/mujoco --extra vla fortifiers-mujoco-eval-neural-guided tmp/neural-intent-v1.pt --episodes 10 --start-seed 1001 --confidence-threshold 0.5 --output public/evidence/neural-guided-eval-10seed.json
```

Result: `10/10` completed, `0` intent rejections, and `23–24` task actions per
episode. The report attributes low-level execution to
`closed-loop-dinner-policy-v1`; this is the defensible neural-guided system
result, not the direct joint behavior-cloning result above.

## Remaining work before claiming challenge compliance

1. Bind the adapter to the official challenge action/observation schema and calibrate the task workspace.
2. Validate the native drawer-tray retrieval, real gripper contact/attachment, and hand-off behavior against the official challenge scene/schema.
3. Improve or replace the direct compact policy until it completes the full native task without relying on the verified controller's planned motion primitives. The current direct checkpoint is `0/10`; the neural-guided stack is `10/10` but is a separate claim.
4. If the submission requires a high-capacity VLA, fine-tune and select a permitted checkpoint with the LeRobot-compatible export, then run that actual checkpoint through the native adapter. The completed CUDA run only trains the compact BC baseline; it does not substitute for a selected foundation VLA.
5. Re-run OpenVINO task and inference benchmarks for the selected submission model on the target device.
6. Run the final simulation and benchmark on Intel Core Ultra Series 2/3 hardware.
7. Attach the generated ten-seed video and its sidecar to the final submission package, then verify the submission's required encoding and presentation format. The local MP4 is now generated; the remaining work is package-specific verification.

## Integration boundary

The project intentionally keeps the `IntelBimanualAdapter` fail-closed until the official Intel/VLA endpoint or SDK, observation schema, action schema, reset API, and safety-stop behavior are supplied. The adapter must never silently substitute the symbolic testbed for a real MuJoCo or Intel run.
