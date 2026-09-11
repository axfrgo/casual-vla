# Fortifiers MuJoCo environment

This is the first physics-backed integration slice for the Intel bimanual
challenge track. It provides:

- a MuJoCo 3.13.0 MJCF scene;
- two visible official SO-101 arms using the downloaded meshes;
- dinner-table objects, drawer, placement zones, and overview camera;
- deterministic object placement and perturbation controls;
- joint-target, drawer, Cartesian move-to, grasp, release, and hand-off actions;
- structured observations containing camera, joint, gripper, object, task, and contact data;
- the official SO-101 MJCF/URDF and STL mesh assets under `assets/so101/SO101`.
- a dinner-table task graph, closed-loop deterministic baseline, fail-closed VLA interface, and JSONL process bridge.

The combined dinner-table scene now uses the official SO-101 MJCF asset twice
with namespaced joints and actuators. The baseline drives those actuators,
requires measured two-jaw contact and lift, and then uses an inactive-at-reset
MuJoCo weld as validated post-contact retention for long transport. It is not a
neural VLA and does not claim validated hardware contact.

## Setup

From the repository root:

```powershell
uv sync --project sim/mujoco
```

## Smoke test

```powershell
uv run --project sim/mujoco fortifiers-mujoco-eval --episodes 10 --output results/mujoco-smoke.json
```

The smoke test validates that ten randomized MuJoCo episodes load and expose
the required state. It is not a policy success score.

## Closed-loop policy evaluation

```powershell
uv run --project sim/mujoco python -m fortifiers_mujoco.evaluate_policy --episodes 10 --output results/mujoco-policy-eval.json
```

The policy consumes each post-action observation and terminates only when the
MuJoCo task state verifies all six steps. The current evaluator varies
placement, mass, friction, object scale, lighting, and background. Its score is
simulation evidence for a deterministic baseline, not VLA or hardware evidence.

## Teach, retry, reset, remember

```powershell
uv run --project sim/mujoco python -m fortifiers_mujoco.apprenticeship
```

This records a bounded hand-off correction in `results/dinner-skill-memory.json`,
retries the same policy, resets into a new seed, and verifies that the persisted
rule is used again.

## Visual inspection

If a desktop OpenGL context is available:

```powershell
uv run --project sim/mujoco python -m mujoco.viewer --mjcf=sim/mujoco/models/dual_arm_dinner_table.xml
```

MuJoCo's Python binding uses `MjModel.from_xml_path`, `MjData`, `mj_step`, and
`Renderer`; those are the low-level primitives used by this adapter.

## Bridge and task graph

The safe process bridge can be started with:

```powershell
uv run --project sim/mujoco python -m fortifiers_mujoco.bridge_server
```

It accepts only JSON commands (`reset`, `observe`, `step`, `policy_step`,
`policy_run`, `task_plan`, and `close`) over stdin/stdout. The task graph is
available independently through `task_plan` and includes a required two-arm
hand-off. Task-level actions now return live task state, including held
objects, verified placements, and the hand-off result. `policy_step` and
`policy_run` exercise the same closed-loop policy through the process boundary.

## OpenVINO benchmark

After supplying a real exported model:

```powershell
uv sync --project sim/mujoco --extra edge
uv run --project sim/mujoco --extra edge python -m fortifiers_mujoco.benchmark_openvino model.xml --device CPU --output results/openvino.json
```

The benchmark reports device, input precision, latency percentiles, and
throughput. Task success remains `null` because no exported VLA model is
connected to this benchmark; the deterministic baseline is evaluated by
`evaluate_policy` instead.
