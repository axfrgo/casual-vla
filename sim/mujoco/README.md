# Fortifiers MuJoCo environment

This is the first physics-backed integration slice for the Intel bimanual
challenge track. It provides:

- a MuJoCo 3.13.0 MJCF scene;
- two visible official SO-101 arms using the downloaded meshes;
- dinner-table objects, drawer, placement zones, and overview camera;
- deterministic object placement and perturbation controls;
- joint-target and drawer actions;
- structured observations containing camera, joint, gripper, object, and contact data.
- the official SO-101 MJCF/URDF and STL mesh assets under `assets/so101/SO101`.
- a dinner-table task graph, fail-closed VLA interface, and JSONL process bridge.

The combined dinner-table scene now uses the official SO-101 MJCF asset twice
with namespaced joints and actuators. Grasp/place planning, VLA inference, and
OpenVINO benchmarking are the next integration layers; this package does not
claim those are complete.

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

It accepts only JSON commands (`reset`, `observe`, `step`, `task_plan`, and
`close`) over stdin/stdout. The task graph is available independently through
`task_plan` and includes a required two-arm hand-off. It does not pretend that
the hand-off succeeds until a real grasp controller is connected.

## OpenVINO benchmark

After supplying a real exported model:

```powershell
uv sync --project sim/mujoco --extra edge
uv run --project sim/mujoco --extra edge python -m fortifiers_mujoco.benchmark_openvino model.xml --device CPU --output results/openvino.json
```

The benchmark reports device, input precision, latency percentiles, and
throughput. Task success remains `null` until the policy is connected to the
MuJoCo evaluation loop.
