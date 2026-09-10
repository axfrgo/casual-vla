# Fortifiers Apprenticeship

Fortifiers is a persistent operational-teaching system. Aegis compiles human
corrections into versioned, auditable skill rules and applies them while an
embodiment plans and executes actions.

## Current validation

The TypeScript teaching and symbolic-contact testbed passes its automated test
suite. The measured symbolic results are explicitly not MuJoCo, VLA, robotics,
or Intel benchmark results.

## MuJoCo integration slice

The physics-backed environment lives in [`sim/mujoco`](sim/mujoco/README.md).
It currently provides a reproducible dinner-table scene, two official SO-101
arms, camera rendering, drawer motion, structured observations, and scene
perturbation controls. Policy integration, grasp/place task execution, and
OpenVINO benchmarking remain open work.

```powershell
uv sync --project sim/mujoco
uv run --project sim/mujoco fortifiers-mujoco-eval --episodes 10
```

## Browser app

```powershell
pnpm install
pnpm dev
```

See [`docs/CHALLENGE_READINESS.md`](docs/CHALLENGE_READINESS.md) for the honest
challenge gap analysis.
