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
arms, camera rendering, drawer motion, structured observations, scene
perturbation controls, a closed-loop deterministic task policy, runtime grasp
constraints, placement verification, and a bimanual cup hand-off. The policy is
an executable baseline, not a VLA; challenge SDK binding, literal drawer
retrieval, validated hardware contact, and OpenVINO benchmarking remain open.

```powershell
uv sync --project sim/mujoco
uv run --project sim/mujoco fortifiers-mujoco-eval --episodes 10
```

Run the connected MuJoCo policy and write a reproducible trace:

```powershell
uv run --project sim/mujoco python -m fortifiers_mujoco.evaluate_policy --episodes 10 --output sim/mujoco/results/mujoco-policy-eval.json
uv run --project sim/mujoco python -m fortifiers_mujoco.apprenticeship
```

## Browser app

```powershell
pnpm install
pnpm dev
```

## Deploy to Vercel

Import this repository in Vercel and use the detected Next.js settings. No
custom build or output-directory override is required. Add
`SPEECHMATICS_API_KEY` in the Vercel project environment if voice transcription
should be enabled; text teaching works without it.

To verify the same production build locally:

```powershell
pnpm build
pnpm start
```

Runtime skill libraries and evaluation reports are written to the ignored
`data/` and `results/` directories. Re-run `pnpm evaluate` to generate fresh
symbolic-testbed evidence; generated outputs are not committed as source.

See [`docs/CHALLENGE_READINESS.md`](docs/CHALLENGE_READINESS.md) for the honest
challenge gap analysis.
