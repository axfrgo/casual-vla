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
arms, camera rendering, drawer motion, literal drawer-tray utensil retrieval,
structured observations, scene perturbation controls, a closed-loop
deterministic task policy, runtime grasp constraints, placement verification,
a bimanual cup hand-off, four camera views, and a camera-language VLA adapter.
The baseline remains an executable non-neural reference. The VLA collector,
LeRobot export, compact imitation checkpoint, checkpoint evaluator, OpenVINO
export, and benchmark are wired as separate evidence paths. The checked-in
ten-seed neural evaluation is a real failure report: all ten episodes reach
drawer-open but stop before fork retrieval. An Intel Core Ultra run and a
permitted high-capacity VLA remain separate submission work.

```powershell
uv sync --project sim/mujoco
uv run --project sim/mujoco fortifiers-mujoco-eval --episodes 10
```

Build the camera/state/action dataset and train/evaluate the compact neural
policy:

```powershell
uv sync --project sim/mujoco --extra vla
uv run --project sim/mujoco fortifiers-mujoco-collect-vla --episodes 50 --output sim/mujoco/results/vla-demonstrations
uv run --project sim/mujoco fortifiers-mujoco-export-lerobot sim/mujoco/results/vla-demonstrations/manifest.json --repo-id <user>/fortifiers-dinner-table-v1 --root sim/mujoco/results/lerobot-dinner-table
uv run --project sim/mujoco --extra vla fortifiers-mujoco-train-imitation sim/mujoco/results/vla-demonstrations/manifest.json --output sim/mujoco/results/imitation-policy.pt
uv run --project sim/mujoco --extra vla fortifiers-mujoco-eval-imitation sim/mujoco/results/imitation-policy.pt --episodes 10 --output sim/mujoco/results/imitation-evaluation.json
uv run --project sim/mujoco --extra vla fortifiers-mujoco-train-neural-intent sim/mujoco/results/vla-demonstrations/manifest.json --output sim/mujoco/results/neural-intent-policy.pt
uv run --project sim/mujoco --extra vla fortifiers-mujoco-eval-neural-guided sim/mujoco/results/neural-intent-policy.pt --episodes 10 --output sim/mujoco/results/neural-guided-evaluation.json
uv sync --project sim/mujoco --extra edge --extra vla
uv run --project sim/mujoco --extra edge --extra vla fortifiers-mujoco-export-imitation-openvino sim/mujoco/results/imitation-policy.pt --output sim/mujoco/results/imitation-policy.xml
uv run --project sim/mujoco --extra edge --extra vla fortifiers-mujoco-eval-openvino sim/mujoco/results/imitation-policy.xml --episodes 10 --device CPU --output sim/mujoco/results/openvino-evaluation.json
uv run --project sim/mujoco --extra edge --extra vla fortifiers-mujoco-benchmark-openvino sim/mujoco/results/imitation-policy.xml --device CPU --output sim/mujoco/results/openvino-benchmark.json
uv sync --project sim/mujoco --extra video
uv run --project sim/mujoco --extra video fortifiers-mujoco-record-video --output public/evidence/dinner-table-10-seed-demo.mp4 --episodes 10 --start-seed 1001
```

The checked-in browser evidence includes the full ten-seed neural evaluation,
the matching OpenVINO task report and CPU inference benchmark. It is
deliberately not a competition pass: the direct neural task result is `0/10`,
while the benchmark is inference-only. The native deterministic baseline
remains the separate `10/10` reference result. The neural-guided report is a
distinct `10/10` completion path: a neural intent gate runs on every decision,
while a measured controller performs the low-level contact-safe motion. It is
not an end-to-end neural joint-target pass.

The video command records native MuJoCo overview frames for seeds `1001`–`1010`
and writes a JSON sidecar next to the H.264 MP4. It is simulation evidence for
the deterministic baseline, not a browser capture, neural-policy video, or
hardware-robot recording.

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
challenge gap analysis and [`docs/COMPETITION_SUBMISSION.md`](docs/COMPETITION_SUBMISSION.md)
for the detailed architecture, provenance, reproduction commands, video
checklist, and competition evidence ledger.
