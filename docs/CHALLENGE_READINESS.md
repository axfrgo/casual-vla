# Intel Physical AI Online Challenge Readiness

Source: `Online_Physical_AI_Challenge_Online (1).pdf` supplied with the project.

## Requirement comparison

| Challenge requirement | Current project | Status |
| --- | --- | --- |
| Natural-language task instruction | Typed lessons and commands are compiled into a validated skill contract | Implemented |
| Multi-modal reasoning from camera observations | MuJoCo exposes RGB/state and a closed-loop baseline consumes structured observations; no vision/VLA policy yet | Partial |
| Dual SO-101 arms | Two namespaced official SO-101 MJCF copies are driven through MuJoCo position actuators; challenge hardware calibration is not verified | Partial |
| MuJoCo simulation | Reproducible MuJoCo 3.13.0 scene, task state, measured grasp/lift gates, post-contact retention, and policy evaluation are implemented | Partial |
| Dinner-table workflow | Drawer opening, utensil/plate/cup pick-place state, and final task verification run in MuJoCo; utensils are currently seeded on the table rather than retrieved from the drawer | Partial |
| Hand-off and complementary bimanual actions | Two-jaw source/receiver contact, post-contact retention switching, and a right-to-left cup hand-off are verified; hardware contact remains unverified | Partial |
| Perturbation across placement, weight, friction, lighting, shape, background | The closed-loop baseline completes a reproducible 10-seed evaluation spanning all six scene knobs | Partial |
| Policy training or fine-tuning with LeRobot-compatible tooling | No VLA training pipeline | Not implemented |
| OpenVINO optimization and Intel device benchmark | No model conversion, device selection, or latency benchmark | Not implemented |
| Intel Core Ultra demonstration | Not verified in this environment | Not implemented |
| Reproducible repository and evaluation | Source, tests, README, pinned MuJoCo package, closed-loop evaluator, and teach/retry/reset/remember harness exist | Partial |
| 10-seed demonstration video | No video artifact | Not implemented |
| Technical architecture summary | This document plus the existing handoff documentation | Implemented |

## What is genuinely demonstrated today

The strongest completed contribution is now a connected vertical slice: the browser still provides the competition-facing mission surface and the Aegis teaching workspace, while `sim/mujoco` provides a real observation-driven controller over two official SO-101 model copies. The controller advances the drawer and arm actuators, gates every grasp on measured two-jaw contact and lift, then uses an explicitly inactive-at-reset MuJoCo weld as validated post-contact retention for long transports and hand-off. It verifies releases and a bimanual hand-off, and completes the six-step task across a reproducible ten-seed perturbation sweep (10/10 in the current run). The apprenticeship harness also records a pre-teaching hand-off failure, compiles a human correction into `handoff_staging`, retries successfully, resets into a new seed, and succeeds from persisted memory.

The current evidence is MuJoCo simulation evidence for a deterministic baseline. It is not evidence of a neural VLA, LeRobot training, physical manipulation, Intel acceleration, or Core Ultra performance.

## Required implementation before claiming challenge compliance

1. Bind the adapter to the official challenge action/observation schema and calibrate the task workspace.
2. Replace the seeded-table utensil shortcut with literal drawer retrieval and validate real gripper contact/attachment and hand-off behavior.
3. Connect the RGB/state observations to a permitted VLA or imitation policy through the existing replaceable inference interface.
4. Re-run the perturbation evaluation for that connected policy across placement, mass, friction, lighting, shape, and background for at least 10 seeds.
5. Add OpenVINO conversion/benchmark tooling for the selected model and record device, precision, latency, throughput, and task success.
6. Run the final simulation and benchmark on Intel Core Ultra Series 2/3 hardware.
7. Produce benchmark output and the 10-seed demonstration video required by the submission package.

## Integration boundary

The project intentionally keeps the `IntelBimanualAdapter` fail-closed until the official Intel/VLA endpoint or SDK, observation schema, action schema, reset API, and safety-stop behavior are supplied. The adapter must never silently substitute the symbolic testbed for a real MuJoCo or Intel run.
