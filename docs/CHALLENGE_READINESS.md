# Intel Physical AI Online Challenge Readiness

Source: `Online_Physical_AI_Challenge_Online (1).pdf` supplied with the project.

## Requirement comparison

| Challenge requirement | Current project | Status |
| --- | --- | --- |
| Natural-language task instruction | Typed lessons and commands are compiled into a validated skill contract | Implemented |
| Multi-modal reasoning from camera observations | Internal symbolic world observations; no camera ingestion or vision model | Partial |
| Dual SO-101 arms | Two logical grippers in the contact testbed; no SO-101 kinematics | Partial |
| MuJoCo simulation | Deterministic discrete contact testbed; explicitly not physics or MuJoCo | Not implemented |
| Dinner-table workflow | Isolation/quarantine workflow with protected clean samples | Not implemented |
| Hand-off and complementary bimanual actions | Planner supports two logical grippers but no real hand-off/pour task | Partial |
| Perturbation across placement, weight, friction, lighting, shape, background | Seeded object identity and count transfer only | Partial |
| Policy training or fine-tuning with LeRobot-compatible tooling | No VLA training pipeline | Not implemented |
| OpenVINO optimization and Intel device benchmark | No model conversion, device selection, or latency benchmark | Not implemented |
| Intel Core Ultra demonstration | Not verified in this environment | Not implemented |
| Reproducible repository and evaluation | Source, tests, deterministic evaluation, and deployment package exist | Partial |
| 10-seed demonstration video | No video artifact | Not implemented |
| Technical architecture summary | This document plus the existing handoff documentation | Implemented |

## What is genuinely demonstrated today

The strongest completed contribution is the persistent operational-teaching layer: a human correction is compiled into explicit transition, prohibition, and preference rules; the controller uses those rules for counterfactual action filtering; contact provenance is recorded; the skill is persisted and versioned; and the learned doctrine generalizes across seeded symbolic scenes.

The current evidence is not evidence of physical manipulation, MuJoCo physics, VLA perception, or Intel acceleration.

## Required implementation before claiming challenge compliance

1. Add a MuJoCo dual-SO-101 environment with camera observations, joint state, gripper state, and a randomized dinner-table scene.
2. Implement an adapter that maps MuJoCo observations/actions to the existing `EmbodimentAdapter` boundary.
3. Add a dinner-table task graph: drawer open, utensil retrieval, plate and cup placement, and at least one hand-off or complementary dual-arm action.
4. Add perturbation controls for placement, mass, friction, lighting, shape, and background, then evaluate at least 10 seeds.
5. Integrate a permitted VLA or imitation policy through a replaceable inference interface.
6. Add OpenVINO conversion/benchmark tooling for the selected model and record device, precision, latency, throughput, and task success.
7. Run the final simulation and benchmark on Intel Core Ultra Series 2/3 hardware.
8. Produce the reproducible README, benchmark output, and 10-seed demonstration video required by the submission package.

## Integration boundary

The project intentionally keeps the `IntelBimanualAdapter` fail-closed until the official Intel/VLA endpoint or SDK, observation schema, action schema, reset API, and safety-stop behavior are supplied. The adapter must never silently substitute the symbolic testbed for a real MuJoCo or Intel run.
