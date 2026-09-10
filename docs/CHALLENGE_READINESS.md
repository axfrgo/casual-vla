# Intel Physical AI Online Challenge Readiness

Source: `Online_Physical_AI_Challenge_Online (1).pdf` supplied with the project.

## Requirement comparison

| Challenge requirement | Current project | Status |
| --- | --- | --- |
| Natural-language task instruction | Typed lessons and commands are compiled into a validated skill contract | Implemented |
| Multi-modal reasoning from camera observations | MuJoCo environment now exposes RGB camera frames and state; no vision/VLA policy yet | Partial |
| Dual SO-101 arms | MuJoCo scene now uses two namespaced copies of the official SO-101 MJCF asset and actuators | Partial |
| MuJoCo simulation | Reproducible MuJoCo 3.13.0 dinner-table scene and smoke evaluation are implemented | Partial |
| Dinner-table workflow | Table, drawer, utensils, plate, cup, and placement zones exist; task execution is not implemented | Partial |
| Hand-off and complementary bimanual actions | Two-arm model and joint controls exist; no grasp, hand-off, or pour task yet | Partial |
| Perturbation across placement, weight, friction, lighting, shape, background | All requested scene knobs are exposed in the MuJoCo reset API; no end-to-end policy evaluation yet | Partial |
| Policy training or fine-tuning with LeRobot-compatible tooling | No VLA training pipeline | Not implemented |
| OpenVINO optimization and Intel device benchmark | No model conversion, device selection, or latency benchmark | Not implemented |
| Intel Core Ultra demonstration | Not verified in this environment | Not implemented |
| Reproducible repository and evaluation | Source, tests, README, pinned MuJoCo package, and scene smoke evaluation exist | Partial |
| 10-seed demonstration video | No video artifact | Not implemented |
| Technical architecture summary | This document plus the existing handoff documentation | Implemented |

## What is genuinely demonstrated today

The strongest completed contribution is still the persistent operational-teaching layer: a human correction is compiled into explicit transition, prohibition, and preference rules; the controller uses those rules for counterfactual action filtering; contact provenance is recorded; the skill is persisted and versioned; and the learned doctrine generalizes across seeded symbolic scenes. A first physics-backed MuJoCo scene is now also available under `sim/mujoco`, with camera rendering and deterministic perturbations.

The current evidence is not evidence of physical manipulation, MuJoCo physics, VLA perception, or Intel acceleration.

## Required implementation before claiming challenge compliance

1. Implement the MuJoCo adapter/action schema for the official arms and calibrate the task workspace.
2. Add a dinner-table task graph: drawer open, utensil retrieval, plate and cup placement, and at least one hand-off or complementary dual-arm action.
3. Connect the RGB/state observations to a permitted VLA or imitation policy through a replaceable inference interface.
4. Evaluate that policy across placement, mass, friction, lighting, shape, and background perturbations for at least 10 seeds.
5. Add OpenVINO conversion/benchmark tooling for the selected model and record device, precision, latency, throughput, and task success.
6. Run the final simulation and benchmark on Intel Core Ultra Series 2/3 hardware.
7. Produce benchmark output and the 10-seed demonstration video required by the submission package.

## Integration boundary

The project intentionally keeps the `IntelBimanualAdapter` fail-closed until the official Intel/VLA endpoint or SDK, observation schema, action schema, reset API, and safety-stop behavior are supplied. The adapter must never silently substitute the symbolic testbed for a real MuJoCo or Intel run.
