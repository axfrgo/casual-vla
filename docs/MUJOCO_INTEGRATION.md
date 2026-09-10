# MuJoCo integration status

## Implemented in this slice

- Pinned `mujoco==3.13.0` Python dependency.
- MJCF scene with table, drawer, clean/place zones, plate, cup, fork, spoon,
  overview camera, and two namespaced official SO-101 arms.
- Deterministic reset by seed.
- Placement, mass, friction, object-scale, lighting, and background controls.
- Structured observations for qpos, qvel, arm joints, grippers, object poses,
  camera pixels, and MuJoCo contacts.
- Joint-target, drawer-open, drawer-close, and no-op actions.
- Ten-episode smoke evaluation command.
- Vendored official SO-101 MJCF/URDF and STL assets from the SO-ARM100
  `Simulation/SO101` directory, with the source license preserved.
- Dinner-table task graph with an explicit two-arm hand-off requirement.
- Fail-closed VLA protocol, JSONL process bridge, and OpenVINO benchmark command.

## Explicit limitations

The official single-arm asset is wired into the combined scene twice with
namespaced joints and actuators. The environment currently does not implement
grasp attachment,
drawer-to-utensil retrieval, plate/cup
placement verification, hand-off behavior, a VLA policy, LeRobot training, or
OpenVINO benchmarking. Those must be added before claiming challenge success.

The TypeScript browser app remains on the symbolic testbed. A process or HTTP
bridge should be added only after the final MuJoCo action and observation schema
is stable; silently substituting one environment for the other would invalidate
the evaluation.
