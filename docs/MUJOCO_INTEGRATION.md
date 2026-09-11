# MuJoCo integration status

## Implemented in this slice

- Pinned `mujoco==3.13.0` Python dependency.
- MJCF scene with table, drawer, clean/place zones, plate, cup, fork, spoon,
  overview camera, and two namespaced official SO-101 arms.
- Deterministic reset by seed.
- Placement, mass, friction, object-scale, lighting, and background controls.
- Structured observations for qpos, qvel, arm joints, grippers, object poses,
  camera pixels, task progress, grasp state, and MuJoCo contacts.
- Joint-target, drawer-open, drawer-close, Cartesian move-to, grasp, release,
  and hand-off actions.
- Ten-episode smoke evaluation command.
- Vendored official SO-101 MJCF/URDF and STL assets from the SO-ARM100
  `Simulation/SO101` directory, with the source license preserved.
- Dinner-table task graph with an explicit two-arm hand-off requirement.
- Closed-loop deterministic dinner-table policy with live post-action
  observation feedback.
- Inactive-at-reset MuJoCo weld slots with grasp-time local anchors, measured
  two-jaw contact/lift gates, placement checks, and a persisted
  teach/retry/reset/remember harness.
- Fail-closed VLA protocol, JSONL process bridge, and OpenVINO benchmark command.

## Explicit limitations

The official single-arm asset is wired into the combined scene twice with
namespaced joints and actuators. The current controller is a deterministic
baseline, not a neural VLA. The task still seeds utensils on the table rather
than modeling literal drawer retrieval. MuJoCo contact and post-contact
retention are simulation evidence only; they do not establish hardware contact
or hardware benchmark performance.
LeRobot training, OpenVINO benchmarking, challenge SDK binding, and Core Ultra
hardware evidence remain open before claiming challenge success.

The TypeScript browser app remains on the symbolic testbed. A process or HTTP
bridge should be added only after the final MuJoCo action and observation schema
is stable; silently substituting one environment for the other would invalidate
the evaluation.
