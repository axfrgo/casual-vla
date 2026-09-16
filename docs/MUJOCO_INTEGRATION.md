# MuJoCo integration status

## Implemented in this slice

- Pinned `mujoco==3.13.0` Python dependency.
- MJCF scene with table, drawer, clean/place zones, plate, cup, fork, spoon,
  four named camera views, and two namespaced official SO-101 arms.
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
- Literal drawer-tray retrieval: drawer motion carries the fork and spoon under
  native stow welds, and retrieval releases each stow weld only after measured
  grasp contact and lift validation.
- Fail-closed VLA protocol, camera-language SmolVLA adapter, RGB/state/action
  demonstration collector, LeRobot exporter, JSONL process bridge, and
  OpenVINO benchmark command.

## Explicit limitations

The official single-arm asset is wired into the combined scene twice with
namespaced joints and actuators. The current controller is a deterministic
baseline, not a neural VLA. Fork and spoon now begin in a native drawer tray
and are released from measured drawer stow constraints when retrieval starts.
MuJoCo contact and post-contact retention are simulation evidence only; they do
not establish hardware contact or hardware benchmark performance.
The compact camera/state/language imitation path is now executable end to end:
ten native demonstrations produced a real checkpoint, the checkpoint was
evaluated through native MuJoCo, and the exact weights were exported to static
FP32 OpenVINO IR. The ten-seed neural evaluation reaches drawer-open but not
fork retrieval (`0/10` complete), so it is a real artifact and failure report
rather than a claimed task pass. The connected OpenVINO CPU task smoke has the
same result; its separate inference-only benchmark measures 2.39 ms mean /
3.18 ms p95 / 417.7 FPS in this environment. A separate camera/state/language
neural intent checkpoint now gates the measured closed-loop controller on every
decision and completes the ten-seed native sweep (`10/10`, zero intent
rejections). That is a neural high-level policy plus verified controller result,
not an end-to-end neural joint-target pass. Challenge SDK binding, improvement
to a full direct neural task pass, and Core Ultra hardware evidence remain open
before claiming challenge success.

The TypeScript browser app remains a deliberate visual surface; the native
JSONL bridge is the integration boundary for MuJoCo policy evaluation. The
browser never substitutes animation for physics evidence.

The current deterministic baseline completes the six-step task for seeds
1001–1010 under the checked-in perturbation sweep. This is reproducible
MuJoCo baseline evidence only; it does not establish VLA, LeRobot, hardware,
Intel Core Ultra, or neural task performance. The neural and OpenVINO reports
are separate and linked from `public/evidence/`.
