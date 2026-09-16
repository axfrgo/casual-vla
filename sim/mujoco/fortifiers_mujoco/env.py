from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised by the install check
    raise ImportError(
        "MuJoCo is not installed. Run `uv sync --project sim/mujoco` first."
    ) from exc

from .task import build_dinner_table_plan


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PACKAGE_ROOT / "models" / "official_dual_so101_dinner_table.xml"
PROXY_MODEL_PATH = PACKAGE_ROOT / "models" / "dual_arm_dinner_table.xml"
OFFICIAL_SO101_DIR = PACKAGE_ROOT / "assets" / "so101" / "SO101"
OFFICIAL_SO101_MJCF_PATH = OFFICIAL_SO101_DIR / "so101_new_calib.xml"
OBJECT_NAMES = ("plate", "cup", "fork", "spoon")
ARM_NAMES = ("left", "right")
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_roll", "wrist_pitch")
CAMERA_NAMES = ("overview", "overhead", "left_oblique", "right_oblique")
# The official SO-101 gripper joint closes toward the negative end of its
# calibrated range. The previous demo used the opposite sign, which made the
# jaw motion look like a grasp while the runtime equality hid the mistake.
GRIPPER_OPEN = 1.20
GRIPPER_CLOSED = -0.12
GRASP_APPROACH_DISTANCE = 0.12
GRASP_POSITION_TOLERANCE = 0.018
GRASP_ORIENTATION_TOLERANCE = 0.32
GRASP_CLOSE_STEPS = 100
GRASP_SETTLE_STEPS = 180
PREGRASP_STEPS = 80
ORIENTATION_WEIGHT = 0.12
RIGHT_CUP_MIN_APPROACH_Y = -0.175
GRASP_PINCH_OFFSETS = {
    # The calibrated pinch frame is centered on the live object body. The
    # fingertip pads are symmetric around that frame, so empirical offsets
    # would bias the jaws onto an edge.
    "left": np.array([0.0, 0.0, 0.0]),
    "right": np.array([0.0, 0.0, 0.0]),
}
GRASP_TARGET_OFFSETS = {
    # The plate uses a front-rim pinch point; placement must preserve that
    # same object-to-pinch transform so the plate center lands on target.
    "fork": np.array([0.0, 0.0, 0.0]),
    "spoon": np.array([0.0, 0.0, 0.0]),
    "plate": np.array([0.0, -0.08, 0.0]),
    # The right arm approaches the cup from the near side; preserving this
    # rim offset also keeps the hand-off receiver on the same contact frame.
    "cup": np.array([0.0, -0.12, 0.0]),
}
OBJECT_TARGETS = {
    "fork": np.array([-0.30, -0.24, 0.96]),
    "spoon": np.array([0.34, -0.28, 0.96]),
    "plate": np.array([-0.22, -0.16, 0.96]),
    "cup": np.array([0.00, -0.25, 0.96]),
}
PLACEMENT_TARGETS = {
    "fork": np.array([-0.30, -0.24, 0.792]),
    "spoon": np.array([0.34, -0.28, 0.792]),
    "plate": np.array([-0.22, -0.16, 0.805]),
    "cup": np.array([0.00, -0.25, 0.88]),
}
INITIAL_OBJECT_TARGETS = {
    "fork": np.array([-0.30, -0.24, 0.96]),
    "spoon": np.array([0.34, -0.28, 0.96]),
    "plate": np.array([-0.22, -0.16, 0.96]),
    "cup": np.array([0.50, -0.05, 0.96]),
}


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    normalized = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(normalized))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return normalized / norm


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = _normalize_quaternion(first)
    bw, bx, by, bz = _normalize_quaternion(second)
    return _normalize_quaternion(
        np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float64,
        )
    )


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    normalized = _normalize_quaternion(quaternion)
    return normalized * np.array([1.0, -1.0, -1.0, -1.0])


def _rotation_vector_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    error_quaternion = _quaternion_multiply(target, _quaternion_conjugate(current))
    if error_quaternion[0] < 0:
        error_quaternion = -error_quaternion
    scalar = float(np.clip(error_quaternion[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(scalar)
    vector_norm = float(np.linalg.norm(error_quaternion[1:]))
    if vector_norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return error_quaternion[1:] / vector_norm * angle


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to MuJoCo's wxyz quaternion."""

    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        return _normalize_quaternion(
            np.array(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        )
    diagonal = np.diag(rotation)
    index = int(np.argmax(diagonal))
    next_index = (index + 1) % 3
    last_index = (index + 2) % 3
    scale = 2.0 * np.sqrt(max(1e-12, 1.0 + diagonal[index] - diagonal[next_index] - diagonal[last_index]))
    quaternion = np.zeros(4, dtype=np.float64)
    quaternion[index + 1] = 0.25 * scale
    quaternion[0] = (rotation[last_index, next_index] - rotation[next_index, last_index]) / scale
    quaternion[next_index + 1] = (rotation[next_index, index] + rotation[index, next_index]) / scale
    quaternion[last_index + 1] = (rotation[last_index, index] + rotation[index, last_index]) / scale
    return _normalize_quaternion(quaternion)


def _z_rotation_quaternion(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)], dtype=np.float64)


@dataclass(frozen=True)
class PerturbationConfig:
    """Scene variables required by the challenge robustness evaluation."""

    placement_jitter: float = 0.12
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    object_scale: float = 1.0
    lighting_scale: float = 1.0
    background: str = "neutral"

    def validate(self) -> None:
        if self.placement_jitter < 0 or self.placement_jitter > 0.35:
            raise ValueError("placement_jitter must be between 0 and 0.35")
        for name in ("mass_scale", "friction_scale", "object_scale", "lighting_scale"):
            value = getattr(self, name)
            if value <= 0 or value > 4:
                raise ValueError(f"{name} must be greater than 0 and at most 4")
        if self.background not in {"neutral", "warm", "cool"}:
            raise ValueError("background must be neutral, warm, or cool")


@dataclass
class FortifiersMuJoCoEnv:
    """A reproducible MuJoCo scene with two official SO-101 arms.

    Physics, free-body objects, drawer motion, camera rendering, randomized
    scene state, calibrated pose control, fingertip contact, and task
    verification are real MuJoCo state. Neural policy inference remains a
    separate integration.
    """

    model_path: Path = MODEL_PATH
    frame_skip: int = 5
    width: int = 640
    height: int = 480
    model: mujoco.MjModel = field(init=False, repr=False)
    data: mujoco.MjData = field(init=False, repr=False)
    seed: int = field(init=False, default=0)
    perturbation: PerturbationConfig = field(init=False, default_factory=PerturbationConfig)
    _renderer: mujoco.Renderer | None = field(init=False, default=None, repr=False)
    _actuator_ids: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _joint_ids: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _body_ids: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _geom_ids: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _object_joint_ids: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _base_geom_friction: dict[int, np.ndarray] = field(init=False, default_factory=dict, repr=False)
    _base_geom_size: dict[int, np.ndarray] = field(init=False, default_factory=dict, repr=False)
    _base_body_mass: dict[int, float] = field(init=False, default_factory=dict, repr=False)
    _base_ambient: np.ndarray = field(init=False, repr=False)
    _site_ids: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _jaw_geom_ids: dict[str, list[int]] = field(init=False, default_factory=dict, repr=False)
    _plate_jaw_geom_ids: dict[str, list[int]] = field(init=False, default_factory=dict, repr=False)
    _cup_jaw_geom_ids: dict[str, list[int]] = field(init=False, default_factory=dict, repr=False)
    _retention_equality_ids: dict[tuple[str, str], int] = field(init=False, default_factory=dict, repr=False)
    _drawer_stow_equality_ids: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _pinch_base_quaternion: dict[str, np.ndarray] = field(init=False, default_factory=dict, repr=False)
    _task_index: int = field(init=False, default=0, repr=False)
    _held_by: dict[str, str | None] = field(init=False, default_factory=dict, repr=False)
    _placed: set[str] = field(init=False, default_factory=set, repr=False)
    _last_action_result: dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    _last_grasp_metrics: dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    _last_pose_goal: dict[str, dict[str, Any]] = field(init=False, default_factory=dict, repr=False)
    _last_safe_joint_targets: dict[str, np.ndarray] = field(init=False, default_factory=dict, repr=False)
    _physics_advanced_in_action: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._rng = np.random.default_rng(0)
        self._index_named_entities()
        mujoco.mj_forward(self.model, self.data)
        self._pinch_base_quaternion = {
            arm: _matrix_to_quaternion(self.data.site_xmat[self._site_ids[f"{arm}_pinch_site"]])
            for arm in ARM_NAMES
        }
        self._base_geom_friction = {
            geom_id: self.model.geom_friction[geom_id].copy()
            for geom_id in (self._geom_ids[f"{name}_geom"] for name in OBJECT_NAMES)
        }
        self._base_geom_size = {
            geom_id: self.model.geom_size[geom_id].copy()
            for geom_id in (self._geom_ids[f"{name}_geom"] for name in OBJECT_NAMES)
        }
        self._base_body_mass = {
            self._body_ids[name]: float(self.model.body_mass[self._body_ids[name]])
            for name in OBJECT_NAMES
        }
        self._base_ambient = self.model.vis.headlight.ambient.copy()
        self._task_plan = build_dinner_table_plan()
        self.reset(0)

    def _index_named_entities(self) -> None:
        for name in self._names(mujoco.mjtObj.mjOBJ_ACTUATOR):
            self._actuator_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
        for name in self._names(mujoco.mjtObj.mjOBJ_JOINT):
            self._joint_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
        for name in self._names(mujoco.mjtObj.mjOBJ_BODY):
            self._body_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
        for name in self._names(mujoco.mjtObj.mjOBJ_GEOM):
            self._geom_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
        for name in self._names(mujoco.mjtObj.mjOBJ_SITE):
            self._site_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, name
            )
        equality_names = set(self._names(mujoco.mjtObj.mjOBJ_EQUALITY))
        for arm in ARM_NAMES:
            for object_name in OBJECT_NAMES:
                equality_name = f"{arm}_{object_name}_retention"
                if equality_name not in equality_names:
                    raise ValueError(f"Model is missing required retention constraint {equality_name}")
                self._retention_equality_ids[(arm, object_name)] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
                )
        for object_name in ("fork", "spoon"):
            equality_name = f"drawer_{object_name}_stow"
            if equality_name not in equality_names:
                raise ValueError(f"Model is missing required drawer stow constraint {equality_name}")
            self._drawer_stow_equality_ids[object_name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
            )
        for arm in ARM_NAMES:
            self._jaw_geom_ids[arm] = [
                self._geom_ids[f"{arm}_fixed_finger_geom"],
                self._geom_ids[f"{arm}_moving_finger_geom"],
            ]
            self._plate_jaw_geom_ids[arm] = [
                self._geom_ids[f"{arm}_plate_fixed_finger_geom"],
                self._geom_ids[f"{arm}_plate_moving_finger_geom"],
            ]
            self._cup_jaw_geom_ids[arm] = [
                self._geom_ids[f"{arm}_cup_fixed_finger_geom"],
                self._geom_ids[f"{arm}_cup_moving_finger_geom"],
            ]
        for object_name in OBJECT_NAMES:
            joint_name = f"{object_name}_free"
            if joint_name not in self._joint_ids:
                raise ValueError(f"Model is missing required joint {joint_name}")
            self._object_joint_ids[object_name] = self._joint_ids[joint_name]

    def _names(self, object_type: mujoco.mjtObj) -> list[str]:
        counts = {
            mujoco.mjtObj.mjOBJ_ACTUATOR: self.model.nu,
            mujoco.mjtObj.mjOBJ_BODY: self.model.nbody,
            mujoco.mjtObj.mjOBJ_GEOM: self.model.ngeom,
            mujoco.mjtObj.mjOBJ_JOINT: self.model.njnt,
            mujoco.mjtObj.mjOBJ_SITE: self.model.nsite,
            mujoco.mjtObj.mjOBJ_EQUALITY: self.model.neq,
        }
        names: list[str] = []
        for index in range(counts[object_type]):
            name = mujoco.mj_id2name(self.model, object_type, index)
            if name:
                names.append(name)
        return names

    def reset(
        self,
        seed: int = 0,
        perturbation: PerturbationConfig | None = None,
    ) -> dict[str, Any]:
        """Reset and randomize one deterministic episode."""

        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed
        self.perturbation = perturbation or PerturbationConfig()
        self.perturbation.validate()
        self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.eq_active[:] = False
        self._task_index = 0
        self._held_by = {object_name: None for object_name in OBJECT_NAMES}
        self._placed = set()
        self._last_action_result = {"ok": True, "message": "episode reset"}
        self.data.ctrl[:] = 0
        self._last_grasp_metrics = {}
        self._last_pose_goal = {}
        self._last_safe_joint_targets = {}

        drawer_joint = self._joint_ids["drawer_slide"]
        # The drawer is closed under the table at the rear of its travel;
        # opening moves it toward the operator/front edge (qpos -> 0).
        self.data.qpos[self.model.jnt_qposadr[drawer_joint]] = 0.30

        # Plate and cup start on the tabletop, while the utensils start in the
        # closed drawer tray. Orientation is randomized around z.
        positions = self._sample_object_positions()
        for object_name, position in positions.items():
            joint_id = self._object_joint_ids[object_name]
            qpos_address = self.model.jnt_qposadr[joint_id]
            angle = float(self._rng.uniform(-np.pi, np.pi))
            self.data.qpos[qpos_address : qpos_address + 7] = [
                position[0],
                position[1],
                position[2],
                np.cos(angle / 2),
                0,
                0,
                np.sin(angle / 2),
            ]

        self._apply_perturbations()
        mujoco.mj_forward(self.model, self.data)
        self._configure_drawer_stow()
        self._set_jaw_collision_enabled(None, False)
        return self.observe(include_camera=False)

    def _sample_object_positions(self) -> dict[str, tuple[float, float, float]]:
        positions: dict[str, tuple[float, float, float]] = {}
        for object_name in OBJECT_NAMES:
            if object_name in {"fork", "spoon"}:
                # The drawer body starts at y=-0.70 and slides along +y.
                # Keep each utensil on the native tray floor, within the side
                # walls, and add a small deterministic placement perturbation.
                drawer_qpos = self.data.qpos[self.model.jnt_qposadr[self._joint_ids["drawer_slide"]]]
                drawer_y = -0.70 + float(drawer_qpos)
                slot_x = -0.24 if object_name == "fork" else 0.24
                slot_y = drawer_y + (-0.06 if object_name == "fork" else 0.06)
                jitter = self._rng.uniform(
                    -self.perturbation.placement_jitter,
                    self.perturbation.placement_jitter,
                    size=2,
                ) * 0.12
                positions[object_name] = (
                    float(np.clip(slot_x + jitter[0], -0.55, 0.55)),
                    float(np.clip(slot_y + jitter[1], drawer_y - 0.20, drawer_y + 0.20)),
                    # The front tray sits immediately below the tabletop.
                    # This keeps the utensil inside the SO-101 grasp envelope
                    # while the drawer stow weld carries it until the
                    # retrieval contact gate begins.
                    0.72,
                )
                continue
            anchor = INITIAL_OBJECT_TARGETS[object_name][:2]
            jitter = self._rng.uniform(
                -self.perturbation.placement_jitter,
                self.perturbation.placement_jitter,
                size=2,
            ) * 0.35
            candidate = anchor + jitter
            candidate[0] = np.clip(candidate[0], -0.42, 0.42)
            candidate[1] = np.clip(candidate[1], -0.30, 0.08)
            positions[object_name] = (float(candidate[0]), float(candidate[1]), 0.86)
        if len(positions) != len(OBJECT_NAMES):
            raise RuntimeError("Could not place all dinner-table objects")
        return positions

    def _apply_perturbations(self) -> None:
        object_geom_names = [f"{name}_geom" for name in OBJECT_NAMES]
        for geom_name in object_geom_names:
            geom_id = self._geom_ids[geom_name]
            self.model.geom_friction[geom_id] = (
                self._base_geom_friction[geom_id] * self.perturbation.friction_scale
            )
            self.model.geom_size[geom_id] = (
                self._base_geom_size[geom_id] * self.perturbation.object_scale
            )

        for object_name in OBJECT_NAMES:
            body_id = self._body_ids[object_name]
            self.model.body_mass[body_id] = (
                self._base_body_mass[body_id] * self.perturbation.mass_scale
            )

        self.model.vis.headlight.ambient[:] = np.clip(
            self._base_ambient * self.perturbation.lighting_scale, 0.05, 1.0
        )
        background_colors = {
            "neutral": (0.12, 0.15, 0.18),
            "warm": (0.22, 0.16, 0.12),
            "cool": (0.10, 0.16, 0.24),
        }
        background = background_colors[self.perturbation.background]
        floor_material_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "floor_mat"
        )
        self.model.mat_rgba[floor_material_id] = (*background, 1.0)
        self.model.vis.rgba.haze[:] = (*background, 1.0)

    def step(self, action: Mapping[str, Any] | Sequence[float] | None = None) -> dict[str, Any]:
        """Advance physics from low-level or task-level actions.

        Supported mappings:

        - ``{"type": "joint_targets", "targets": {name: value}}``
        - ``{"type": "open_drawer"}`` / ``{"type": "close_drawer"}``
        - ``{"type": "move_to", "arm": "left", "target": [x, y, z]}``
        - ``{"type": "grasp", "arm": "left", "object": "plate"}``
        - ``{"type": "release", "arm": "left", "object": "plate"}``
        - ``{"type": "handoff", "object": "cup", "from": "right", "to": "left"}``
        - ``{"type": "noop"}``

        Task-level actions are verified against live MuJoCo state. A grasp is
        accepted only after the named fingertip meshes produce contact force
        against the object while the calibrated pinch frame is aligned. After
        that contact gate, an inactive native MuJoCo weld may be activated to
        stabilize the measured grasp through lift and long transport; the
        grasp is still rejected unless the subsequent lift passes. It is never
        an initial-grasp shortcut or a browser animation.
        """

        self._physics_advanced_in_action = False
        if action is None:
            self.data.ctrl[:] = 0
        elif isinstance(action, Mapping):
            self._apply_mapping_action(action)
        else:
            controls = np.asarray(action, dtype=np.float64)
            if controls.shape != (self.model.nu,):
                raise ValueError(f"raw action must have shape {(self.model.nu,)}")
            self.data.ctrl[:] = controls

        if not self._physics_advanced_in_action:
            self._advance_physics()
        return self.observe(include_camera=False)

    def control_spec(self) -> dict[str, Any]:
        """Return the ordered actuator contract used by neural policies."""

        names = tuple(self._names(mujoco.mjtObj.mjOBJ_ACTUATOR))
        return {
            "names": list(names),
            "ranges": {
                name: self.model.actuator_ctrlrange[self._actuator_ids[name]].astype(float).tolist()
                for name in names
            },
            "nu": int(self.model.nu),
        }

    def control_vector(self) -> np.ndarray:
        """Return the last commanded absolute target in contract order."""

        names = self.control_spec()["names"]
        return np.asarray(
            [self.data.ctrl[self._actuator_ids[name]] for name in names],
            dtype=np.float32,
        )

    def step_vla(self, action: Mapping[str, Any] | Sequence[float]) -> dict[str, Any]:
        """Execute a low-level policy action with physical task event gates.

        A VLA emits continuous actuator targets, not symbolic ``grasp`` or
        ``release`` commands. This method runs those targets through MuJoCo,
        then asks the existing measured-contact task gates whether a semantic
        event should occur. It never marks progress from a model token alone.
        """

        spec = self.control_spec()
        names = spec["names"]
        ranges = spec["ranges"]
        if isinstance(action, Mapping):
            raw_targets = action.get("targets")
            if not isinstance(raw_targets, Mapping):
                raise ValueError("VLA action mapping requires a targets mapping")
            values = [raw_targets.get(name) for name in names]
        else:
            values = np.asarray(action, dtype=np.float64).reshape(-1).tolist()
        if len(values) != len(names):
            raise ValueError(f"VLA action must contain {len(names)} actuator targets")
        targets = {
            name: float(np.clip(float(value), ranges[name][0], ranges[name][1]))
            for name, value in zip(names, values)
        }
        observation = self.step({"type": "joint_targets", "targets": targets})
        self._advance_vla_task_events()
        return self.observe(include_camera=False)

    def _advance_vla_task_events(self) -> None:
        """Translate physical low-level state into guarded task events."""

        step_id = self._current_step_id()
        if step_id is None:
            return
        if step_id == "open_drawer":
            drawer_id = self._joint_ids["drawer_slide"]
            drawer_position = float(self.data.qpos[self.model.jnt_qposadr[drawer_id]])
            if drawer_position <= 0.06:
                self._task_index += 1
                self._last_action_result = {"ok": True, "action": "vla_open_drawer", "message": "drawer opened"}
            return

        event_by_step = {
            "retrieve_fork": ("left", "fork"),
            "retrieve_spoon": ("right", "spoon"),
            "place_plate": ("left", "plate"),
            "place_cup": ("left", "cup"),
        }
        if step_id in event_by_step:
            arm, object_name = event_by_step[step_id]
            object_state = self._held_by[object_name]
            gripper = self._gripper_position(arm)
            closed = gripper <= (GRIPPER_OPEN + GRIPPER_CLOSED) / 2
            if object_state is None and closed and self._near_grasp_target(arm, object_name, 0.035):
                self._apply_grasp({"type": "grasp", "arm": arm, "object": object_name})
            elif object_state == arm and not closed:
                self._apply_release({"type": "release", "arm": arm, "object": object_name})
            return

        if step_id != "handoff_cup":
            return
        held_by = self._held_by["cup"]
        if held_by is None:
            if self._gripper_is_closed("right") and self._near_grasp_target("right", "cup", 0.035):
                self._apply_grasp({"type": "grasp", "arm": "right", "object": "cup"})
            return
        if held_by == "right" and self._gripper_is_closed("left") and self._near_grasp_target("left", "cup", 0.035):
            self._apply_handoff({"type": "handoff", "object": "cup", "from": "right", "to": "left"})

    def _gripper_position(self, arm: str) -> float:
        joint_id = self._joint_ids[f"{arm}_gripper"]
        return float(self.data.qpos[self.model.jnt_qposadr[joint_id]])

    def _gripper_is_closed(self, arm: str) -> bool:
        return self._gripper_position(arm) <= (GRIPPER_OPEN + GRIPPER_CLOSED) / 2

    def _near_grasp_target(self, arm: str, object_name: str, tolerance: float) -> bool:
        target, _ = self._object_grasp_pose(object_name, arm)
        pinch = self.data.site_xpos[self._pinch_site(arm)]
        return bool(np.linalg.norm(pinch - target) <= tolerance)

    def _apply_mapping_action(self, action: Mapping[str, Any]) -> None:
        action_type = action.get("type")
        if action_type == "noop":
            self.data.ctrl[:] = 0
            return
        if action_type in {"open_drawer", "close_drawer"}:
            actuator_id = self._actuator_ids["drawer_position"]
            target = 0.0 if action_type == "open_drawer" else 0.30
            self.data.ctrl[actuator_id] = target
            # The drawer servo is deliberately low-gain, so hold its target
            # for a physically meaningful interval instead of treating a
            # single integrator tick as "open".
            self._advance_physics(steps=max(self.frame_skip, 360))
            self._physics_advanced_in_action = True
            opened = self.observe()["drawer"] <= 0.06
            ok = action_type != "open_drawer" or opened
            if ok and action_type == "open_drawer" and self._current_step_id() == "open_drawer":
                self._task_index += 1
            self._last_action_result = {
                "ok": ok,
                "action": action_type,
                "message": "drawer opened" if opened else "drawer target not reached",
            }
            return
        if action_type == "move_to":
            self._apply_move_to(action)
            return
        if action_type == "grasp":
            self._apply_grasp(action)
            return
        if action_type == "release":
            self._apply_release(action)
            return
        if action_type == "handoff":
            self._apply_handoff(action)
            return
        if action_type != "joint_targets":
            raise ValueError(f"unsupported MuJoCo action type: {action_type!r}")
        targets = action.get("targets")
        if not isinstance(targets, Mapping):
            raise ValueError("joint_targets action requires a targets mapping")
        for name, value in targets.items():
            if name not in self._actuator_ids:
                raise ValueError(f"unknown actuator: {name}")
            self.data.ctrl[self._actuator_ids[name]] = float(value)
        self._last_action_result = {"ok": True, "action": "joint_targets"}

    def _current_step_id(self) -> str | None:
        if self._task_index >= len(self._task_plan):
            return None
        return self._task_plan[self._task_index].id

    def _advance_physics(self, steps: int | None = None) -> None:
        for _ in range(steps if steps is not None else self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def _require_arm(self, value: Any) -> str:
        if value not in ARM_NAMES:
            raise ValueError(f"arm must be one of {ARM_NAMES}, got {value!r}")
        return str(value)

    def _require_object(self, value: Any) -> str:
        if value not in OBJECT_NAMES:
            raise ValueError(f"object must be one of {OBJECT_NAMES}, got {value!r}")
        return str(value)

    def _arm_site(self, arm: str) -> int:
        return self._site_ids[f"{arm}_grasp_site"]

    def _pinch_site(self, arm: str) -> int:
        return self._site_ids[f"{arm}_pinch_site"]

    def _object_site(self, object_name: str) -> int:
        return self._site_ids[f"{object_name}_grasp_site"]

    def _set_jaw_collision_enabled(self, arm: str | None, enabled: bool, object_name: str | None = None) -> None:
        selected_arms = ARM_NAMES if arm is None else (arm,)
        for selected_arm in selected_arms:
            for geom_id in (
                self._jaw_geom_ids[selected_arm]
                + self._plate_jaw_geom_ids[selected_arm]
                + self._cup_jaw_geom_ids[selected_arm]
            ):
                self.model.geom_contype[geom_id] = 0
            active_jaws = (
                self._plate_jaw_geom_ids[selected_arm]
                if object_name == "plate"
                else self._cup_jaw_geom_ids[selected_arm]
                if object_name == "cup"
                else self._jaw_geom_ids[selected_arm]
            )
            if enabled:
                for geom_id in active_jaws:
                    self.model.geom_contype[geom_id] = 4
        active_objects = {
            selected_object
            for selected_object, holder in self._held_by.items()
            if holder is not None
        }
        if enabled and object_name is not None:
            active_objects.add(object_name)
        for selected_object in OBJECT_NAMES:
            geom_id = self._geom_ids[f"{selected_object}_geom"]
            active = selected_object in active_objects
            stowed = selected_object in self._drawer_stow_equality_ids and self._drawer_stow_is_active(selected_object)
            # Held/stowed objects contact only the calibrated jaw category;
            # otherwise a utensil in the drawer could collide with the table
            # top or tray before retrieval. Released objects restore normal
            # table contact so gravity and placement remain physical.
            self.model.geom_contype[geom_id] = 0 if active or stowed else 2
            self.model.geom_conaffinity[geom_id] = 4 if active else 0 if stowed else 2

    def _retention_is_active(self, arm: str, object_name: str) -> bool:
        return bool(self.data.eq_active[self._retention_equality_ids[(arm, object_name)]])

    def _drawer_stow_is_active(self, object_name: str) -> bool:
        return bool(self.data.eq_active[self._drawer_stow_equality_ids[object_name]])

    def _set_drawer_stow(self, object_name: str, enabled: bool, *, forward: bool = True) -> None:
        """Carry a stowed utensil with the drawer until retrieval begins."""

        equality_id = self._drawer_stow_equality_ids[object_name]
        if enabled:
            drawer_body = self._body_ids["drawer"]
            object_body = self._body_ids[object_name]
            drawer_rotation = self.data.xmat[drawer_body].reshape(3, 3)
            object_rotation = self.data.xmat[object_body].reshape(3, 3)
            # MuJoCo stores the body2-local anchor first and the body1-local
            # anchor second. Attach the utensil center to its measured slot in
            # the drawer frame, preserving the slot offset during motion.
            object_anchor = np.zeros(3, dtype=np.float64)
            drawer_anchor = drawer_rotation.T @ (self.data.xpos[object_body] - self.data.xpos[drawer_body])
            relative_rotation = drawer_rotation.T @ object_rotation
            self.model.eq_data[equality_id, 0:3] = object_anchor
            self.model.eq_data[equality_id, 3:6] = drawer_anchor
            self.model.eq_data[equality_id, 6:10] = _matrix_to_quaternion(relative_rotation)
            self.data.eq_active[equality_id] = True
        else:
            self.data.eq_active[equality_id] = False
        if forward:
            mujoco.mj_forward(self.model, self.data)

    def _configure_drawer_stow(self) -> None:
        for object_name in ("fork", "spoon"):
            self._set_drawer_stow(object_name, True, forward=False)
        mujoco.mj_forward(self.model, self.data)

    def _set_retention(
        self,
        arm: str,
        object_name: str,
        enabled: bool,
        *,
        forward: bool = True,
    ) -> None:
        """Toggle the post-contact MuJoCo retention constraint.

        The reference pose is measured from the live body transforms after a
        two-jaw contact and pre-lift drift gate. This means the constraint
        stabilizes a measured grasp before and during lift; it never
        manufactures the contact that made the grasp valid. ``forward=False``
        lets hand-off switch constraints atomically before the next solver
        update.
        """

        equality_id = self._retention_equality_ids[(arm, object_name)]
        if enabled:
            gripper_body = self._body_ids[f"{arm}_gripper"]
            object_body = self._body_ids[object_name]
            gripper_rotation = self.data.xmat[gripper_body].reshape(3, 3)
            object_rotation = self.data.xmat[object_body].reshape(3, 3)
            # MuJoCo stores the object-local anchor first, then the
            # body1-local anchor, followed by the body2-relative quaternion.
            # Anchor the gripper side at the calibrated pinch site and solve
            # the matching object-local point from the live transforms.
            gripper_anchor = self.model.site_pos[self._pinch_site(arm)].copy()
            world_anchor = self.data.xpos[gripper_body] + gripper_rotation @ gripper_anchor
            object_anchor = object_rotation.T @ (world_anchor - self.data.xpos[object_body])
            relative_rotation = gripper_rotation.T @ object_rotation
            self.model.eq_data[equality_id, 0:3] = object_anchor
            self.model.eq_data[equality_id, 3:6] = gripper_anchor
            self.model.eq_data[equality_id, 6:10] = _matrix_to_quaternion(relative_rotation)
            self.data.eq_active[equality_id] = True
        else:
            self.data.eq_active[equality_id] = False
        if forward:
            mujoco.mj_forward(self.model, self.data)

    def _object_grasp_pose(self, object_name: str, arm: str = "left") -> tuple[np.ndarray, np.ndarray]:
        """Estimate a calibrated pinch pose from live object geometry.

        This is deterministic geometric perception for the MuJoCo benchmark:
        the object pose comes from the simulator state and the height is chosen
        from the known collision primitive. A camera/VLA adapter can replace
        this estimator without changing the controller contract.
        """

        body_id = self._body_ids[object_name]
        # Use the settled live body center. The reset pose is intentionally
        # above the tabletop so gravity can settle each primitive before the
        # visual/geometric estimator publishes a grasp target.
        position = self.data.xpos[body_id].copy() + GRASP_PINCH_OFFSETS[arm]
        if object_name == "plate":
            # A plate is not graspable at its center: both fingers would land
            # inside the disk. Use the near rim, with a small vertical lift,
            # so the calibrated pads can oppose the plate edge from the front
            # of the table while remaining reachable by the left arm.
            position += GRASP_TARGET_OFFSETS[object_name]
        elif object_name == "cup":
            position += GRASP_TARGET_OFFSETS[object_name]
            # The right base has a hard near-side reach boundary. For a cup
            # reset toward the back edge, use the closest reachable approach
            # plane rather than repeatedly commanding an unreachable target;
            # physical jaw contact and lift still gate acceptance.
            if arm == "right":
                position[1] = min(position[1], RIGHT_CUP_MIN_APPROACH_Y)
        object_yaw = 0.0
        if object_name in {"fork", "spoon"}:
            quaternion = self.data.xquat[body_id]
            object_yaw = float(2.0 * np.arctan2(quaternion[3], quaternion[0]))
        base = self._pinch_base_quaternion[arm]
        orientation = _quaternion_multiply(_z_rotation_quaternion(object_yaw), base)
        return position, orientation

    def _contact_metrics(self, arm: str, object_name: str) -> dict[str, Any]:
        object_geom = self._geom_ids[f"{object_name}_geom"]
        jaw_ids = set(
            self._plate_jaw_geom_ids[arm]
            if object_name == "plate"
            else self._cup_jaw_geom_ids[arm]
            if object_name == "cup"
            else self._jaw_geom_ids[arm]
        )
        contact_count = 0
        contacting_jaws: set[int] = set()
        normal_force = 0.0
        penetration = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_a = int(contact.geom1)
            geom_b = int(contact.geom2)
            if object_geom not in {geom_a, geom_b}:
                continue
            jaw_id = geom_b if geom_a == object_geom else geom_a
            if jaw_id not in jaw_ids:
                continue
            contact_count += 1
            contacting_jaws.add(jaw_id)
            penetration += max(0.0, -float(contact.dist))
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            normal_force += abs(float(force[0]))
        pair_distances = {
            str(jaw_id): float(mujoco.mj_geomDistance(self.model, self.data, object_geom, jaw_id, 0.25, np.zeros(6)))
            for jaw_id in jaw_ids
        }
        return {
            "contact_count": contact_count,
            "jaw_count": len(contacting_jaws),
            "normal_force": normal_force,
            "penetration": penetration,
            "jaw_distances": pair_distances,
            "verified": bool(contact_count > 0 and normal_force > 0.01),
        }

    def _arm_joint_qpos_addresses(self, arm: str) -> list[int]:
        return [
            self.model.jnt_qposadr[self._joint_ids[f"{arm}_{joint}"]]
            for joint in ARM_JOINTS
        ]

    def _arm_joint_dof_addresses(self, arm: str) -> list[int]:
        return [
            self.model.jnt_dofadr[self._joint_ids[f"{arm}_{joint}"]]
            for joint in ARM_JOINTS
        ]

    def _solve_ik(
        self,
        arm: str,
        target: np.ndarray,
        target_quaternion: np.ndarray | None = None,
        site_id: int | None = None,
        seed_qpos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Solve damped-least-squares position or weighted 6D pose IK.

        SO-101 has five arm joints, so an arbitrary six degree-of-freedom pose
        is overconstrained. We solve the full weighted pose and report both
        residuals; the safety gate rejects a grasp when the orientation is not
        aligned instead of silently accepting a position-only approximation.
        The live state is restored after probing candidate qpos.
        """

        site_id = self._arm_site(arm) if site_id is None else site_id
        addresses = self._arm_joint_qpos_addresses(arm)
        dofs = self._arm_joint_dof_addresses(arm)
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        candidate = saved_qpos.copy() if seed_qpos is None else np.asarray(seed_qpos, dtype=np.float64).copy()
        target = np.asarray(target, dtype=np.float64)
        target_quaternion = None if target_quaternion is None else _normalize_quaternion(target_quaternion)
        for iteration in range(56):
            self.data.qpos[:] = candidate
            mujoco.mj_forward(self.model, self.data)
            position_error = target - self.data.site_xpos[site_id]
            current_quaternion = _matrix_to_quaternion(self.data.site_xmat[site_id])
            orientation_error = (
                np.zeros(3, dtype=np.float64)
                if target_quaternion is None
                else _rotation_vector_error(target_quaternion, current_quaternion)
            )
            position_only_phase = target_quaternion is not None and iteration < 28
            if float(np.linalg.norm(position_error)) < 0.0015 and (
                target_quaternion is None
                or (not position_only_phase and float(np.linalg.norm(orientation_error)) < 0.025)
            ):
                break
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
            if target_quaternion is None or position_only_phase:
                error = position_error
                jacobian = jacp[:, dofs]
                damping = 0.018
            else:
                error = np.concatenate([position_error, ORIENTATION_WEIGHT * orientation_error])
                jacobian = np.vstack([jacp[:, dofs], ORIENTATION_WEIGHT * jacr[:, dofs]])
                damping = 0.035
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(jacobian.shape[0]), error
            )
            for address, joint_delta, joint_name in zip(addresses, delta, ARM_JOINTS):
                joint_id = self._joint_ids[f"{arm}_{joint_name}"]
                candidate[address] = np.clip(
                    candidate[address] + 0.62 * joint_delta,
                    self.model.jnt_range[joint_id, 0],
                    self.model.jnt_range[joint_id, 1],
                )
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[:] = candidate
        mujoco.mj_forward(self.model, self.data)
        position_residual = float(np.linalg.norm(target - self.data.site_xpos[site_id]))
        orientation_residual = 0.0
        if target_quaternion is not None:
            orientation_residual = float(
                np.linalg.norm(
                    _rotation_vector_error(
                        target_quaternion,
                        _matrix_to_quaternion(self.data.site_xmat[site_id]),
                    )
                )
            )
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)
        return candidate[addresses].copy(), position_residual, orientation_residual

    def _solve_ik_best(
        self,
        arm: str,
        target: np.ndarray,
        target_quaternion: np.ndarray | None = None,
        site_id: int | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Use multiple collision-free IK seeds when a local branch stalls."""

        best = self._solve_ik(arm, target, target_quaternion, site_id)
        if best[1] < 0.045:
            return best
        saved_qpos = self.data.qpos.copy()
        addresses = self._arm_joint_qpos_addresses(arm)
        seeds: list[np.ndarray] = []
        neutral = saved_qpos.copy()
        neutral[addresses] = 0.0
        seeds.append(neutral)
        midpoint = saved_qpos.copy()
        for address, joint_name in zip(addresses, ARM_JOINTS):
            joint_id = self._joint_ids[f"{arm}_{joint_name}"]
            midpoint[address] = np.mean(self.model.jnt_range[joint_id])
        seeds.append(midpoint)
        for seed in seeds:
            candidate = self._solve_ik(arm, target, target_quaternion, site_id, seed)
            if candidate[1] + 0.02 * candidate[2] < best[1] + 0.02 * best[2]:
                best = candidate
        return best

    def _set_arm_control(self, arm: str, targets: Sequence[float], gripper: float | None = None) -> None:
        for joint, value in zip(ARM_JOINTS, targets):
            self.data.ctrl[self._actuator_ids[f"{arm}_{joint}_motor"]] = float(value)
        if gripper is not None:
            self.data.ctrl[self._actuator_ids[f"{arm}_gripper_motor"]] = float(gripper)

    def _candidate_site_quaternion(self, candidate: np.ndarray, site_id: int) -> np.ndarray:
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        self.data.qpos[:] = candidate
        mujoco.mj_forward(self.model, self.data)
        quaternion = _matrix_to_quaternion(self.data.site_xmat[site_id])
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.model, self.data)
        return quaternion

    def _apply_move_to(self, action: Mapping[str, Any]) -> None:
        arm = self._require_arm(action.get("arm"))
        target = np.asarray(action.get("target"), dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("move_to target must be a finite [x, y, z] position")
        orientation = action.get("orientation", "auto")
        target_quaternion = None
        auto_orientation = orientation is None or orientation == "auto"
        if not auto_orientation:
            target_quaternion = np.asarray(orientation, dtype=np.float64)
            if target_quaternion.shape != (4,) or not np.isfinite(target_quaternion).all():
                raise ValueError("move_to orientation must be a finite [w, x, y, z] quaternion")
        frame = action.get("frame", "grasp")
        site_id = self._pinch_site(arm) if frame == "pinch" else self._arm_site(arm)
        gripper = action.get("gripper")
        if isinstance(gripper, str):
            gripper = GRIPPER_CLOSED if gripper == "closed" else GRIPPER_OPEN
        if frame not in {"grasp", "pinch"}:
            raise ValueError("move_to frame must be grasp or pinch")
        held_object = next(
            (object_name for object_name, holder in self._held_by.items() if holder == arm),
            None,
        )
        holding_object = held_object is not None and frame == "pinch"
        if holding_object and held_object == "cup" and auto_orientation and not self._retention_is_active(arm, held_object):
            # A held object defines the current tool orientation. Re-solving
            # an arbitrary "auto" orientation at every pre-retention waypoint
            # can rotate the jaws out of their measured contact envelope.
            # Once retention is validated, the native weld preserves the
            # measured object pose while the arm uses position-only transport.
            target_quaternion = _matrix_to_quaternion(self.data.site_xmat[site_id])
            auto_orientation = False

        approach = bool(action.get("approach", False))
        trajectory: list[str] = []
        if approach:
            approach_vector = np.asarray(action.get("approach_vector", [0.0, 0.0, 1.0]), dtype=np.float64)
            if approach_vector.shape != (3,) or not np.isfinite(approach_vector).all():
                raise ValueError("approach_vector must be a finite 3D vector")
            norm = float(np.linalg.norm(approach_vector))
            if norm < 1e-9:
                raise ValueError("approach_vector must not be zero")
            approach_vector /= norm
            pregrasp_target = target + approach_vector * float(action.get("approach_distance", GRASP_APPROACH_DISTANCE))
            pregrasp_targets, _, _ = self._solve_ik_best(
                arm, pregrasp_target, None if auto_orientation else target_quaternion, site_id
            )
            pregrasp_steps = max(32, int(action.get("pregrasp_steps", PREGRASP_STEPS)))
            self._set_arm_control(arm, pregrasp_targets, None if gripper is None else float(gripper))
            self._advance_physics(steps=pregrasp_steps)
            self._physics_advanced_in_action = True
            trajectory.append("pregrasp")

        targets, _, orientation_residual = self._solve_ik_best(
            arm, target, None if auto_orientation else target_quaternion, site_id
        )
        if auto_orientation:
            candidate = self.data.qpos.copy()
            addresses = self._arm_joint_qpos_addresses(arm)
            candidate[addresses] = targets
            target_quaternion = self._candidate_site_quaternion(candidate, site_id)
            orientation_residual = 0.0
        settle_steps = max(24, int(action.get("settle_steps", GRASP_SETTLE_STEPS)))
        if not any(holder == arm for holder in self._held_by.values()):
            self._set_jaw_collision_enabled(arm, False)
        start_position = self.data.site_xpos[site_id].copy()
        segment_count = (
            max(1, int(np.ceil(np.linalg.norm(target - start_position) / 0.0015)))
            if holding_object
            else 1
        )
        contact_lost = False
        transport_contact_trace: list[dict[str, Any]] = []
        retention_active = bool(holding_object and self._retention_is_active(arm, held_object))
        for segment_index in range(segment_count):
            if segment_count > 1:
                progress = float(segment_index + 1) / segment_count
                segment_target = start_position + progress * (target - start_position)
                targets, _, orientation_residual = self._solve_ik_best(
                    arm, segment_target, None if auto_orientation else target_quaternion, site_id
                )
                trajectory.append(f"hold_segment_{segment_index + 1}")
            else:
                segment_target = target
                trajectory.append("settle")
            self._set_arm_control(arm, targets, None if gripper is None else float(gripper))
            segment_steps = 3 if holding_object else settle_steps
            self._advance_physics(steps=max(24, segment_steps))
            self._physics_advanced_in_action = True
            if holding_object:
                contact = self._contact_metrics(arm, held_object)
                transport_contact_trace.append(
                    {
                        "segment": segment_index + 1,
                        "distance_to_target": float(np.linalg.norm(target - self.data.site_xpos[site_id])),
                        "jaw_count": contact["jaw_count"],
                        "normal_force": contact["normal_force"],
                        "retention_active": retention_active,
                    }
                )
                # During a short transport the pads should remain visibly in
                # contact. If the solver reports a transient pad separation
                # after the contact gate, the measured post-contact weld
                # is the auditable retention mechanism that keeps the object
                # attached. Without that constraint, contact loss is a real
                # failure rather than something to silently ignore.
                if not retention_active and (not contact["verified"] or contact["jaw_count"] < 2):
                    contact_lost = True
                    break
        distance = float(np.linalg.norm(target - self.data.site_xpos[site_id]))
        actual_orientation_residual = 0.0
        if target_quaternion is not None:
            actual_orientation_residual = float(
                np.linalg.norm(
                    _rotation_vector_error(
                        target_quaternion,
                        _matrix_to_quaternion(self.data.site_xmat[site_id]),
                    )
                )
            )
        self._last_action_result = {
            "ok": not contact_lost and distance < GRASP_POSITION_TOLERANCE and (
                target_quaternion is None or actual_orientation_residual < GRASP_ORIENTATION_TOLERANCE
            ),
            "action": "move_to",
            "arm": arm,
            "frame": frame,
            "target": target.tolist(),
            "distance": distance,
            "orientation_error": actual_orientation_residual,
            "ik_residual": distance,
            "trajectory": trajectory,
        }
        if transport_contact_trace:
            self._last_action_result["transport_contact"] = transport_contact_trace
        if retention_active:
            self._last_action_result["retention"] = {
                "active": True,
                "constraint": f"{arm}_{held_object}_retention",
                "validated_after": "two_jaw_contact_and_lift",
            }
        if contact_lost:
            self._last_action_result["message"] = "held-object contact lost during segmented transport"
        elif retention_active and holding_object:
            self._last_action_result["message"] = "transport completed under validated MuJoCo retention"
        self._last_pose_goal[arm] = {
            "frame": frame,
            "target": target.tolist(),
            "orientation": None if target_quaternion is None else target_quaternion.tolist(),
            "distance": distance,
            "orientation_error": actual_orientation_residual,
        }
        # Keep a recoverable joint-space waypoint. A physical contact attempt
        # can move the arm or object slightly; retries must start from the last
        # verified pose instead of solving from an arbitrary lifted posture.
        if self._last_action_result["ok"]:
            self._last_safe_joint_targets[arm] = np.asarray(targets, dtype=np.float64).copy()

    def _lift_held_object(self, arm: str, object_name: str) -> tuple[bool, dict[str, Any]]:
        before = self.data.xpos[self._body_ids[object_name]].copy()
        start = self.data.site_xpos[self._pinch_site(arm)].copy()
        lift_target = start.copy()
        lift_targets = self.data.qpos[self._arm_joint_qpos_addresses(arm)].copy()
        lift_orientation = _matrix_to_quaternion(self.data.site_xmat[self._pinch_site(arm)])
        lift_metrics: dict[str, Any] = {"verified": False, "jaw_count": 0, "height": 0.0}
        lifted_height = 0.0
        self._set_jaw_collision_enabled(arm, True, object_name)
        # A single 14 cm servo jump makes the pads slide across a light flat
        # utensil. Use short, observable lift increments and stop as soon as
        # two-jaw contact has produced a meaningful clearance. This keeps the
        # grasp inside the measured contact envelope instead of assuming that
        # a final pose implies a stable hold.
        for lift_height in np.arange(0.015, 0.076, 0.015):
            lift_target = start + np.array([0.0, 0.0, float(lift_height)])
            lift_targets, _, _ = self._solve_ik_best(arm, lift_target, None, self._pinch_site(arm))
            self._set_arm_control(arm, lift_targets, GRIPPER_CLOSED)
            self._advance_physics(steps=30)
            self._physics_advanced_in_action = True
            lift_metrics = self._contact_metrics(arm, object_name)
            lifted_height = float(self.data.xpos[self._body_ids[object_name]][2] - before[2])
            if lift_metrics["verified"] and lift_metrics["jaw_count"] >= 2 and lifted_height >= 0.045:
                break
            if not lift_metrics["verified"] and lift_height >= 0.045:
                break
        lift_candidate = self.data.qpos.copy()
        lift_candidate[self._arm_joint_qpos_addresses(arm)] = lift_targets
        lift_orientation = self._candidate_site_quaternion(lift_candidate, self._pinch_site(arm))
        self._last_pose_goal[arm] = {
            "frame": "pinch",
            "target": lift_target.tolist(),
            "orientation": lift_orientation.tolist(),
            "distance": float(np.linalg.norm(self.data.site_xpos[self._pinch_site(arm)] - lift_target)),
            "orientation_error": 0.0,
        }
        return bool(lift_metrics["verified"] and lift_metrics["jaw_count"] >= 2 and lifted_height >= 0.045), {
            **lift_metrics,
            "height": lifted_height,
        }

    def _apply_grasp(self, action: Mapping[str, Any]) -> None:
        arm = self._require_arm(action.get("arm"))
        object_name = self._require_object(action.get("object"))
        if self._held_by[object_name] is not None:
            self._last_action_result = {"ok": False, "action": "grasp", "message": "object already held"}
            return
        if self._current_step_id() not in {"retrieve_fork", "retrieve_spoon", "place_plate", "handoff_cup"}:
            self._last_action_result = {"ok": False, "action": "grasp", "message": "grasp is not valid for the current task step"}
            return

        # Search a very small calibrated neighborhood. The arm first attempts
        # the estimator's pose, then makes sub-centimeter corrections based on
        # live jaw/object contact instead of accepting a lucky near miss.
        micro_offsets = (
            np.zeros(3),
            np.array([0.0, -0.010, 0.0]),
            np.array([0.0, 0.010, 0.0]),
            np.array([-0.010, 0.0, 0.0]),
            np.array([0.010, 0.0, 0.0]),
        )
        attempts: list[dict[str, Any]] = []
        safe_targets = self._last_safe_joint_targets.get(arm)
        ok = False
        distance = float("inf")
        orientation_error = float("inf")
        object_drift = float("inf")
        metrics: dict[str, Any] = {"verified": False}
        for attempt_index, micro_offset in enumerate(micro_offsets):
            target, object_orientation = self._object_grasp_pose(object_name, arm)
            target += micro_offset
            if attempt_index:
                self._set_jaw_collision_enabled(arm, False)
                pregrasp_target = target + np.array([0.0, 0.0, GRASP_APPROACH_DISTANCE])
                pregrasp_targets, _, _ = self._solve_ik_best(arm, pregrasp_target, None, self._pinch_site(arm))
                self._set_arm_control(arm, pregrasp_targets, GRIPPER_OPEN)
                self._advance_physics(steps=120)
                targets, _, _ = self._solve_ik_best(arm, target, None, self._pinch_site(arm))
                candidate = self.data.qpos.copy()
                candidate[self._arm_joint_qpos_addresses(arm)] = targets
                target_orientation = self._candidate_site_quaternion(candidate, self._pinch_site(arm))
                self._set_arm_control(arm, targets, GRIPPER_OPEN)
                self._advance_physics(steps=180)
                self._last_pose_goal[arm] = {
                    "frame": "pinch",
                    "target": target.tolist(),
                    "orientation": target_orientation.tolist(),
                }
            pose_goal = self._last_pose_goal.get(arm, {})
            target_quaternion = np.asarray(pose_goal.get("orientation") or object_orientation, dtype=np.float64)
            distance = float(np.linalg.norm(self.data.site_xpos[self._pinch_site(arm)] - target))
            orientation_error = float(
                np.linalg.norm(
                    _rotation_vector_error(
                        target_quaternion,
                        _matrix_to_quaternion(self.data.site_xmat[self._pinch_site(arm)]),
                    )
                )
            )
            if distance > 0.035 or orientation_error > GRASP_ORIENTATION_TOLERANCE:
                attempts.append({"index": attempt_index, "distance": distance, "orientation_error": orientation_error, "accepted": False, "reason": "pose_gate"})
                continue
            before = self.data.xpos[self._body_ids[object_name]].copy()
            # Close the calibrated jaws in free space before exposing the
            # object contact envelope. This removes the open-jaw sweep that
            # can shove a thin utensil or tall cup away from the estimator's
            # target, while still requiring measured contact after settling.
            self._set_jaw_collision_enabled(arm, False)
            self.data.ctrl[self._actuator_ids[f"{arm}_gripper_motor"]] = GRIPPER_CLOSED
            self._advance_physics(steps=GRASP_CLOSE_STEPS)
            self._set_jaw_collision_enabled(arm, True, object_name)
            # Give contacts two solver ticks to register, then lift. A long
            # settle window lets a light object sink or get pushed along the
            # tabletop before the lift gate has a chance to validate it.
            self._advance_physics(steps=2)
            self._physics_advanced_in_action = True
            metrics = self._contact_metrics(arm, object_name)
            after = self.data.xpos[self._body_ids[object_name]].copy()
            object_drift = float(np.linalg.norm(after - before))
            contact_ok = bool(metrics["verified"] and metrics["jaw_count"] >= 2 and object_drift < 0.045)
            lift_metrics: dict[str, Any] | None = None
            if contact_ok:
                # Keep the utensil fixed in its drawer slot while the jaws
                # close and the contact gate is measured. Only after that
                # gate passes do we switch from drawer stow to gripper
                # retention, so gravity cannot turn retrieval into a drop.
                if object_name in self._drawer_stow_equality_ids and self._drawer_stow_is_active(object_name):
                    self._set_drawer_stow(object_name, False, forward=False)
                self._held_by[object_name] = arm
                # Activate only after two-jaw contact and the pre-lift drift
                # gate. The native constraint stabilizes that measured
                # contact while the next phase verifies a real lift.
                self._set_retention(arm, object_name, True)
                ok, lift_metrics = self._lift_held_object(arm, object_name)
                if not ok:
                    self._set_retention(arm, object_name, False)
                    self._held_by[object_name] = None
            attempts.append({
                "index": attempt_index,
                "distance": distance,
                "orientation_error": orientation_error,
                "contact": metrics,
                "object_drift": object_drift,
                "lift": lift_metrics,
                "accepted": ok,
            })
            if ok:
                metrics = {**metrics, "lift": lift_metrics}
                metrics["retention"] = {
                    "active": True,
                    "constraint": f"{arm}_{object_name}_retention",
                    "validated_after": "two_jaw_contact_and_lift",
                }
                break
            self.data.ctrl[self._actuator_ids[f"{arm}_gripper_motor"]] = GRIPPER_OPEN
            self._advance_physics(steps=24)
            self._set_jaw_collision_enabled(arm, False)
            # A failed lift leaves the arm at its lift waypoint. Recover to
            # the last verified pre-grasp joint posture before trying another
            # micro-offset, otherwise the next IK solve starts from a bad
            # branch and can never return to the object.
            if lift_metrics is not None and safe_targets is not None:
                self._set_arm_control(arm, safe_targets, GRIPPER_OPEN)
                self._advance_physics(steps=180)
                self._physics_advanced_in_action = True
        if not ok:
            self._held_by[object_name] = None
            self.data.ctrl[self._actuator_ids[f"{arm}_gripper_motor"]] = GRIPPER_OPEN
            self._advance_physics(steps=24)
            self._set_jaw_collision_enabled(arm, False)
        self._last_grasp_metrics = {
            "arm": arm,
            "object": object_name,
            "distance": distance,
            "orientation_error": orientation_error,
            "object_drift": object_drift,
            "attempts": attempts,
            **metrics,
        }
        self._last_action_result = {
            "ok": ok,
            "action": "grasp",
            "arm": arm,
            "object": object_name,
            "distance": distance,
            "orientation_error": orientation_error,
            "object_drift": object_drift,
            "contact": metrics,
            "attempts": attempts,
            "message": "physical jaw grasp and lift verified" if ok else "physical contact verification failed",
        }

    def _destination_reached(self, object_name: str, destination: str | None) -> bool:
        if destination == "table_surface":
            position = self.data.xpos[self._body_ids[object_name]]
            # Match the physical tabletop footprint (the colored place area
            # is only a visual cue, not the boundary of a valid utensil drop).
            return bool(-1.45 <= position[0] <= 1.45 and -0.80 <= position[1] <= 0.80 and position[2] <= 1.02)
        if destination == "place_area":
            position = self.data.xpos[self._body_ids[object_name]]
            return bool(np.linalg.norm(position[:2] - OBJECT_TARGETS[object_name][:2]) < 0.18 and position[2] <= 1.02)
        return False

    def _apply_release(self, action: Mapping[str, Any]) -> None:
        arm = self._require_arm(action.get("arm"))
        object_name = self._require_object(action.get("object"))
        if self._held_by[object_name] != arm:
            self._last_action_result = {"ok": False, "action": "release", "message": "object is not held by this arm"}
            return
        self.data.ctrl[self._actuator_ids[f"{arm}_gripper_motor"]] = GRIPPER_OPEN
        self._set_retention(arm, object_name, False)
        self._advance_physics(steps=max(self.frame_skip, 40))
        self._held_by[object_name] = None
        self._set_jaw_collision_enabled(arm, False)
        self._physics_advanced_in_action = True
        step = self._task_plan[self._task_index] if self._task_index < len(self._task_plan) else None
        ok = bool(step and step.object_name == object_name and self._destination_reached(object_name, step.destination))
        if ok:
            self._placed.add(object_name)
            self._task_index += 1
        self._last_action_result = {
            "ok": ok,
            "action": "release",
            "arm": arm,
            "object": object_name,
            "destination": step.destination if step else None,
            "message": "placement verified" if ok else "placement verification failed",
        }

    def _apply_handoff(self, action: Mapping[str, Any]) -> None:
        object_name = self._require_object(action.get("object"))
        from_arm = self._require_arm(action.get("from", "right"))
        to_arm = self._require_arm(action.get("to", "left"))
        if self._current_step_id() != "handoff_cup" or object_name != "cup":
            self._last_action_result = {"ok": False, "action": "handoff", "message": "handoff is not valid for the current task step"}
            return
        if self._held_by[object_name] != from_arm:
            self._last_action_result = {"ok": False, "action": "handoff", "message": "source arm does not hold the object"}
            return
        source_contact = self._contact_metrics(from_arm, object_name)
        source_retention = self._retention_is_active(from_arm, object_name)
        if not source_contact["verified"] and not source_retention:
            self._last_action_result = {
                "ok": False,
                "action": "handoff",
                "message": "source grasp lost contact before handoff",
                "contact": source_contact,
            }
            return
        target, object_orientation = self._object_grasp_pose(object_name, to_arm)
        receiver_pose_goal = self._last_pose_goal.get(to_arm, {})
        target_quaternion = np.asarray(
            receiver_pose_goal.get("orientation") or object_orientation,
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(self.data.site_xpos[self._pinch_site(to_arm)] - target))
        if distance > 0.028:
            self._last_action_result = {"ok": False, "action": "handoff", "message": "receiving arm is not close enough", "distance": distance}
            return

        self._set_jaw_collision_enabled(to_arm, True, object_name)
        self.data.ctrl[self._actuator_ids[f"{to_arm}_gripper_motor"]] = GRIPPER_CLOSED
        self._advance_physics(steps=GRASP_CLOSE_STEPS)
        self._physics_advanced_in_action = True
        receiver_contact = self._contact_metrics(to_arm, object_name)
        receiver_orientation_error = float(
            np.linalg.norm(
                _rotation_vector_error(
                    target_quaternion,
                    _matrix_to_quaternion(self.data.site_xmat[self._pinch_site(to_arm)]),
                )
            )
        )
        ok = bool(receiver_contact["verified"] and receiver_contact["jaw_count"] >= 2 and receiver_orientation_error < GRASP_ORIENTATION_TOLERANCE)
        if ok:
            # Capture the receiver's measured contact pose before dropping
            # the source constraint. Both constraints are switched before a
            # solver step so the cup cannot teleport between arms.
            self._set_retention(to_arm, object_name, True, forward=False)
            self._set_retention(from_arm, object_name, False, forward=False)
            mujoco.mj_forward(self.model, self.data)
            self.data.ctrl[self._actuator_ids[f"{from_arm}_gripper_motor"]] = GRIPPER_OPEN
            self._advance_physics(steps=35)
            self._set_jaw_collision_enabled(from_arm, False)
            self._held_by[object_name] = to_arm
            self._task_index += 1
        else:
            self.data.ctrl[self._actuator_ids[f"{to_arm}_gripper_motor"]] = GRIPPER_OPEN
            self._advance_physics(steps=24)
            self._set_jaw_collision_enabled(to_arm, False)
        self._last_action_result = {
            "ok": ok,
            "action": "handoff",
            "object": object_name,
            "from": from_arm,
            "to": to_arm,
            "distance": distance,
            "orientation_error": receiver_orientation_error,
            "contact": receiver_contact,
            "source_contact": source_contact,
            "source_retention": source_retention,
            "retention": {
                "active": ok,
                "constraint": f"{to_arm}_{object_name}_retention" if ok else None,
                "validated_after": "receiver_two_jaw_contact" if ok else None,
            },
            "message": "bimanual hand-off verified" if ok else "receiving physical grasp failed",
        }

    def observe(self, include_camera: bool = False) -> dict[str, Any]:
        """Return camera, joint, gripper, object, and contact observations."""

        objects: dict[str, dict[str, Any]] = {}
        for object_name in OBJECT_NAMES:
            body_id = self._body_ids[object_name]
            grasp_target, grasp_orientation = self._object_grasp_pose(object_name, "left")
            arm_grasp_targets = {
                arm: self._object_grasp_pose(object_name, arm)[0].astype(float).tolist()
                for arm in ARM_NAMES
            }
            arm_grasp_orientations = {
                arm: self._object_grasp_pose(object_name, arm)[1].astype(float).tolist()
                for arm in ARM_NAMES
            }
            objects[object_name] = {
                "position": self.data.xpos[body_id].astype(float).tolist(),
                "grasp_site": self.data.site_xpos[self._object_site(object_name)].astype(float).tolist(),
                "grasp_target": grasp_target.astype(float).tolist(),
                "grasp_target_by_arm": arm_grasp_targets,
                "grasp_orientation": grasp_orientation.astype(float).tolist(),
                "grasp_orientation_by_arm": arm_grasp_orientations,
                "grasp_estimator": "calibrated-mujoco-geometry",
                "quaternion": self.data.xquat[body_id].astype(float).tolist(),
                "body_id": int(body_id),
                "held_by": self._held_by.get(object_name),
                "placed": object_name in self._placed,
            }

        arms = {
            arm: {
                "joints": [
                    float(self.data.qpos[self.model.jnt_qposadr[self._joint_ids[f"{arm}_{joint}"]]])
                    for joint in ARM_JOINTS
                ],
                "gripper": float(self.data.qpos[self.model.jnt_qposadr[self._joint_ids[f"{arm}_gripper"]]]),
                "end_effector": self.data.site_xpos[self._arm_site(arm)].astype(float).tolist(),
                "pinch_point": self.data.site_xpos[self._pinch_site(arm)].astype(float).tolist(),
                "pinch_orientation": _matrix_to_quaternion(self.data.site_xmat[self._pinch_site(arm)]).astype(float).tolist(),
                "jaw_geoms": [int(geom_id) for geom_id in self._jaw_geom_ids[arm]],
                "retention": {
                    object_name: {
                        "active": self._retention_is_active(arm, object_name),
                        "constraint": f"{arm}_{object_name}_retention",
                    }
                    for object_name in OBJECT_NAMES
                },
            }
            for arm in ARM_NAMES
        }

        result: dict[str, Any] = {
            "environment": "mujoco-dual-arm-dinner-table",
            "model": self.model_path.name,
            "seed": self.seed,
            "time": float(self.data.time),
            "qpos": self.data.qpos.astype(float).tolist(),
            "qvel": self.data.qvel.astype(float).tolist(),
            "arms": arms,
            "objects": objects,
            "drawer": float(self.data.qpos[self.model.jnt_qposadr[self._joint_ids["drawer_slide"]]]),
            "contacts": self.contacts(),
            "grasp_evidence": dict(self._last_grasp_metrics),
            "perturbation": {
                "placement_jitter": self.perturbation.placement_jitter,
                "mass_scale": self.perturbation.mass_scale,
                "friction_scale": self.perturbation.friction_scale,
                "object_scale": self.perturbation.object_scale,
                "lighting_scale": self.perturbation.lighting_scale,
                "background": self.perturbation.background,
            },
            "task": {
                "index": self._task_index,
                "next": self._current_step_id(),
                "completed": self._task_index >= len(self._task_plan),
                "held_by": dict(self._held_by),
                "placed": sorted(self._placed),
                "plan": [step.id for step in self._task_plan],
                "last_action": dict(self._last_action_result),
            },
        }
        if include_camera:
            available = {
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, index)
                for index in range(self.model.ncam)
            }
            camera_views = {
                name: self.render(name).tolist()
                for name in CAMERA_NAMES
                if name in available
            }
            if "overview" not in camera_views:
                raise ValueError("MuJoCo scene must provide an overview camera for VLA observations")
            result["camera_rgb"] = camera_views["overview"]
            result["camera_rgb_views"] = camera_views
        return result

    def contacts(self) -> list[dict[str, Any]]:
        contacts: list[dict[str, Any]] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_a = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
            geom_b = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
            contacts.append({"geom1": geom_a, "geom2": geom_b, "distance": float(contact.dist)})
        return contacts

    def render(self, camera_name: str = "overview") -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "FortifiersMuJoCoEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
