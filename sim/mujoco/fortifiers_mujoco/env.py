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


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PACKAGE_ROOT / "models" / "official_dual_so101_dinner_table.xml"
PROXY_MODEL_PATH = PACKAGE_ROOT / "models" / "dual_arm_dinner_table.xml"
OFFICIAL_SO101_DIR = PACKAGE_ROOT / "assets" / "so101" / "SO101"
OFFICIAL_SO101_MJCF_PATH = OFFICIAL_SO101_DIR / "so101_new_calib.xml"
OBJECT_NAMES = ("plate", "cup", "fork", "spoon")
ARM_NAMES = ("left", "right")


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

    Physics, free-body objects, drawer motion, camera rendering, and randomized
    scene state are real MuJoCo state. Policy inference and task-level grasp
    planning remain separate integrations.
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

    def __post_init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._rng = np.random.default_rng(0)
        self._index_named_entities()
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
        for object_name in OBJECT_NAMES:
            joint_name = f"{object_name}_free"
            if joint_name not in self._joint_ids:
                raise ValueError(f"Model is missing required joint {joint_name}")
            self._object_joint_ids[object_name] = self._joint_ids[joint_name]

    def _names(self, object_type: mujoco.mjtObj) -> list[str]:
        names: list[str] = []
        for index in range(self.model.nbody if object_type == mujoco.mjtObj.mjOBJ_BODY else self.model.ngeom if object_type == mujoco.mjtObj.mjOBJ_GEOM else self.model.njnt if object_type == mujoco.mjtObj.mjOBJ_JOINT else self.model.nu):
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

        drawer_joint = self._joint_ids["drawer_slide"]
        self.data.qpos[self.model.jnt_qposadr[drawer_joint]] = 0.05 + 0.24 * self._rng.random()

        # Keep all objects on the tabletop while ensuring each reset changes
        # the visual arrangement.  Orientation is randomized around z.
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
        return self.observe(include_camera=False)

    def _sample_object_positions(self) -> dict[str, tuple[float, float, float]]:
        center = np.array([0.0, 0.04])
        available: list[tuple[float, float]] = []
        for x in np.linspace(-0.7, 0.7, 8):
            for y in np.linspace(-0.20, 0.38, 5):
                available.append((float(x), float(y)))
        self._rng.shuffle(available)
        positions: dict[str, tuple[float, float, float]] = {}
        minimum_distance = 0.22
        for object_name in OBJECT_NAMES:
            for x, y in available:
                point = np.array([x, y])
                if all(np.linalg.norm(point - np.array(existing[:2])) >= minimum_distance for existing in positions.values()):
                    jitter = self._rng.uniform(
                        -self.perturbation.placement_jitter,
                        self.perturbation.placement_jitter,
                        size=2,
                    )
                    candidate = center + point * 0.55 + jitter
                    candidate[0] = np.clip(candidate[0], -0.95, 0.95)
                    candidate[1] = np.clip(candidate[1], -0.40, 0.52)
                    positions[object_name] = (float(candidate[0]), float(candidate[1]), 0.86)
                    break
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
        """Advance physics from either joint targets or raw actuator targets.

        Supported mappings:

        - ``{"type": "joint_targets", "targets": {name: value}}``
        - ``{"type": "open_drawer"}`` / ``{"type": "close_drawer"}``
        - ``{"type": "noop"}``

        Grasp/place semantics are intentionally not hidden here yet; they will
        be added once the official SO-101 asset and policy action convention
        are selected.
        """

        if action is None:
            self.data.ctrl[:] = 0
        elif isinstance(action, Mapping):
            self._apply_mapping_action(action)
        else:
            controls = np.asarray(action, dtype=np.float64)
            if controls.shape != (self.model.nu,):
                raise ValueError(f"raw action must have shape {(self.model.nu,)}")
            self.data.ctrl[:] = controls

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self.observe(include_camera=False)

    def _apply_mapping_action(self, action: Mapping[str, Any]) -> None:
        action_type = action.get("type")
        if action_type == "noop":
            self.data.ctrl[:] = 0
            return
        if action_type in {"open_drawer", "close_drawer"}:
            actuator_id = self._actuator_ids["drawer_position"]
            self.data.ctrl[actuator_id] = 0.30 if action_type == "open_drawer" else 0.0
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

    def observe(self, include_camera: bool = False) -> dict[str, Any]:
        """Return camera, joint, gripper, object, and contact observations."""

        objects: dict[str, dict[str, Any]] = {}
        for object_name in OBJECT_NAMES:
            body_id = self._body_ids[object_name]
            objects[object_name] = {
                "position": self.data.xpos[body_id].astype(float).tolist(),
                "quaternion": self.data.xquat[body_id].astype(float).tolist(),
                "body_id": int(body_id),
            }

        arms = {
            arm: {
                "joints": [
                    float(self.data.qpos[self.model.jnt_qposadr[self._joint_ids[f"{arm}_{joint}"]]])
                    for joint in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_roll", "wrist_pitch")
                ],
                "gripper": float(self.data.qpos[self.model.jnt_qposadr[self._joint_ids[f"{arm}_gripper"]]]),
                "end_effector": self.data.site_xpos[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_grasp_site")
                ].astype(float).tolist(),
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
            "perturbation": {
                "placement_jitter": self.perturbation.placement_jitter,
                "mass_scale": self.perturbation.mass_scale,
                "friction_scale": self.perturbation.friction_scale,
                "object_scale": self.perturbation.object_scale,
                "lighting_scale": self.perturbation.lighting_scale,
                "background": self.perturbation.background,
            },
        }
        if include_camera:
            result["camera_rgb"] = self.render().tolist()
        return result

    def contacts(self) -> list[dict[str, Any]]:
        contacts: list[dict[str, Any]] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_a = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
            geom_b = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
            contacts.append({"geom1": geom_a, "geom2": geom_b, "distance": float(contact.dist)})
        return contacts

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self._renderer.update_scene(self.data, camera="overview")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "FortifiersMuJoCoEnv":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
