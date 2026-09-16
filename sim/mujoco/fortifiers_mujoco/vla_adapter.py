"""Camera-and-language policy adapter for the dinner-table environment.

The adapter is intentionally optional. Importing this module does not install
or import PyTorch/LeRobot; a real checkpoint is required before it can emit an
action. The MuJoCo environment remains the authority for contact, retention,
release, hand-off, and task-success verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .env import CAMERA_NAMES, FortifiersMuJoCoEnv
from .policy import PolicyAction


STATE_NAMES = (
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_gripper",
    "drawer_slide",
    "task_progress",
    "task_phase_0",
    "task_phase_1",
    "task_phase_2",
    "task_phase_3",
    "task_phase_4",
    "task_phase_5",
    "task_phase_6",
)


@dataclass(frozen=True)
class VLAConfig:
    """Runtime configuration for a fine-tuned LeRobot-compatible policy."""

    checkpoint: str
    device: str = "cpu"
    camera_names: tuple[str, ...] = ("overview", "overhead", "left_oblique")
    robot_type: str = "fortifiers_dual_so101"

    def validate(self) -> None:
        if not self.checkpoint.strip():
            raise ValueError("VLA checkpoint must be a local path or Hub model id")
        if not self.camera_names:
            raise ValueError("VLA requires at least one camera view")
        unknown = set(self.camera_names) - set(CAMERA_NAMES)
        if unknown:
            raise ValueError(f"Unknown MuJoCo camera views: {sorted(unknown)}")


def state_vector(observation: Mapping[str, Any]) -> np.ndarray:
    """Pack the live structured observation into the training state frame."""

    values: list[float] = []
    arms = observation.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("VLA observation is missing the arms mapping")
    for arm in ("left", "right"):
        arm_state = arms.get(arm)
        if not isinstance(arm_state, Mapping):
            raise ValueError(f"VLA observation is missing the {arm} arm")
        joints = arm_state.get("joints")
        if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)) or len(joints) != 5:
            raise ValueError(f"VLA observation has an invalid {arm} joint vector")
        values.extend(float(value) for value in joints)
        values.append(float(arm_state["gripper"]))
    values.append(float(observation["drawer"]))
    task = observation.get("task", {})
    plan = task.get("plan", []) if isinstance(task, Mapping) else []
    index = float(task.get("index", 0)) if isinstance(task, Mapping) else 0.0
    values.append(index / max(1, len(plan)))
    phase_count = len(plan) + 1 if plan else 7
    phase_index = int(np.clip(index, 0, phase_count - 1))
    values.extend(1.0 if phase == phase_index else 0.0 for phase in range(7))
    return np.asarray(values, dtype=np.float32)


def observation_frame(
    observation: Mapping[str, Any],
    instruction: str,
    camera_names: Sequence[str],
) -> dict[str, Any]:
    """Convert MuJoCo output into the raw LeRobot observation convention.

    The returned keys intentionally omit the ``observation.`` prefix because
    LeRobot's ``build_inference_frame`` adds that prefix from feature metadata.
    Images stay HWC uint8 until LeRobot's processor normalizes and permutes
    them, matching the official policy examples.
    """

    views = observation.get("camera_rgb_views", {})
    if not isinstance(views, Mapping):
        views = {}
    frame: dict[str, Any] = {"state": state_vector(observation), "task": instruction}
    for index, camera_name in enumerate(camera_names, start=1):
        image = views.get(camera_name)
        if image is None and camera_name == "overview":
            image = observation.get("camera_rgb")
        if image is None:
            raise ValueError(
                f"MuJoCo observation is missing camera view {camera_name!r}; "
                "call observe(include_camera=True) and use the generated scene"
            )
        array = np.asarray(image, dtype=np.uint8)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Camera {camera_name!r} must be an HWC RGB image")
        frame[f"camera{index}"] = array
    return frame


def dataset_features(
    action_names: Sequence[str],
    camera_names: Sequence[str],
    image_shape: tuple[int, int, int],
) -> dict[str, dict[str, Any]]:
    """Return the versioned feature contract used for collection/training."""

    features: dict[str, dict[str, Any]] = {
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": list(action_names),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
    }
    for index, _camera_name in enumerate(camera_names, start=1):
        features[f"observation.images.camera{index}"] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }
    return features


def action_to_joint_targets(
    action: Any,
    control_spec: Mapping[str, Any],
) -> dict[str, float]:
    """Validate and clamp a model action to the MuJoCo actuator contract."""

    if hasattr(action, "detach"):
        action = action.detach().to("cpu").numpy()
    vector = np.asarray(action, dtype=np.float64)
    if vector.ndim == 3:
        vector = vector[0, 0]
    elif vector.ndim == 2:
        vector = vector[0]
    vector = vector.reshape(-1)
    names = list(control_spec["names"])
    ranges = control_spec["ranges"]
    if vector.shape != (len(names),):
        raise ValueError(
            f"VLA action shape {tuple(vector.shape)} does not match the MuJoCo actuator contract "
            f"({len(names)},)"
        )
    if not np.isfinite(vector).all():
        raise ValueError("VLA emitted a non-finite actuator target")
    return {
        name: float(np.clip(value, ranges[name][0], ranges[name][1]))
        for name, value in zip(names, vector)
    }


class SmolVLAAdapter:
    """Run a fine-tuned SmolVLA checkpoint against live MuJoCo camera state."""

    name = "smolvla-camera-language"

    def __init__(self, config: VLAConfig, env: FortifiersMuJoCoEnv) -> None:
        config.validate()
        self.config = config
        self.env = env
        self.control = env.control_spec()
        self.features = dataset_features(
            self.control["names"],
            config.camera_names,
            (env.height, env.width, 3),
        )
        try:
            import torch
            from lerobot.policies import make_pre_post_processors
            try:
                from lerobot.policies.smolvla import SmolVLAPolicy
            except ImportError:
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.utils import build_inference_frame
        except ImportError as exc:
            raise RuntimeError(
                "SmolVLA is not installed. Install the optional VLA dependencies "
                "and provide a fine-tuned checkpoint before using --policy smolvla."
            ) from exc

        self._torch = torch
        self._build_inference_frame = build_inference_frame
        self.device = torch.device(config.device)
        self.model = SmolVLAPolicy.from_pretrained(config.checkpoint).to(self.device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            config.checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def reset(self) -> None:
        """Clear recurrent/chunked policy state at the start of an episode."""

        reset = getattr(self.model, "reset", None)
        if callable(reset):
            reset()

    def predict(self, observation: Mapping[str, Any], instruction: str) -> PolicyAction:
        raw_frame = observation_frame(observation, instruction, self.config.camera_names)
        inference_frame = self._build_inference_frame(
            observation=raw_frame,
            device=self.device,
            ds_features=self.features,
            task=instruction,
            robot_type=self.config.robot_type,
        )
        with self._torch.inference_mode():
            action = self.model.select_action(self.preprocess(inference_frame))
        action = self.postprocess(action)
        targets = action_to_joint_targets(action, self.control)
        return PolicyAction(
            "vla_joint_targets",
            None,
            None,
            {
                "type": "joint_targets",
                "targets": targets,
                "policy": self.name,
                "checkpoint": self.config.checkpoint,
                "camera_views": list(self.config.camera_names),
            },
        )
