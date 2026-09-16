"""Small camera/state/instruction behavior-cloning policy.

This is a genuine neural imitation path for local iteration. It is deliberately
separate from :class:`SmolVLAAdapter`: the compact model is easy to train and
export on a CPU, while the SmolVLA adapter remains the higher-capacity VLA
submission path. Both emit the same guarded MuJoCo actuator contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .env import FortifiersMuJoCoEnv
from .policy import PolicyAction
from .vla_adapter import STATE_NAMES, action_to_joint_targets, observation_frame, state_vector


LANGUAGE_VOCAB = (
    "set",
    "dinner",
    "table",
    "open",
    "drawer",
    "retrieve",
    "fork",
    "spoon",
    "place",
    "plate",
    "hand",
    "off",
    "cup",
    "right",
    "left",
    "before",
    "final",
)

# The neural action contract follows MuJoCo's actuator order. The observation
# contract keeps wrist roll before wrist pitch because that is the ergonomic
# arm-state order exposed to VLA/LeRobot. Keep this mapping explicit: swapping
# the two wrist joints silently corrupts every learned grasp target.
CONTROL_STATE_INDICES = (12, 0, 1, 2, 4, 3, 5, 6, 7, 8, 10, 9, 11)


def relative_delta_scales(action_names: list[str]) -> np.ndarray:
    """Return conservative per-step target increments in actuator order."""

    return np.asarray(
        [
            0.30 if name == "drawer_position" else 0.15 if "gripper" in name else 0.20
            for name in action_names
        ],
        dtype=np.float32,
    )


try:  # Keep the base MuJoCo install importable without the optional VLA stack.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in the dependency check
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is not installed. Install the optional VLA dependencies "
            "with `uv sync --project sim/mujoco --extra vla`."
        )
    return torch


def language_vector(instruction: str) -> np.ndarray:
    """Encode the instruction as a deterministic bag-of-words input."""

    tokens = set(re.findall(r"[a-z]+", instruction.lower()))
    return np.asarray([1.0 if word in tokens else 0.0 for word in LANGUAGE_VOCAB], dtype=np.float32)


if nn is not None:

    class CameraStatePolicy(nn.Module):
        """Compact multi-view behavior-cloning network."""

        def __init__(self, num_cameras: int, state_dim: int, action_dim: int) -> None:
            super().__init__()
            self.num_cameras = num_cameras
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.visual = nn.Sequential(
                nn.Conv2d(num_cameras * 3, 24, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.state = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
            )
            self.language = nn.Sequential(
                nn.Linear(len(LANGUAGE_VOCAB), 32),
                nn.GELU(),
            )
            self.head = nn.Sequential(
                nn.Linear(64 + 64 + 32, 128),
                nn.GELU(),
                nn.Linear(128, action_dim),
                nn.Tanh(),
            )

        def forward(self, images: Any, state: Any, language: Any) -> Any:
            visual = self.visual(images).flatten(1)
            return self.head(torch.cat((visual, self.state(state), self.language(language)), dim=1))


    class CameraStateIntentPolicy(nn.Module):
        """Compact neural task-intent head for a verified control stack.

        This head predicts which live task step should execute next. It is
        intentionally separate from ``CameraStatePolicy``: the former is a
        neural high-level decision boundary, while the latter is a direct
        joint-target behavior-cloning policy. A runtime evaluator must gate
        the measured controller on this prediction and fail closed on a
        disagreement.
        """

        def __init__(self, num_cameras: int, state_dim: int, num_steps: int) -> None:
            super().__init__()
            self.num_cameras = num_cameras
            self.state_dim = state_dim
            self.num_steps = num_steps
            self.visual = nn.Sequential(
                nn.Conv2d(num_cameras * 3, 8, kernel_size=5, stride=4, padding=2),
                nn.GELU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=4, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.state = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
            )
            self.language = nn.Sequential(
                nn.Linear(len(LANGUAGE_VOCAB), 16),
                nn.GELU(),
            )
            self.head = nn.Sequential(
                nn.Linear(16 + 32 + 16, 64),
                nn.GELU(),
                nn.Linear(64, num_steps),
            )

        def forward(self, images: Any, state: Any, language: Any) -> Any:
            visual = self.visual(images).flatten(1)
            return self.head(torch.cat((visual, self.state(state), self.language(language)), dim=1))

else:

    class CameraStatePolicy:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            _require_torch()

    class CameraStateIntentPolicy:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            _require_torch()


def model_inputs(observation: Mapping[str, Any], instruction: str, camera_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare fixed-shape tensors shared by PyTorch and OpenVINO adapters."""

    frame = observation_frame(observation, instruction, camera_names)
    images = np.concatenate(
        [np.asarray(frame[f"camera{index}"], dtype=np.float32) for index, _ in enumerate(camera_names, start=1)],
        axis=-1,
    )
    images = np.transpose(images / 255.0, (2, 0, 1))[None, ...]
    state = np.asarray(frame["state"], dtype=np.float32)[None, ...]
    language = language_vector(instruction)[None, ...]
    return images, state, language


def _ranges(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    names = list(payload["action_names"])
    ranges = payload["action_ranges"]
    low = np.asarray([ranges[name][0] for name in names], dtype=np.float32)
    high = np.asarray([ranges[name][1] for name in names], dtype=np.float32)
    return low, high


def denormalize_action(action: Any, payload: Mapping[str, Any]) -> np.ndarray:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    low, high = _ranges(payload)
    if vector.shape != low.shape:
        raise ValueError(f"imitation policy action shape {vector.shape} does not match {low.shape}")
    return low + ((np.clip(vector, -1.0, 1.0) + 1.0) * 0.5) * (high - low)


def state_to_control_vector(observation: Mapping[str, Any]) -> np.ndarray:
    """Pack live proprioception in the ordered actuator contract."""

    state = state_vector(observation)
    return state[list(CONTROL_STATE_INDICES)].astype(np.float32)


def decode_action(action: Any, payload: Mapping[str, Any], observation: Mapping[str, Any]) -> np.ndarray:
    """Decode either legacy absolute targets or robust relative targets."""

    vector = np.asarray(action.detach().cpu().numpy() if hasattr(action, "detach") else action, dtype=np.float32).reshape(-1)
    if payload.get("action_mode") != "relative_to_proprio":
        return denormalize_action(vector, payload)
    scales = np.asarray(payload["delta_scales"], dtype=np.float32)
    if vector.shape != scales.shape:
        raise ValueError(f"imitation policy delta shape {vector.shape} does not match {scales.shape}")
    return state_to_control_vector(observation) + np.clip(vector, -1.0, 1.0) * scales


def load_checkpoint(checkpoint: Path, device: str) -> tuple[Any, dict[str, Any]]:
    runtime = _require_torch()
    payload = runtime.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "fortifiers.imitation-checkpoint.v1":
        raise ValueError("checkpoint is not a fortifiers.imitation-checkpoint.v1 artifact")
    config = payload["model_config"]
    model = CameraStatePolicy(
        num_cameras=int(config["num_cameras"]),
        state_dim=int(config["state_dim"]),
        action_dim=int(config["action_dim"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def load_intent_checkpoint(checkpoint: Path, device: str) -> tuple[Any, dict[str, Any]]:
    """Load a neural task-intent checkpoint with an explicit schema gate."""

    runtime = _require_torch()
    payload = runtime.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "fortifiers.neural-intent-checkpoint.v1":
        raise ValueError("checkpoint is not a fortifiers.neural-intent-checkpoint.v1 artifact")
    config = payload["model_config"]
    model = CameraStateIntentPolicy(
        num_cameras=int(config["num_cameras"]),
        state_dim=int(config["state_dim"]),
        num_steps=int(config["num_steps"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


class ImitationPolicyAdapter:
    """Run a trained compact camera policy through native MuJoCo."""

    name = "camera-state-imitation-v1"

    def __init__(self, checkpoint: str | Path, env: FortifiersMuJoCoEnv, device: str = "cpu") -> None:
        self.device = device
        self.env = env
        self.model, self.payload = load_checkpoint(Path(checkpoint), device)
        self.control = env.control_spec()
        if tuple(self.payload["action_names"]) != tuple(self.control["names"]):
            raise ValueError("checkpoint actuator names do not match the live MuJoCo control contract")
        self.camera_names = tuple(self.payload["camera_names"])
        if tuple(self.payload["state_names"]) != STATE_NAMES:
            raise ValueError("checkpoint state metadata does not match the live MuJoCo state contract")

    def reset(self) -> None:
        return None

    def predict(self, observation: Mapping[str, Any], instruction: str) -> PolicyAction:
        runtime = _require_torch()
        images, state, language = model_inputs(observation, instruction, self.camera_names)
        with runtime.inference_mode():
            output = self.model(
                runtime.from_numpy(images).to(self.device),
                runtime.from_numpy(state).to(self.device),
                runtime.from_numpy(language).to(self.device),
            )
        targets = action_to_joint_targets(decode_action(output, self.payload, observation), self.control)
        return PolicyAction(
            "imitation_joint_targets",
            None,
            None,
            {
                "type": "joint_targets",
                "targets": targets,
                "policy": self.name,
                "checkpoint": str(self.payload.get("checkpoint", "")),
                "camera_views": list(self.camera_names),
            },
        )


class OpenVINOImitationPolicyAdapter:
    """Run the same trained policy through a compiled OpenVINO IR model."""

    name = "camera-state-imitation-openvino-v1"

    def __init__(self, model_path: str | Path, env: FortifiersMuJoCoEnv, device: str = "CPU") -> None:
        try:
            import openvino as ov
        except ImportError as exc:  # pragma: no cover - depends on optional edge extra
            raise RuntimeError("OpenVINO is not installed. Install the optional edge dependencies.") from exc
        self.env = env
        self.device = device
        self.model_path = Path(model_path)
        self.payload = json.loads(self.model_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.camera_names = tuple(self.payload["camera_names"])
        self.control = env.control_spec()
        if tuple(self.payload["action_names"]) != tuple(self.control["names"]):
            raise ValueError("OpenVINO model actuator names do not match the live MuJoCo control contract")
        self.compiled = ov.Core().compile_model(str(self.model_path), device)
        self.request = self.compiled.create_infer_request()
        self._input_specs = list(self.payload.get("inputs", []))

    def reset(self) -> None:
        return None

    def predict(self, observation: Mapping[str, Any], instruction: str) -> PolicyAction:
        images, state, language = model_inputs(observation, instruction, self.camera_names)
        tensors = {"images": images, "state": state, "language": language}
        feeds: dict[Any, np.ndarray] = {}
        for index, port in enumerate(self.compiled.inputs):
            semantic_name = None
            port_names = set(port.names)
            for spec in self._input_specs:
                if spec.get("name") in port_names or spec.get("name") == port.any_name:
                    semantic_name = spec.get("key")
                    break
            if semantic_name not in tensors:
                semantic_name = ("images", "state", "language")[index]
            feeds[port] = tensors[semantic_name]
        result = self.request.infer(feeds)
        action = next(iter(result.values()))
        targets = action_to_joint_targets(decode_action(action, self.payload, observation), self.control)
        return PolicyAction(
            "imitation_joint_targets",
            None,
            None,
            {
                "type": "joint_targets",
                "targets": targets,
                "policy": self.name,
                "checkpoint": str(self.model_path),
                "camera_views": list(self.camera_names),
                "device": self.device,
            },
        )
