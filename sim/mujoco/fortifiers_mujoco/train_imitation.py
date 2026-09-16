"""Train the compact camera/state/instruction behavior-cloning policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .imitation_policy import (
    CONTROL_STATE_INDICES,
    LANGUAGE_VOCAB,
    STATE_NAMES,
    CameraStatePolicy,
    _require_torch,
    language_vector,
    relative_delta_scales,
)


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes")
    if not episodes:
        raise ValueError("demonstration manifest has no episodes")
    camera_names = tuple(manifest["camera_names"])
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    images: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    for episode_index, episode in enumerate(episodes):
        data = np.load(manifest_path.parent / episode["path"])
        episode_states = np.asarray(data["state"], dtype=np.float32)
        states.append(episode_states)
        actions.append(np.asarray(data["action"], dtype=np.float32))
        episode_ids.append(np.full(len(episode_states), episode_index, dtype=np.int64))
        images.append(
            np.concatenate(
                [np.asarray(data[f"image_{name}"], dtype=np.float32) for name in camera_names],
                axis=-1,
            )
        )
    image_array = np.transpose(np.concatenate(images, axis=0) / 255.0, (0, 3, 1, 2))
    state_array = np.concatenate(states, axis=0)
    action_array = np.concatenate(actions, axis=0)
    return manifest, image_array, state_array, action_array, np.concatenate(episode_ids, axis=0)


def train(
    manifest_path: Path,
    output: Path,
    epochs: int = 160,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 7,
    augmentation_repeats: int = 2,
    device: str = "auto",
) -> dict[str, Any]:
    runtime = _require_torch()
    if epochs < 1 or batch_size < 1 or augmentation_repeats < 1:
        raise ValueError("epochs, batch_size, and augmentation_repeats must be positive")
    if device == "auto":
        device = "cuda" if runtime.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not runtime.cuda.is_available():
        raise RuntimeError("CUDA training was requested but torch.cuda.is_available() is false")
    train_device = runtime.device(device)
    runtime.manual_seed(seed)
    np.random.seed(seed)
    manifest, image_array, state_array, action_array, episode_ids = _load_manifest(manifest_path)
    if state_array.shape[1] != len(STATE_NAMES):
        raise ValueError("demonstration state dimension does not match the MuJoCo state contract")
    camera_names = tuple(manifest["camera_names"])
    action_names = list(manifest["action_names"])
    action_ranges = manifest.get("control_ranges")
    if not action_ranges:
        raise ValueError("manifest is missing control_ranges; recollect demonstrations with the current collector")
    low = np.asarray([action_ranges[name][0] for name in action_names], dtype=np.float32)
    high = np.asarray([action_ranges[name][1] for name in action_names], dtype=np.float32)
    # Align the supervised relative target with MuJoCo actuator order. The
    # observation schema stores wrist roll before wrist pitch, while the
    # official actuator schema stores pitch before roll.
    state_as_action = state_array[:, list(CONTROL_STATE_INDICES)]
    delta_scales = relative_delta_scales(action_names)
    delta_actions = action_array - state_as_action
    normalized_actions = np.clip(delta_actions / delta_scales, -1.0, 1.0)
    language = np.repeat(language_vector(manifest["instruction"])[None, :], len(state_array), axis=0)

    count = len(state_array)
    generator = np.random.default_rng(seed)
    episode_count = len(manifest["episodes"])
    if episode_count >= 2:
        episode_order = generator.permutation(episode_count)
        validation_episode_count = max(1, int(round(episode_count * 0.2)))
        validation_episode_ids = set(int(value) for value in episode_order[:validation_episode_count])
        validation_indices = np.flatnonzero(np.isin(episode_ids, list(validation_episode_ids)))
        train_indices = np.flatnonzero(~np.isin(episode_ids, list(validation_episode_ids)))
    else:
        # A one-episode smoke run is useful for validating the pipeline, but it
        # cannot provide episode-level generalization evidence.
        order = generator.permutation(count)
        split = max(1, int(count * 0.8)) if count > 1 else count
        train_indices = order[:split]
        validation_indices = order[split:]
    tensors = tuple(
        runtime.from_numpy(array)
        for array in (image_array, state_array, language, normalized_actions)
    )
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    dataset = TensorDataset(*tensors)
    base_indices = train_indices.tolist()

    class StateNoiseDataset(Dataset):
        """Repeat training frames with small proprioceptive perturbations.

        Absolute-target behavior cloning is otherwise vulnerable to a small
        early tracking error: the rollout leaves the teacher manifold and the
        model has never seen the resulting state. The camera frame stays real;
        only the measured state is perturbed within a conservative sensor/
        tracking envelope.
        """

        def __init__(self, source: Any, indices: list[int], repeats: int) -> None:
            self.source = source
            self.indices = indices
            self.repeats = repeats

        def __len__(self) -> int:
            return len(self.indices) * self.repeats

        def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any]:
            source_index = self.indices[index % len(self.indices)]
            image, state, language_value, action = self.source[source_index]
            state = state.clone()
            if index >= len(self.indices):
                noise = runtime.zeros_like(state)
                noise[:13] = runtime.randn(13) * 0.025
                noise[13:] = 0
                state = state + noise
            return image, state, language_value, action

    loader = DataLoader(
        StateNoiseDataset(dataset, base_indices, augmentation_repeats),
        batch_size=min(batch_size, len(base_indices)),
        shuffle=True,
        generator=runtime.Generator().manual_seed(seed),
    )
    model = CameraStatePolicy(len(camera_names), len(STATE_NAMES), len(action_names)).to(train_device)
    optimizer = runtime.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = runtime.nn.SmoothL1Loss()
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    final_train_loss = float("inf")
    final_validation_loss: float | None = None
    for _epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for batch_images, batch_state, batch_language, batch_action in loader:
            batch_images = batch_images.to(train_device)
            batch_state = batch_state.to(train_device)
            batch_language = batch_language.to(train_device)
            batch_action = batch_action.to(train_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_images, batch_state, batch_language)
            loss = loss_fn(prediction, batch_action)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        final_train_loss = sum(train_losses) / len(train_losses)
        if len(validation_indices):
            model.eval()
            with runtime.inference_mode():
                validation_prediction = model(
                    tensors[0][validation_indices].to(train_device),
                    tensors[1][validation_indices].to(train_device),
                    tensors[2][validation_indices].to(train_device),
                )
                final_validation_loss = float(
                    loss_fn(validation_prediction, tensors[3][validation_indices].to(train_device)).cpu()
                )
        else:
            final_validation_loss = None
        score = final_validation_loss if final_validation_loss is not None else final_train_loss
        if score < best_loss:
            best_loss = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint state")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "fortifiers.imitation-checkpoint.v1",
        "checkpoint": str(output),
        "model_type": "camera-state-instruction-behavior-cloning",
        "model_config": {
            "num_cameras": len(camera_names),
            "state_dim": len(STATE_NAMES),
            "action_dim": len(action_names),
            "language_dim": len(LANGUAGE_VOCAB),
            "image_height": int(image_array.shape[2]),
            "image_width": int(image_array.shape[3]),
        },
        "camera_names": list(camera_names),
        "state_names": list(STATE_NAMES),
        "action_names": action_names,
        "action_ranges": action_ranges,
        "action_mode": "relative_to_proprio",
        "delta_scales": delta_scales.tolist(),
        "instruction": manifest["instruction"],
        "language_vocab": list(LANGUAGE_VOCAB),
        "state_dict": best_state,
        "training": {
            "seed": seed,
            "epochs": epochs,
            "augmentation_repeats": augmentation_repeats,
            "samples": count,
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "episodes": episode_count,
            "validation_episode_ids": sorted(validation_episode_ids) if episode_count >= 2 else [],
            "train_loss": final_train_loss,
            "validation_loss": final_validation_loss,
            "source_manifest": str(manifest_path),
            "device": str(train_device),
            "cuda_available": bool(runtime.cuda.is_available()),
        },
        "evidence_boundary": "Real neural imitation checkpoint trained from native demonstrations; generalization and Intel performance require the separate evaluation reports.",
    }
    runtime.save(payload, output)
    return {key: value for key, value in payload.items() if key != "state_dict"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the camera/state imitation policy")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/imitation-policy.pt"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--augmentation-repeats", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.manifest,
                args.output,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.seed,
                args.augmentation_repeats,
                args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
