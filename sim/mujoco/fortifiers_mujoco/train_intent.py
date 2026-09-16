"""Train a compact neural task-intent policy from native demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .imitation_policy import (
    LANGUAGE_VOCAB,
    STATE_NAMES,
    CameraStateIntentPolicy,
    _require_torch,
    language_vector,
)
from .task import build_dinner_table_plan
from .train_imitation import _load_manifest


def train(
    manifest_path: Path,
    output: Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 11,
) -> dict[str, Any]:
    runtime = _require_torch()
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    runtime.manual_seed(seed)
    np.random.seed(seed)

    manifest, image_array, state_array, _action_array, episode_ids = _load_manifest(manifest_path)
    if state_array.shape[1] != len(STATE_NAMES):
        raise ValueError("demonstration state dimension does not match the MuJoCo state contract")
    task_step_ids = [step.id for step in build_dinner_table_plan()]
    phase_features = state_array[:, 14 : 14 + len(task_step_ids)]
    labels = np.asarray(np.argmax(phase_features, axis=1), dtype=np.int64)
    language = np.repeat(language_vector(manifest["instruction"])[None, :], len(state_array), axis=0)

    episode_count = len(manifest["episodes"])
    generator = np.random.default_rng(seed)
    if episode_count >= 2:
        episode_order = generator.permutation(episode_count)
        validation_episode_count = max(1, int(round(episode_count * 0.2)))
        validation_episode_ids = set(int(value) for value in episode_order[:validation_episode_count])
        validation_indices = np.flatnonzero(np.isin(episode_ids, list(validation_episode_ids)))
        train_indices = np.flatnonzero(~np.isin(episode_ids, list(validation_episode_ids)))
    else:
        order = generator.permutation(len(state_array))
        split = max(1, int(len(state_array) * 0.8)) if len(state_array) > 1 else len(state_array)
        train_indices = order[:split]
        validation_indices = order[split:]
        validation_episode_ids = set()

    tensors = tuple(runtime.from_numpy(array) for array in (image_array, state_array, language))
    labels_tensor = runtime.from_numpy(labels)
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(tensors[0], tensors[1], tensors[2], labels_tensor)
    loader = DataLoader(
        TensorDataset(
            tensors[0][train_indices],
            tensors[1][train_indices],
            tensors[2][train_indices],
            labels_tensor[train_indices],
        ),
        batch_size=min(batch_size, len(train_indices)),
        shuffle=True,
        generator=runtime.Generator().manual_seed(seed),
    )
    del dataset

    model = CameraStateIntentPolicy(len(manifest["camera_names"]), len(STATE_NAMES), len(task_step_ids))
    optimizer = runtime.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = runtime.nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    final_train_loss = float("inf")
    final_validation_loss: float | None = None
    final_validation_accuracy: float | None = None
    for _epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch_images, batch_state, batch_language, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_images, batch_state, batch_language)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_train_loss = sum(losses) / len(losses)

        model.eval()
        with runtime.inference_mode():
            if len(validation_indices):
                validation_logits = model(
                    tensors[0][validation_indices],
                    tensors[1][validation_indices],
                    tensors[2][validation_indices],
                )
                final_validation_loss = float(loss_fn(validation_logits, labels_tensor[validation_indices]).cpu())
                final_validation_accuracy = float(
                    (validation_logits.argmax(dim=1) == labels_tensor[validation_indices]).float().mean().cpu()
                )
            else:
                final_validation_loss = None
                final_validation_accuracy = None
        score = final_validation_loss if final_validation_loss is not None else final_train_loss
        if score < best_loss:
            best_loss = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no intent checkpoint state")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "fortifiers.neural-intent-checkpoint.v1",
        "checkpoint": str(output),
        "model_type": "camera-state-language-task-intent-classifier",
        "model_config": {
            "num_cameras": len(manifest["camera_names"]),
            "state_dim": len(STATE_NAMES),
            "num_steps": len(task_step_ids),
            "language_dim": len(LANGUAGE_VOCAB),
            "image_height": int(image_array.shape[2]),
            "image_width": int(image_array.shape[3]),
        },
        "camera_names": list(manifest["camera_names"]),
        "state_names": list(STATE_NAMES),
        "language_vocab": list(LANGUAGE_VOCAB),
        "task_step_ids": task_step_ids,
        "instruction": manifest["instruction"],
        "state_dict": best_state,
        "training": {
            "seed": seed,
            "epochs": epochs,
            "samples": len(state_array),
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "episodes": episode_count,
            "validation_episode_ids": sorted(validation_episode_ids),
            "train_loss": final_train_loss,
            "validation_loss": final_validation_loss,
            "validation_accuracy": final_validation_accuracy,
            "source_manifest": str(manifest_path),
        },
        "evidence_boundary": "Real neural task-intent checkpoint trained from native demonstrations; completion is evaluated only through the separately named verified low-level controller.",
    }
    runtime.save(payload, output)
    return {key: value for key, value in payload.items() if key != "state_dict"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the camera/state/language task-intent policy")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/neural-intent-policy.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output, args.epochs, args.batch_size, args.learning_rate, args.seed), indent=2))


if __name__ == "__main__":
    main()
