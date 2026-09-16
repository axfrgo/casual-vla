"""Export the local VLA demonstration staging format to LeRobot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .vla_adapter import dataset_features


def export(manifest_path: Path, repo_id: str, root: Path, fps: int) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed. Install the optional VLA dependencies before exporting."
        ) from exc

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    camera_names = tuple(manifest["camera_names"])
    action_names = tuple(manifest["action_names"])
    first_episode = manifest["episodes"][0]
    first_data = np.load(manifest_path.parent / first_episode["path"])
    image_shape = tuple(int(value) for value in first_data[f"image_{camera_names[0]}"][0].shape)
    features = dataset_features(action_names, camera_names, image_shape)
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root)
    for episode in manifest["episodes"]:
        data = np.load(manifest_path.parent / episode["path"])
        for index in range(int(episode["steps"])):
            frame = {
                "observation.state": data["state"][index],
                "action": data["action"][index],
                "task": manifest["instruction"],
            }
            for camera_index, camera_name in enumerate(camera_names, start=1):
                frame[f"observation.images.camera{camera_index}"] = data[f"image_{camera_name}"][index]
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VLA staging demonstrations to LeRobot")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be positive")
    export(args.manifest, args.repo_id, args.root, args.fps)
    print(f"Exported LeRobot dataset {args.repo_id} to {args.root}")


if __name__ == "__main__":
    main()
