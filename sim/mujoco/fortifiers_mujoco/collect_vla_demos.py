"""Collect reproducible RGB/state/action demonstrations for VLA fine-tuning.

The output is a dependency-light staging format. It preserves the same camera
and actuator contract used by :mod:`vla_adapter`; a LeRobot export can consume
the per-episode arrays without rerunning MuJoCo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .env import FortifiersMuJoCoEnv
from .evaluate_policy import INSTRUCTION, perturbation_for_episode
from .policy import DinnerTablePolicy
from .vla_adapter import STATE_NAMES, observation_frame


def collect(
    episodes: int,
    start_seed: int,
    output: Path,
    camera_names: tuple[str, ...],
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with FortifiersMuJoCoEnv(width=160, height=120) as env:
        control_names = env.control_spec()["names"]
        control_ranges = env.control_spec()["ranges"]
        teacher = DinnerTablePolicy()
        for offset in range(episodes):
            seed = start_seed + offset
            env.reset(seed, perturbation_for_episode(offset))
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            images: dict[str, list[np.ndarray]] = {name: [] for name in camera_names}
            trace: list[dict[str, Any]] = []
            for index in range(180):
                before = env.observe(include_camera=True)
                frame = observation_frame(before, INSTRUCTION, camera_names)
                states.append(frame["state"])
                for camera_index, camera_name in enumerate(camera_names, start=1):
                    images[camera_name].append(frame[f"camera{camera_index}"])
                decision = teacher.predict(before, INSTRUCTION)
                after = env.step(decision.payload)
                # Label the command that actually entered MuJoCo, rather than
                # the post-physics qpos. This keeps the supervised target
                # aligned with the adapter's absolute-actuator contract even
                # when a high-level teacher action takes several solver ticks.
                actions.append(env.control_vector())
                trace.append({
                    "index": index + 1,
                    "action_type": decision.action_type,
                    "task_index": after["task"]["index"],
                })
                if after["task"]["completed"]:
                    break
            episode_path = output / f"episode_{seed}.npz"
            np.savez_compressed(
                episode_path,
                state=np.stack(states),
                action=np.stack(actions),
                **{f"image_{name}": np.stack(values) for name, values in images.items()},
            )
            rows.append({
                "seed": seed,
                "path": episode_path.name,
                "steps": len(states),
                "success": bool(after["task"]["completed"]),
                "perturbation": after["perturbation"],
                "trace": trace,
            })

    manifest = {
        "schema": "fortifiers.vla-demonstrations.v1",
        "instruction": INSTRUCTION,
        "camera_names": list(camera_names),
        "state_names": list(STATE_NAMES),
        "action_names": control_names,
        "control_ranges": control_ranges,
        "camera_shape": [120, 160, 3],
        "image_format": "HWC uint8 RGB",
        "action_format": "absolute MuJoCo actuator targets",
        "episodes": rows,
        "note": "Synthetic expert demonstrations from the verified MuJoCo baseline; not VLA evaluation evidence.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RGB/state/action VLA demonstrations")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=1001)
    parser.add_argument("--output", type=Path, default=Path("results/vla-demonstrations"))
    parser.add_argument("--camera", action="append", dest="cameras", default=None)
    args = parser.parse_args()
    cameras = tuple(args.cameras) if args.cameras else ("overview", "overhead", "left_oblique")
    print(json.dumps(collect(args.episodes, args.start_seed, args.output, cameras), indent=2))


if __name__ == "__main__":
    main()
