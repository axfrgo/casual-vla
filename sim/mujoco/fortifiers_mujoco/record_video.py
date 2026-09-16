"""Record a reproducible native MuJoCo dinner-table demonstration video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from .evaluate_policy import INSTRUCTION, perturbation_for_episode
from .policy import DinnerTablePolicy
from .env import FortifiersMuJoCoEnv


def _frame(image: np.ndarray, title: str, detail: str) -> np.ndarray:
    """Add a restrained evidence overlay without changing the scene pixels."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional video tooling
        raise RuntimeError("Video recording requires opencv-python in the video extra") from exc

    canvas = np.ascontiguousarray(image.copy())
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (7, 16, 22), thickness=-1)
    cv2.putText(canvas, title, (14, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (141, 240, 201), 1, cv2.LINE_AA)
    cv2.putText(canvas, detail, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (197, 210, 217), 1, cv2.LINE_AA)
    return canvas


def record(
    output: Path,
    episodes: int = 10,
    start_seed: int = 1001,
    fps: int = 8,
    width: int = 320,
    height: int = 240,
    hold_frames: int = 2,
) -> dict[str, Any]:
    if episodes < 1 or fps < 1 or width < 64 or height < 64 or hold_frames < 1:
        raise ValueError("episodes, fps, dimensions, and hold_frames must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    policy = DinnerTablePolicy()
    rows: list[dict[str, Any]] = []
    with FortifiersMuJoCoEnv(width=width, height=height) as env, imageio.get_writer(
        str(output), fps=fps, codec="libx264", quality=7, macro_block_size=1
    ) as writer:
        for offset in range(episodes):
            seed = start_seed + offset
            observation = env.reset(seed, perturbation_for_episode(offset))
            episode_frames = 0
            action_count = 0
            failed_actions = 0
            while not observation["task"]["completed"]:
                decision = policy.predict(observation, INSTRUCTION)
                observation = env.step(decision.payload)
                result = dict(observation["task"]["last_action"])
                action_count += 1
                failed_actions += int(not result.get("ok", True))
                title = f"FORTIFIERS / NATIVE MUJOCO / SEED {seed}"
                detail = (
                    f"STEP {observation['task']['index']}/6  |  ACTION {action_count}  |  "
                    f"NEXT {observation['task']['next']}  |  {'PASS' if observation['task']['completed'] else 'LIVE'}"
                )
                rendered = _frame(env.render("overview"), title, detail)
                for _ in range(hold_frames):
                    writer.append_data(rendered)
                    episode_frames += 1
                if action_count >= 180:
                    raise RuntimeError(f"seed {seed} exceeded the 180-action recording bound")
            rows.append(
                {
                    "seed": seed,
                    "passed": bool(observation["task"]["completed"]),
                    "actions": action_count,
                    "failed_actions": failed_actions,
                    "frames": episode_frames,
                    "perturbation": observation["perturbation"],
                }
            )
    report = {
        "report_schema": "fortifiers.mujoco.video-evidence.v1",
        "video": str(output),
        "video_format": "H.264 MP4",
        "environment": "mujoco-dual-arm-dinner-table",
        "controller": "closed-loop-dinner-policy-v1",
        "instruction": INSTRUCTION,
        "camera": "overview",
        "resolution": [width, height],
        "fps": fps,
        "episodes": rows,
        "successes": sum(1 for row in rows if row["passed"]),
        "total": len(rows),
        "success_rate": sum(1 for row in rows if row["passed"]) / len(rows),
        "evidence_boundary": "Native MuJoCo visualization of the deterministic baseline; this video is not browser physics, neural-policy evidence, or a hardware-robot recording.",
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a native MuJoCo dinner-table evidence video")
    parser.add_argument("--output", type=Path, default=Path("public/evidence/dinner-table-10-seed-demo.mp4"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1001)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--hold-frames", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(record(args.output, args.episodes, args.start_seed, args.fps, args.width, args.height, args.hold_frames), indent=2))


if __name__ == "__main__":
    main()
