"""Evaluate a real camera-language policy through the MuJoCo task gates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .env import FortifiersMuJoCoEnv
from .evaluate_policy import INSTRUCTION, perturbation_for_episode
from .vla_adapter import SmolVLAAdapter, VLAConfig


def evaluate_vla(
    checkpoint: str,
    episodes: int = 1,
    start_seed: int = 1001,
    device: str = "cpu",
    camera_names: tuple[str, ...] = ("overview", "overhead", "left_oblique"),
    max_actions: int = 240,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if max_actions < 1:
        raise ValueError("max_actions must be positive")

    rows: list[dict[str, Any]] = []
    with FortifiersMuJoCoEnv(width=320, height=240) as env:
        policy = SmolVLAAdapter(
            VLAConfig(checkpoint=checkpoint, device=device, camera_names=camera_names),
            env,
        )
        for offset in range(episodes):
            seed = start_seed + offset
            env.reset(seed, perturbation_for_episode(offset))
            policy.reset()
            trace: list[dict[str, Any]] = []
            inference_ms: list[float] = []
            for index in range(max_actions):
                observation = env.observe(include_camera=True)
                started = time.perf_counter_ns()
                decision = policy.predict(observation, INSTRUCTION)
                inference_ms.append((time.perf_counter_ns() - started) / 1_000_000)
                observation = env.step_vla(decision.payload)
                trace.append(
                    {
                        "index": index + 1,
                        "action_type": decision.action_type,
                        "result": observation["task"]["last_action"],
                        "task_index": observation["task"]["index"],
                    }
                )
                if observation["task"]["completed"]:
                    break
            completed = bool(observation["task"]["completed"])
            rows.append(
                {
                    "seed": seed,
                    "perturbation": observation["perturbation"],
                    "passed": completed,
                    "completed": completed,
                    "task_index": observation["task"]["index"],
                    "next": observation["task"]["next"],
                    "actions": len(trace),
                    "inference_ms": {
                        "mean": sum(inference_ms) / len(inference_ms) if inference_ms else None,
                        "p95": sorted(inference_ms)[min(len(inference_ms) - 1, int(len(inference_ms) * 0.95))]
                        if inference_ms
                        else None,
                    },
                    "trace": trace,
                }
            )

    return {
        "report_schema": "fortifiers.mujoco.vla-evaluation.v1",
        "environment": "mujoco-dual-arm-dinner-table",
        "policy": "smolvla-camera-language",
        "policy_kind": "camera-language-conditioned neural policy",
        "checkpoint": checkpoint,
        "device": device,
        "camera_views": list(camera_names),
        "instruction": INSTRUCTION,
        "episodes": rows,
        "successes": sum(1 for row in rows if row["passed"]),
        "total": len(rows),
        "success_rate": sum(1 for row in rows if row["passed"]) / len(rows),
        "evidence_boundary": "Neural policy result only when the checkpoint and device are real; not a browser animation or deterministic baseline result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a real SmolVLA checkpoint in MuJoCo")
    parser.add_argument("checkpoint", help="Local checkpoint path or Hugging Face model id")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=1001)
    parser.add_argument("--max-actions", type=int, default=240)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera", action="append", dest="cameras", default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cameras = tuple(args.cameras) if args.cameras else ("overview", "overhead", "left_oblique")
    report = evaluate_vla(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        start_seed=args.start_seed,
        device=args.device,
        camera_names=cameras,
        max_actions=args.max_actions,
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
