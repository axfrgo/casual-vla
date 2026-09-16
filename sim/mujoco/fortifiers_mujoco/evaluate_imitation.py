"""Evaluate a trained camera/state/instruction policy through native MuJoCo."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .env import FortifiersMuJoCoEnv
from .evaluate_policy import INSTRUCTION, perturbation_for_episode
from .imitation_policy import ImitationPolicyAdapter


def evaluate_imitation(
    checkpoint: str,
    episodes: int = 10,
    start_seed: int = 1001,
    device: str = "cpu",
    max_actions: int = 240,
    width: int = 160,
    height: int = 120,
    patience_actions: int = 24,
) -> dict[str, Any]:
    if episodes < 1 or max_actions < 1 or patience_actions < 1:
        raise ValueError("episodes, max_actions, and patience_actions must be positive")

    rows: list[dict[str, Any]] = []
    with FortifiersMuJoCoEnv(width=width, height=height) as env:
        policy = ImitationPolicyAdapter(checkpoint, env, device=device)
        for offset in range(episodes):
            seed = start_seed + offset
            observation = env.reset(seed, perturbation_for_episode(offset))
            policy.reset()
            trace: list[dict[str, Any]] = []
            inference_ms: list[float] = []
            failures = 0
            stalled_actions = 0
            stop_reason = "max_actions"
            for index in range(max_actions):
                observation = env.observe(include_camera=True)
                started = time.perf_counter_ns()
                decision = policy.predict(observation, INSTRUCTION)
                inference_ms.append((time.perf_counter_ns() - started) / 1_000_000)
                observation = env.step_vla(decision.payload)
                result = dict(observation["task"]["last_action"])
                if not result.get("ok", True):
                    failures += 1
                trace.append(
                    {
                        "index": index + 1,
                        "action_type": decision.action_type,
                        "result": result,
                        "task_index": observation["task"]["index"],
                    }
                )
                if observation["task"]["completed"]:
                    stop_reason = "completed"
                    break
                if observation["task"]["index"] == (trace[-2]["task_index"] if len(trace) > 1 else 0):
                    stalled_actions += 1
                else:
                    stalled_actions = 0
                if stalled_actions >= patience_actions:
                    stop_reason = "no_task_progress"
                    break
            completed = bool(observation["task"]["completed"])
            sorted_inference = sorted(inference_ms)
            rows.append(
                {
                    "seed": seed,
                    "perturbation": observation["perturbation"],
                    "passed": completed,
                    "completed": completed,
                    "task_index": observation["task"]["index"],
                    "next": observation["task"]["next"],
                    "actions": len(trace),
                    "failed_actions": failures,
                    "stop_reason": stop_reason,
                    "inference_ms": {
                        "mean": sum(inference_ms) / len(inference_ms) if inference_ms else None,
                        "p95": sorted_inference[min(len(sorted_inference) - 1, int(len(sorted_inference) * 0.95))]
                        if sorted_inference
                        else None,
                    },
                    "trace": trace,
                }
            )

    return {
        "report_schema": "fortifiers.mujoco.imitation-evaluation.v1",
        "environment": "mujoco-dual-arm-dinner-table",
        "policy": "camera-state-instruction-imitation",
        "policy_kind": "camera/state/language-conditioned neural behavior-cloning policy",
        "checkpoint": checkpoint,
        "device": device,
        "camera_views": list(policy.camera_names),
        "instruction": INSTRUCTION,
        "episodes": rows,
        "successes": sum(1 for row in rows if row["passed"]),
        "total": len(rows),
        "success_rate": sum(1 for row in rows if row["passed"]) / len(rows),
        "evidence_boundary": "Success is computed from live MuJoCo actuator stepping and measured task gates for this real neural checkpoint; it is not browser animation, a deterministic baseline, or hardware evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained imitation checkpoint in MuJoCo")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1001)
    parser.add_argument("--max-actions", type=int, default=240)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--patience-actions", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_imitation(
        str(args.checkpoint), args.episodes, args.start_seed, args.device, args.max_actions, args.width, args.height, args.patience_actions
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
