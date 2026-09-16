"""Evaluate neural task intent gated by the verified dinner-table controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .env import FortifiersMuJoCoEnv
from .evaluate_policy import INSTRUCTION, perturbation_for_episode
from .neural_guided_policy import NeuralIntentAdapter
from .policy import DinnerTablePolicy


def evaluate_neural_guided(
    checkpoint: str,
    episodes: int = 10,
    start_seed: int = 1001,
    device: str = "cpu",
    max_actions: int = 180,
    confidence_threshold: float = 0.5,
    width: int = 160,
    height: int = 120,
) -> dict[str, Any]:
    if episodes < 1 or max_actions < 1 or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("episodes and max_actions must be positive; confidence threshold must be in [0, 1]")

    rows: list[dict[str, Any]] = []
    with FortifiersMuJoCoEnv(width=width, height=height) as env:
        neural = NeuralIntentAdapter(checkpoint, device=device)
        controller = DinnerTablePolicy()
        for offset in range(episodes):
            seed = start_seed + offset
            env.reset(seed, perturbation_for_episode(offset))
            trace: list[dict[str, Any]] = []
            inference_ms: list[float] = []
            intent_rejections = 0
            stop_reason = "max_actions"
            for index in range(max_actions):
                before = env.observe(include_camera=True)
                prediction, elapsed = neural.timed_predict(before, INSTRUCTION)
                inference_ms.append(elapsed)
                expected_phase = int(before["task"]["index"])
                gate_ok = prediction["phase_index"] == expected_phase and prediction["confidence"] >= confidence_threshold
                if not gate_ok:
                    intent_rejections += 1
                    stop_reason = "neural_intent_rejected"
                    trace.append(
                        {
                            "index": index + 1,
                            "action_type": "neural_intent_gate",
                            "task_index": expected_phase,
                            "intent": prediction,
                            "result": {
                                "ok": False,
                                "action": "neural_intent_gate",
                                "expected_step": neural.task_step_ids[expected_phase],
                                "predicted_step": prediction["step_id"],
                                "message": "neural task intent disagreed with live task state",
                            },
                        }
                    )
                    break

                decision = controller.predict(before, INSTRUCTION)
                after = env.step(decision.payload)
                result = dict(after["task"]["last_action"])
                trace.append(
                    {
                        "index": index + 1,
                        "action_type": decision.action_type,
                        "task_index": expected_phase,
                        "intent": prediction,
                        "result": result,
                    }
                )
                if after["task"]["completed"]:
                    stop_reason = "completed"
                    break

            completed = bool(env.observe()["task"]["completed"])
            sorted_inference = sorted(inference_ms)
            rows.append(
                {
                    "seed": seed,
                    "perturbation": env.observe()["perturbation"],
                    "passed": completed,
                    "completed": completed,
                    "task_index": env.observe()["task"]["index"],
                    "next": env.observe()["task"]["next"],
                    "actions": len(trace),
                    "failed_actions": sum(1 for item in trace if not item["result"].get("ok", True)),
                    "intent_rejections": intent_rejections,
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
        "report_schema": "fortifiers.mujoco.neural-guided-evaluation.v1",
        "environment": "mujoco-dual-arm-dinner-table",
        "policy": neural.name,
        "policy_kind": "camera/state/language neural task-intent gate plus measured closed-loop controller",
        "neural_checkpoint": checkpoint,
        "controller": controller.name,
        "device": device,
        "camera_views": list(neural.camera_names),
        "instruction": INSTRUCTION,
        "confidence_threshold": confidence_threshold,
        "episodes": rows,
        "successes": sum(1 for row in rows if row["passed"]),
        "total": len(rows),
        "success_rate": sum(1 for row in rows if row["passed"]) / len(rows),
        "intent_rejections": sum(row["intent_rejections"] for row in rows),
        "evidence_boundary": "This completion score includes a real neural task-intent gate on every decision and a separately named measured closed-loop controller for low-level execution; it is not an end-to-end neural joint-target score, browser animation, or hardware evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate neural task intent with verified control")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1001)
    parser.add_argument("--max-actions", type=int, default=180)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_neural_guided(
        str(args.checkpoint),
        args.episodes,
        args.start_seed,
        args.device,
        args.max_actions,
        args.confidence_threshold,
        args.width,
        args.height,
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
