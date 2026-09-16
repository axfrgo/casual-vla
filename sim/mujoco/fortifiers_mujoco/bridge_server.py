from __future__ import annotations

import json
import argparse
import sys
from typing import Any

from .env import FortifiersMuJoCoEnv, PerturbationConfig
from .evaluate_policy import INSTRUCTION
from .policy import DinnerTablePolicy, VLAInference
from .task import plan_as_dicts
from .imitation_policy import ImitationPolicyAdapter
from .vla_adapter import SmolVLAAdapter, VLAConfig


def respond(
    payload: dict[str, Any],
    env: FortifiersMuJoCoEnv,
    policy: VLAInference | None = None,
) -> dict[str, Any]:
    command = payload.get("command")
    if command == "reset":
        seed = payload.get("seed", 0)
        raw_perturbation = payload.get("perturbation")
        perturbation = (
            PerturbationConfig(**raw_perturbation)
            if isinstance(raw_perturbation, dict)
            else None
        )
        observation = env.reset(int(seed), perturbation)
        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        return {"ok": True, "observation": observation}
    if command == "observe":
        return {"ok": True, "observation": env.observe(bool(payload.get("camera", False)))}
    if command == "step":
        action = payload.get("action")
        if not isinstance(action, dict):
            raise ValueError("step requires an action object")
        return {"ok": True, "observation": env.step(action)}
    if command in {"policy_step", "policy_run"}:
        selected_policy = policy or DinnerTablePolicy()
        instruction = str(payload.get("instruction", INSTRUCTION))
        limit = int(payload.get("max_actions", 180)) if command == "policy_run" else 1
        trace: list[dict[str, Any]] = []
        uses_camera = isinstance(selected_policy, (SmolVLAAdapter, ImitationPolicyAdapter))
        observation = env.observe(include_camera=uses_camera)
        for _ in range(limit):
            decision = selected_policy.predict(observation, instruction)
            observation = (
                env.step_vla(decision.payload)
                if decision.action_type in {"vla_joint_targets", "imitation_joint_targets"}
                else env.step(decision.payload)
            )
            trace.append(
                {
                    "action_type": decision.action_type,
                    "arm": decision.arm,
                    "target": decision.target,
                    "result": observation["task"]["last_action"],
                    "task_index": observation["task"]["index"],
                }
            )
            if command == "policy_step" or observation["task"]["completed"]:
                break
            observation = env.observe(include_camera=uses_camera)
        return {
            "ok": True,
            "policy": selected_policy.name,
            "instruction": instruction,
            "trace": trace,
            "observation": observation,
        }
    if command == "task_plan":
        return {"ok": True, "steps": plan_as_dicts()}
    if command == "close":
        return {"ok": True, "closed": True}
    raise ValueError(f"unsupported bridge command: {command!r}")


def main() -> None:
    """Run a newline-delimited JSON bridge; never evaluates input as code."""

    parser = argparse.ArgumentParser(description="Run the MuJoCo task bridge")
    parser.add_argument("--policy", choices=("baseline", "smolvla", "imitation"), default="baseline")
    parser.add_argument("--checkpoint", help="SmolVLA model id/path or local imitation checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera", action="append", dest="cameras", default=None)
    args = parser.parse_args()

    with FortifiersMuJoCoEnv() as env:
        if args.policy == "smolvla":
            if not args.checkpoint:
                parser.error("--checkpoint is required with --policy smolvla")
            cameras = tuple(args.cameras) if args.cameras else ("overview", "overhead", "left_oblique")
            policy: VLAInference = SmolVLAAdapter(
                VLAConfig(checkpoint=args.checkpoint, device=args.device, camera_names=cameras),
                env,
            )
        elif args.policy == "imitation":
            if not args.checkpoint:
                parser.error("--checkpoint is required with --policy imitation")
            policy = ImitationPolicyAdapter(args.checkpoint, env, device=args.device)
        else:
            policy = DinnerTablePolicy()
        for line in sys.stdin:
            if not line.strip():
                continue
            payload: dict[str, Any] = {}
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                result = respond(payload, env, policy)
            except Exception as exc:  # keep the process alive for the caller
                result = {"ok": False, "error": str(exc)}
            sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            if payload.get("command") == "close":
                break


if __name__ == "__main__":
    main()
