from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .env import FortifiersMuJoCoEnv, PerturbationConfig
from .policy import DinnerTablePolicy


INSTRUCTION = "Set the dinner table: open the drawer, retrieve the fork and spoon, place the plate, hand the cup from right to left, and place the cup."


def perturbation_for_episode(offset: int) -> PerturbationConfig:
    return PerturbationConfig(
        placement_jitter=0.04 + (offset % 4) * 0.025,
        mass_scale=1.0 + (offset % 3) * 0.08,
        friction_scale=0.90 + (offset % 5) * 0.07,
        object_scale=0.98 + (offset % 3) * 0.03,
        lighting_scale=0.80 + (offset % 4) * 0.12,
        background=("neutral", "warm", "cool")[offset % 3],
    )


def run_episode(
    env: FortifiersMuJoCoEnv,
    policy: DinnerTablePolicy,
    seed: int,
    perturbation: PerturbationConfig,
    max_actions: int = 180,
) -> dict[str, Any]:
    observation = env.reset(seed, perturbation)
    actions: list[dict[str, Any]] = []
    failures = 0
    motion_retries = 0
    for index in range(max_actions):
        decision = policy.predict(observation, INSTRUCTION)
        observation = env.step(decision.payload)
        result = dict(observation["task"]["last_action"])
        actions.append(
            {
                "index": index + 1,
                "policy_action": decision.action_type,
                "arm": decision.arm,
                "target": decision.target,
                "result": result,
                "task_index": observation["task"]["index"],
            }
        )
        if not result.get("ok", True):
            if decision.action_type == "move_to":
                motion_retries += 1
            else:
                failures += 1
        if observation["task"]["completed"]:
            break
    completed = bool(observation["task"]["completed"])
    return {
        "seed": seed,
        "perturbation": observation["perturbation"],
        "passed": completed,
        "completed": completed,
        "task_index": observation["task"]["index"],
        "next": observation["task"]["next"],
        "actions": len(actions),
        "failed_actions": failures,
        "motion_retries": motion_retries,
        "held_by": observation["task"]["held_by"],
        "placed": observation["task"]["placed"],
        "last_action": observation["task"]["last_action"],
        "trace": actions,
    }


def evaluate(
    episodes: int = 1,
    start_seed: int = 1001,
    learned_rules: frozenset[str] = frozenset(),
    max_actions: int = 180,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    policy = DinnerTablePolicy(learned_rules=learned_rules)
    rows: list[dict[str, Any]] = []
    with FortifiersMuJoCoEnv() as env:
        for offset in range(episodes):
            rows.append(
                run_episode(
                    env,
                    policy,
                    start_seed + offset,
                    perturbation_for_episode(offset),
                    max_actions=max_actions,
                )
            )
    return {
        "environment": "mujoco-dual-arm-dinner-table",
        "policy": policy.name,
        "policy_kind": "deterministic observation-driven baseline; not a VLA",
        "instruction": INSTRUCTION,
        "episodes": rows,
        "successes": sum(1 for row in rows if row["passed"]),
        "total": len(rows),
        "success_rate": sum(1 for row in rows if row["passed"]) / len(rows),
        "learned_rules": sorted(learned_rules),
        "note": "Success is computed from live MuJoCo task state after physical actuator stepping, measured two-jaw contact/lift gates, post-contact retention, release checks, and hand-off verification; it is not VLA or hardware evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the closed-loop MuJoCo dinner-table policy")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=1001)
    parser.add_argument("--max-actions", type=int, default=180)
    parser.add_argument("--rule", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.episodes > 1000:
        parser.error("--episodes must be between 1 and 1000")
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    report = evaluate(args.episodes, args.start_seed, frozenset(args.rule), args.max_actions)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
