from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .env import FortifiersMuJoCoEnv
from .evaluate_policy import perturbation_for_episode, run_episode
from .policy import DinnerTablePolicy


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_PATH = PACKAGE_ROOT / "results" / "dinner-skill-memory.json"
DEFAULT_OUTPUT_PATH = PACKAGE_ROOT / "results" / "dinner-apprenticeship.json"
HANDOFF_LESSON = "Bring the held cup to the shared hand-off zone before asking the left arm to receive it."


@dataclass
class PersistentSkillMemory:
    """Small JSON-backed memory contract for the physical task harness."""

    path: Path
    rules: set[str] = field(default_factory=set)
    version: str = "1.0"
    lessons: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "PersistentSkillMemory":
        if not path.is_file():
            return cls(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            rules=set(data.get("rules", [])),
            version=str(data.get("version", "1.0")),
            lessons=list(data.get("lessons", [])),
        )

    def clear(self) -> None:
        self.rules.clear()
        self.version = "1.0"
        self.lessons.clear()
        self.save()

    def teach(self, text: str) -> str:
        normalized = text.casefold()
        if "cup" not in normalized or "hand-off" not in normalized and "handoff" not in normalized:
            raise ValueError("This harness only compiles the explicit cup hand-off staging lesson")
        if not any(token in normalized for token in ("zone", "stage", "shared", "before")):
            raise ValueError("The hand-off lesson must name a shared staging step")
        rule_id = "handoff_staging"
        self.rules.add(rule_id)
        next_version = int(self.version.split(".")[-1]) + 1
        self.version = f"1.{next_version}"
        self.lessons.append({"text": text, "rule": rule_id, "version": self.version})
        self.save()
        return rule_id

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"version": self.version, "rules": sorted(self.rules), "lessons": self.lessons},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def summarize(row: dict[str, Any]) -> dict[str, Any]:
    milestones: list[dict[str, Any]] = []
    previous_task_index = 0
    for action in row["trace"]:
        if action["task_index"] != previous_task_index or action["policy_action"] in {"handoff", "release"} and action["result"].get("ok"):
            milestones.append(
                {
                    "action": action["policy_action"],
                    "target": action["target"],
                    "task_index": action["task_index"],
                    "message": action["result"].get("message"),
                }
            )
            previous_task_index = action["task_index"]
    return {
        "passed": row["passed"],
        "actions": row["actions"],
        "failed_actions": row["failed_actions"],
        "motion_retries": row["motion_retries"],
        "task_index": row["task_index"],
        "next": row["next"],
        "placed": row["placed"],
        "held_by": row["held_by"],
        "milestones": milestones,
    }


def run_demo(
    memory_path: Path = DEFAULT_MEMORY_PATH,
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
    seed: int = 1001,
    fresh: bool = True,
) -> dict[str, Any]:
    memory = PersistentSkillMemory.load(memory_path)
    if fresh:
        memory.clear()
    with FortifiersMuJoCoEnv() as env:
        baseline = run_episode(
            env,
            DinnerTablePolicy(
                learned_rules=frozenset(memory.rules),
                require_handoff_skill=True,
            ),
            seed,
            perturbation_for_episode(0),
            max_actions=160,
        )
        compiled_rule = memory.teach(HANDOFF_LESSON)
        retry = run_episode(
            env,
            DinnerTablePolicy(
                learned_rules=frozenset(memory.rules),
                require_handoff_skill=True,
            ),
            seed,
            perturbation_for_episode(0),
            max_actions=180,
        )
        reset_and_remember = run_episode(
            env,
            DinnerTablePolicy(
                learned_rules=frozenset(memory.rules),
                require_handoff_skill=True,
            ),
            seed + 1,
            perturbation_for_episode(1),
            max_actions=180,
        )
    report = {
        "environment": "mujoco-dual-arm-dinner-table",
        "policy": "closed-loop-dinner-policy-v1",
        "policy_kind": "deterministic observation-driven baseline; not a VLA",
        "lesson": HANDOFF_LESSON,
        "compiled_rule": compiled_rule,
        "memory": {
            "path": str(memory_path),
            "version": memory.version,
            "rules": sorted(memory.rules),
            "lessons": memory.lessons,
        },
        "baseline_before_teaching": summarize(baseline),
        "retry_after_teaching": summarize(retry),
        "reset_and_remember": summarize(reset_and_remember),
        "claim_boundary": "This proves persistence of a bounded execution rule in the MuJoCo policy harness. It is not evidence of neural VLA learning, LeRobot training, hardware control, or OpenVINO acceleration.",
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MuJoCo teach/retry/reset/remember demonstration")
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--preserve-memory", action="store_true")
    args = parser.parse_args()
    report = run_demo(args.memory, args.output, args.seed, fresh=not args.preserve_memory)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
