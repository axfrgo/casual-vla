from __future__ import annotations

import json
import sys
from typing import Any

from .env import FortifiersMuJoCoEnv, PerturbationConfig
from .task import plan_as_dicts


def respond(payload: dict[str, Any], env: FortifiersMuJoCoEnv) -> dict[str, Any]:
    command = payload.get("command")
    if command == "reset":
        seed = payload.get("seed", 0)
        raw_perturbation = payload.get("perturbation")
        perturbation = (
            PerturbationConfig(**raw_perturbation)
            if isinstance(raw_perturbation, dict)
            else None
        )
        return {"ok": True, "observation": env.reset(int(seed), perturbation)}
    if command == "observe":
        return {"ok": True, "observation": env.observe(bool(payload.get("camera", False)))}
    if command == "step":
        action = payload.get("action")
        if not isinstance(action, dict):
            raise ValueError("step requires an action object")
        return {"ok": True, "observation": env.step(action)}
    if command == "task_plan":
        return {"ok": True, "steps": plan_as_dicts()}
    if command == "close":
        return {"ok": True, "closed": True}
    raise ValueError(f"unsupported bridge command: {command!r}")


def main() -> None:
    """Run a newline-delimited JSON bridge; never evaluates input as code."""

    with FortifiersMuJoCoEnv() as env:
        for line in sys.stdin:
            if not line.strip():
                continue
            payload: dict[str, Any] = {}
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                result = respond(payload, env)
            except Exception as exc:  # keep the process alive for the caller
                result = {"ok": False, "error": str(exc)}
            sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            if payload.get("command") == "close":
                break


if __name__ == "__main__":
    main()
