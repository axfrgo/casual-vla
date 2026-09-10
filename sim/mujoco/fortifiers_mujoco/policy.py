from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PolicyAction:
    """Normalized policy output before safety validation and execution."""

    action_type: str
    arm: str | None
    target: str | None
    payload: dict[str, Any]


class VLAInference(Protocol):
    """Replaceable interface for a real VLA or imitation policy."""

    name: str

    def predict(self, observation: dict[str, Any], instruction: str) -> PolicyAction:
        ...


class UnconfiguredVLA:
    """Fail-closed placeholder; it cannot generate fake robot actions."""

    name = "unconfigured-vla"

    def predict(self, observation: dict[str, Any], instruction: str) -> PolicyAction:
        del observation, instruction
        raise RuntimeError(
            "No VLA policy is configured. Provide a real model adapter before executing robot actions."
        )
