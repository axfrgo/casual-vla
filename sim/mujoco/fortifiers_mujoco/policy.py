from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .env import GRASP_PINCH_OFFSETS, GRASP_TARGET_OFFSETS, PLACEMENT_TARGETS


# Measured overlap corridor for the current official SO-101 bases. This point
# keeps the cup above the tabletop while remaining inside both arm workspaces
# (the old center point was reachable by neither arm with a stable margin).
HANDOFF_POINT = np.array([0.07, -0.50, 0.96])
TRANSPORT_CLEARANCE = 0.08


@dataclass(frozen=True)
class PolicyAction:
    """Normalized policy output before safety validation and execution."""

    action_type: str
    arm: str | None
    target: str | None
    payload: dict[str, Any]


def _distance(first: Any, second: Any) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)))


@dataclass
class DinnerTablePolicy:
    """A deterministic observation-driven controller for the dinner task.

    This is intentionally not called a VLA: it is a transparent closed-loop
    baseline that consumes the same structured observations a future camera
    VLA adapter will consume. Its value here is executable task grounding,
    physical verification, and a stable benchmark target.
    """

    learned_rules: frozenset[str] = frozenset()
    name: str = "closed-loop-dinner-policy-v1"
    reach_tolerance: float = 0.018
    require_handoff_skill: bool = False
    _episode_seed: int | None = field(default=None, init=False, repr=False)
    _placement_stage: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _placement_destinations: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def predict(self, observation: dict[str, Any], instruction: str) -> PolicyAction:
        del instruction
        seed = observation.get("seed")
        if seed != self._episode_seed:
            self._episode_seed = seed
            self._placement_stage.clear()
            self._placement_destinations.clear()
        task = observation["task"]
        step_id = task["next"]
        if task["completed"]:
            return PolicyAction("noop", None, None, {"type": "noop"})
        if step_id == "open_drawer":
            return PolicyAction("open_drawer", "right", None, {"type": "open_drawer"})

        if step_id in {"retrieve_fork", "retrieve_spoon", "place_plate", "place_cup"}:
            object_name = {
                "retrieve_fork": "fork",
                "retrieve_spoon": "spoon",
                "place_plate": "plate",
                "place_cup": "cup",
            }[step_id]
            arm = {
                "retrieve_fork": "left",
                "retrieve_spoon": "right",
                "place_plate": "left",
                "place_cup": "left",
            }[step_id]
            return self._pick_place_action(observation, arm, object_name)

        if step_id == "handoff_cup":
            cup = observation["objects"]["cup"]
            held_by = cup["held_by"]
            if held_by is None:
                return self._move_or_grasp(observation, "right", "cup")
            if held_by == "right":
                if self.require_handoff_skill and "handoff_staging" not in self.learned_rules:
                    # Deliberately expose the pre-teaching failure used by the
                    # apprenticeship demo: the receiving arm is asked to
                    # reach for a cup that is still across the table.
                    if _distance(observation["arms"]["left"]["pinch_point"], cup["grasp_target"]) > self.reach_tolerance:
                        return PolicyAction(
                            "move_to",
                            "left",
                            "cup",
                            {
                                "type": "move_to",
                                "arm": "left",
                                "target": cup["grasp_target"],
                                "orientation": "auto",
                                "frame": "pinch",
                                "approach": True,
                                "gripper": "open",
                            },
                        )
                    return PolicyAction(
                        "handoff",
                        "left",
                        "cup",
                        {"type": "handoff", "object": "cup", "from": "right", "to": "left"},
                    )
                if _distance(observation["arms"]["right"]["pinch_point"], HANDOFF_POINT) > self.reach_tolerance:
                    return PolicyAction(
                        "move_to",
                        "right",
                        "cup",
                        {
                            "type": "move_to",
                            "arm": "right",
                            "target": HANDOFF_POINT.tolist(),
                            "orientation": observation["arms"]["right"]["pinch_orientation"],
                            "frame": "pinch",
                            "gripper": "closed",
                        },
                    )
                if _distance(observation["arms"]["left"]["pinch_point"], cup["grasp_target"]) > self.reach_tolerance:
                    return PolicyAction(
                        "move_to",
                        "left",
                        "cup",
                        {
                            "type": "move_to",
                            "arm": "left",
                            "target": cup["grasp_target"],
                            "orientation": "auto",
                            "frame": "pinch",
                            "approach": True,
                            "gripper": "open",
                        },
                    )
                return PolicyAction(
                    "handoff",
                    "left",
                    "cup",
                    {"type": "handoff", "object": "cup", "from": "right", "to": "left"},
                )

        return PolicyAction("noop", None, None, {"type": "noop"})

    def _move_or_grasp(self, observation: dict[str, Any], arm: str, object_name: str) -> PolicyAction:
        object_state = observation["objects"][object_name]
        target = np.asarray(
            object_state.get("grasp_target_by_arm", {}).get(arm, object_state["grasp_target"]),
            dtype=np.float64,
        )
        if _distance(observation["arms"][arm]["pinch_point"], target) > self.reach_tolerance:
            return PolicyAction(
                "move_to",
                arm,
                object_name,
                {
                    "type": "move_to",
                    "arm": arm,
                    "target": target.tolist(),
                    "orientation": "auto",
                    "frame": "pinch",
                    "approach": True,
                    "gripper": "open",
                },
            )
        return PolicyAction(
            "grasp",
            arm,
            object_name,
            {"type": "grasp", "arm": arm, "object": object_name},
        )

    def _pick_place_action(self, observation: dict[str, Any], arm: str, object_name: str) -> PolicyAction:
        object_state = observation["objects"][object_name]
        if object_state["held_by"] != arm:
            return self._move_or_grasp(observation, arm, object_name)
        object_position = np.asarray(object_state["position"], dtype=np.float64)
        if object_name not in self._placement_destinations:
            if object_name in {"fork", "spoon"}:
                # Both utensils only require a verified tabletop placement.
                # Keep their observed XY fixed for the whole placement phase;
                # recomputing it after every contact step would chase slip.
                destination = np.array(
                    [object_position[0], object_position[1], PLACEMENT_TARGETS[object_name][2] + 0.02],
                    dtype=np.float64,
                )
            else:
                destination = PLACEMENT_TARGETS[object_name].copy()
            self._placement_destinations[object_name] = (
                destination + GRASP_TARGET_OFFSETS[object_name] + GRASP_PINCH_OFFSETS[arm]
            )
        destination = self._placement_destinations[object_name]
        # The SO-101 wrist has a stable two-pad lift envelope of roughly
        # 5–6 cm above the tabletop. Keep transport just above that envelope
        # rather than demanding an unnecessarily high 14 cm swing.
        if object_name in {"fork", "spoon"}:
            transport = np.array(
                [destination[0], destination[1], PLACEMENT_TARGETS[object_name][2] + TRANSPORT_CLEARANCE],
                dtype=np.float64,
            )
        else:
            transport = destination + np.array([0.0, 0.0, TRANSPORT_CLEARANCE])
        pinch_point = observation["arms"][arm]["pinch_point"]
        # Utensils are intentionally scored as a safe tabletop drop anywhere
        # on the surface; plate and cup keep their explicit place-zone gate.
        placement_xy = destination[:2] if object_name in {"fork", "spoon"} else PLACEMENT_TARGETS[object_name][:2]
        xy_error = float(np.linalg.norm(object_position[:2] - placement_xy))
        # The rigid retention pose keeps the plate's rim a few centimeters
        # above its center target; release once its footprint is safely over
        # the zone rather than demanding an impossible center-height match.
        surface_clearance = 0.10 if object_name in {"plate", "cup"} else 0.045
        at_transport = _distance(pinch_point, transport.tolist()) <= self.reach_tolerance
        stage = self._placement_stage.get(object_name, "transport")
        if stage == "transport" and not at_transport:
            target = transport.tolist()
        else:
            self._placement_stage[object_name] = "lower"
            # Once horizontally over the destination, keep descending until
            # the object itself reaches the tabletop. This avoids retracting
            # back to the transport waypoint when contact briefly limits the
            # final approach.
            xy_gate = 0.08 if object_name in {"fork", "spoon"} else 0.18
            if xy_error < xy_gate and object_position[2] <= PLACEMENT_TARGETS[object_name][2] + surface_clearance:
                self._placement_stage.pop(object_name, None)
                return PolicyAction(
                    "release",
                    arm,
                    object_name,
                    {"type": "release", "arm": arm, "object": object_name},
                )
            target = destination.tolist()
        if _distance(pinch_point, target) > self.reach_tolerance:
            transport_orientation: str | list[float] = (
                "auto"
            )
            return PolicyAction(
                "move_to",
                arm,
                object_name,
                {
                    "type": "move_to",
                    "arm": arm,
                    "target": target,
                    "orientation": transport_orientation,
                    "frame": "pinch",
                    "gripper": "closed",
                },
            )
        # The tool point can be within tolerance while the object is still
        # settling above the surface. Re-issue the same closed, orientation-
        # preserving target so the environment can complete its contact-aware
        # placement instead of crashing the evaluation loop.
        return PolicyAction(
            "move_to",
            arm,
            object_name,
            {
                "type": "move_to",
                "arm": arm,
                "target": target,
                "orientation": "auto",
                "frame": "pinch",
                "gripper": "closed",
            },
        )


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
