from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskStep:
    id: str
    verb: str
    object_name: str | None
    arm: str
    destination: str | None = None
    complementary_with: str | None = None


DINNER_TABLE_STEPS = (
    TaskStep("open_drawer", "open_drawer", None, "right"),
    TaskStep("retrieve_fork", "pick", "fork", "left", "table_surface"),
    TaskStep("retrieve_spoon", "pick", "spoon", "right", "table_surface"),
    TaskStep("place_plate", "pick_and_place", "plate", "left", "place_area"),
    TaskStep("handoff_cup", "handoff", "cup", "right", "left", "place_cup"),
    TaskStep("place_cup", "place", "cup", "left", "place_area", "handoff_cup"),
)


def build_dinner_table_plan() -> list[TaskStep]:
    """Return the deterministic task skeleton a VLA policy must ground."""

    return list(DINNER_TABLE_STEPS)


def validate_dinner_table_plan(steps: list[TaskStep]) -> None:
    if {step.arm for step in steps} != {"left", "right"}:
        raise ValueError("dinner-table plan must use both arms")
    if not any(step.verb == "handoff" for step in steps):
        raise ValueError("dinner-table plan must include a bimanual hand-off")
    seen = {step.id for step in steps}
    for step in steps:
        if step.complementary_with and step.complementary_with not in seen:
            raise ValueError(f"missing complementary step for {step.id}")


def plan_as_dicts(steps: list[TaskStep] | None = None) -> list[dict[str, object]]:
    selected = steps or build_dinner_table_plan()
    validate_dinner_table_plan(selected)
    return [asdict(step) for step in selected]
