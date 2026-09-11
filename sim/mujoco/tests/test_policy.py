from fortifiers_mujoco.apprenticeship import run_demo
from fortifiers_mujoco.evaluate_policy import evaluate
from pathlib import Path

from fortifiers_mujoco.bridge_server import respond
from fortifiers_mujoco.env import FortifiersMuJoCoEnv


def test_closed_loop_policy_completes_dinner_task() -> None:
    report = evaluate(episodes=1, start_seed=1001, max_actions=180)
    episode = report["episodes"][0]
    assert report["successes"] == 1
    assert episode["passed"] is True
    assert episode["task_index"] == 6
    assert episode["placed"] == ["cup", "fork", "plate", "spoon"]
    handoffs = [action for action in episode["trace"] if action["policy_action"] == "handoff"]
    assert handoffs
    assert handoffs[-1]["result"]["message"] == "bimanual hand-off verified"
    assert handoffs[-1]["result"]["contact"]["jaw_count"] == 2
    assert handoffs[-1]["result"]["retention"]["active"] is True


def test_teach_retry_reset_remember_persists_rule() -> None:
    test_root = Path("results")
    report = run_demo(
        memory_path=test_root / "test-skill-memory.json",
        output_path=test_root / "test-apprenticeship.json",
        seed=1001,
    )
    assert report["baseline_before_teaching"]["passed"] is False
    assert report["compiled_rule"] == "handoff_staging"
    assert report["retry_after_teaching"]["passed"] is True
    assert report["reset_and_remember"]["passed"] is True
    assert report["memory"]["rules"] == ["handoff_staging"]


def test_jsonl_bridge_can_run_the_same_policy() -> None:
    with FortifiersMuJoCoEnv() as env:
        respond({"command": "reset", "seed": 1001}, env)
        result = respond({"command": "policy_run", "max_actions": 180}, env)
    assert result["policy"] == "closed-loop-dinner-policy-v1"
    assert result["observation"]["task"]["completed"] is True
