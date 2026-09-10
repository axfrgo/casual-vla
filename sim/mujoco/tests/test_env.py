from pathlib import Path

import numpy as np
import mujoco

from fortifiers_mujoco import (
    OFFICIAL_SO101_MJCF_PATH,
    FortifiersMuJoCoEnv,
    PerturbationConfig,
)


def test_model_path_is_repo_local() -> None:
    with FortifiersMuJoCoEnv() as env:
        assert Path(env.model_path).name == "official_dual_so101_dinner_table.xml"
        assert env.model.nu == 13


def test_official_so101_mjcf_asset_is_available() -> None:
    assert OFFICIAL_SO101_MJCF_PATH.is_file()
    model = mujoco.MjModel.from_xml_path(str(OFFICIAL_SO101_MJCF_PATH))
    assert model.nu == 6
    assert model.ngeom > 0


def test_reset_is_deterministic_and_perturbations_do_not_accumulate() -> None:
    perturbation = PerturbationConfig(
        placement_jitter=0.15,
        mass_scale=1.2,
        friction_scale=1.3,
        object_scale=1.1,
        lighting_scale=0.8,
        background="cool",
    )
    with FortifiersMuJoCoEnv() as env:
        first = env.reset(17, perturbation)
        first_positions = first["objects"]
        first_mass = float(env.model.body_mass[env._body_ids["plate"]])
        second = env.reset(17, perturbation)
        assert first_positions == second["objects"]
        assert float(env.model.body_mass[env._body_ids["plate"]]) == first_mass


def test_observation_contains_camera_and_physics_state() -> None:
    with FortifiersMuJoCoEnv(width=160, height=120) as env:
        observation = env.reset(3)
        assert len(observation["qpos"]) == env.model.nq
        assert set(observation["objects"]) == {"plate", "cup", "fork", "spoon"}
        env.step({"type": "open_drawer"})
        assert env.observe()["drawer"] > 0
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (120, 160, 3)


def test_task_graph_requires_bimanual_handoff() -> None:
    from fortifiers_mujoco.task import build_dinner_table_plan, validate_dinner_table_plan

    plan = build_dinner_table_plan()
    validate_dinner_table_plan(plan)
    assert {step.arm for step in plan} == {"left", "right"}
    assert any(step.verb == "handoff" for step in plan)
