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
        assert env.model.neq == 10
        assert not any(env.data.eq_active[equality_id] for equality_id in env._retention_equality_ids.values())
        assert all(env.data.eq_active[equality_id] for equality_id in env._drawer_stow_equality_ids.values())


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
        assert set(observation["objects"]["cup"]["grasp_target_by_arm"]) == {"left", "right"}
        env.step({"type": "open_drawer"})
        assert env.observe()["drawer"] > 0
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (120, 160, 3)
        camera_observation = env.observe(include_camera=True)
        assert set(camera_observation["camera_rgb_views"]) == {
            "overview",
            "overhead",
            "left_oblique",
            "right_oblique",
        }
        assert np.asarray(camera_observation["camera_rgb_views"]["overhead"]).shape == (120, 160, 3)
        assert len(env.control_spec()["names"]) == env.model.nu
        assert env.control_vector().shape == (env.model.nu,)


def test_drawer_stow_carries_utensil_until_measured_grasp() -> None:
    with FortifiersMuJoCoEnv() as env:
        reset = env.reset(1001)
        fork_before = np.asarray(reset["objects"]["fork"]["position"])
        spoon_before = np.asarray(reset["objects"]["spoon"]["position"])
        assert all(env.data.eq_active[equality_id] for equality_id in env._drawer_stow_equality_ids.values())

        opened = env.step({"type": "open_drawer"})
        fork_open = np.asarray(opened["objects"]["fork"]["position"])
        spoon_open = np.asarray(opened["objects"]["spoon"]["position"])
        assert opened["task"]["next"] == "retrieve_fork"
        assert fork_open[1] < fork_before[1] - 0.20
        assert spoon_open[1] < spoon_before[1] - 0.20

        target = opened["objects"]["fork"]["grasp_target_by_arm"]["left"]
        moved = env.step(
            {
                "type": "move_to",
                "arm": "left",
                "target": target,
                "frame": "pinch",
                "orientation": "auto",
                "approach": True,
                "gripper": "open",
            }
        )
        assert moved["task"]["last_action"]["ok"]
        grasped = env.step({"type": "grasp", "arm": "left", "object": "fork"})
        assert grasped["task"]["last_action"]["ok"]
        assert grasped["task"]["held_by"]["fork"] == "left"
        assert not env._drawer_stow_is_active("fork")
        assert env._retention_is_active("left", "fork")


def test_task_graph_requires_bimanual_handoff() -> None:
    from fortifiers_mujoco.task import build_dinner_table_plan, validate_dinner_table_plan

    plan = build_dinner_table_plan()
    validate_dinner_table_plan(plan)
    assert {step.arm for step in plan} == {"left", "right"}
    assert any(step.verb == "handoff" for step in plan)
