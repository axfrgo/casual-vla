import numpy as np
import pytest

from fortifiers_mujoco.vla_adapter import (
    STATE_NAMES,
    action_to_joint_targets,
    dataset_features,
    observation_frame,
    state_vector,
)
from fortifiers_mujoco.imitation_policy import CameraStateIntentPolicy, state_to_control_vector


def _observation() -> dict:
    return {
        "drawer": 0.12,
        "arms": {
            "left": {"joints": [1, 2, 3, 4, 5], "gripper": 1.2},
            "right": {"joints": [6, 7, 8, 9, 10], "gripper": -0.12},
        },
        "camera_rgb": np.zeros((8, 10, 3), dtype=np.uint8),
        "camera_rgb_views": {
            "overview": np.zeros((8, 10, 3), dtype=np.uint8),
            "overhead": np.ones((8, 10, 3), dtype=np.uint8),
        },
    }


def test_state_vector_is_stable_and_typed() -> None:
    state = state_vector(_observation())
    assert state.dtype == np.float32
    assert state.shape == (len(STATE_NAMES),)
    assert state.tolist()[:5] == [1, 2, 3, 4, 5]
    assert state[5] == pytest.approx(1.2)
    assert state[12] == pytest.approx(0.12)
    assert state[13] == pytest.approx(0.0)
    assert state[14] == pytest.approx(1.0)
    assert state[-1] == pytest.approx(0.0)


def test_neural_control_vector_matches_actuator_wrist_order() -> None:
    control = state_to_control_vector(_observation())
    np.testing.assert_allclose(control, [0.12, 1, 2, 3, 5, 4, 1.2, 6, 7, 8, 10, 9, -0.12])


def test_neural_intent_policy_emits_the_six_phase_contract() -> None:
    torch = pytest.importorskip("torch")
    model = CameraStateIntentPolicy(num_cameras=3, state_dim=21, num_steps=6)
    with torch.inference_mode():
        logits = model(
            torch.zeros((2, 9, 32, 40)),
            torch.zeros((2, 21)),
            torch.zeros((2, 17)),
        )
    assert tuple(logits.shape) == (2, 6)


def test_observation_frame_maps_requested_camera_views() -> None:
    frame = observation_frame(_observation(), "set the table", ("overview", "overhead"))
    assert frame["task"] == "set the table"
    assert frame["state"].shape == (21,)
    assert frame["camera1"].shape == (8, 10, 3)
    assert int(frame["camera2"].max()) == 1


def test_dataset_features_and_action_contract() -> None:
    names = ["drawer_position", "left_shoulder_pan_motor"]
    features = dataset_features(names, ("overview", "overhead"), (8, 10, 3))
    assert features["action"]["shape"] == (2,)
    assert features["observation.state"]["shape"] == (21,)
    assert features["observation.images.camera2"]["dtype"] == "image"

    targets = action_to_joint_targets(
        np.array([-1.0, 99.0], dtype=np.float32),
        {"names": names, "ranges": {names[0]: (0.0, 0.3), names[1]: (-2.0, 2.0)}},
    )
    assert targets == {"drawer_position": 0.0, "left_shoulder_pan_motor": 2.0}


def test_action_contract_rejects_wrong_shape_and_nan() -> None:
    spec = {"names": ["a"], "ranges": {"a": (-1.0, 1.0)}}
    with pytest.raises(ValueError, match="does not match"):
        action_to_joint_targets([0.0, 1.0], spec)
    with pytest.raises(ValueError, match="non-finite"):
        action_to_joint_targets([float("nan")], spec)
