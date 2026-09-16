"""Export the trained imitation checkpoint to an OpenVINO IR model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .imitation_policy import LANGUAGE_VOCAB, CameraStatePolicy, _require_torch


def export(checkpoint: Path, output: Path) -> dict[str, object]:
    runtime = _require_torch()
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError("OpenVINO is not installed. Install the optional edge dependencies.") from exc

    payload = runtime.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "fortifiers.imitation-checkpoint.v1":
        raise ValueError("checkpoint is not a fortifiers.imitation-checkpoint.v1 artifact")
    config = payload["model_config"]
    model = CameraStatePolicy(
        int(config["num_cameras"]),
        int(config["state_dim"]),
        int(config["action_dim"]),
    ).eval()
    model.load_state_dict(payload["state_dict"])
    example = (
        runtime.zeros(1, int(config["num_cameras"]) * 3, int(config["image_height"]), int(config["image_width"]), dtype=runtime.float32),
        runtime.zeros(1, int(config["state_dim"]), dtype=runtime.float32),
        runtime.zeros(1, len(LANGUAGE_VOCAB), dtype=runtime.float32),
    )
    converted = ov.convert_model(model, example_input=example)
    input_shapes = (
        (1, int(config["num_cameras"]) * 3, int(config["image_height"]), int(config["image_width"])),
        (1, int(config["state_dim"])),
        (1, len(LANGUAGE_VOCAB)),
    )
    converted.reshape({port: list(shape) for port, shape in zip(converted.inputs, input_shapes)})
    output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(converted, str(output))
    input_specs = [
        {"key": key, "name": port.any_name, "shape": list(shape), "precision": str(port.get_element_type())}
        for key, port, shape in zip(("images", "state", "language"), converted.inputs, input_shapes)
    ]
    sidecar = {
        "schema": "fortifiers.imitation-openvino.v1",
        "model": str(output),
        "model_artifact": output.name,
        "source_checkpoint": str(checkpoint),
        "checkpoint_artifact": checkpoint.name,
        "camera_names": payload["camera_names"],
        "state_names": payload["state_names"],
        "action_names": payload["action_names"],
        "action_ranges": payload["action_ranges"],
        "instruction": payload["instruction"],
        "language_vocab": payload["language_vocab"],
        "action_mode": payload["action_mode"],
        "delta_scales": payload["delta_scales"],
        "model_config": config,
        "inputs": input_specs,
        "precision": "FP32",
        "evidence_boundary": "OpenVINO IR exported from the exact trained imitation checkpoint; task success and device performance require their separate evaluation reports.",
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    checkpoint_metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    checkpoint_metadata["checkpoint"] = str(checkpoint)
    checkpoint_metadata["checkpoint_artifact"] = checkpoint.name
    checkpoint_metadata["artifact"] = "trained camera/state/language imitation checkpoint metadata"
    output.with_suffix(".checkpoint.json").write_text(
        json.dumps(checkpoint_metadata, indent=2) + "\n", encoding="utf-8"
    )
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an imitation checkpoint to OpenVINO IR")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/imitation-policy.xml"))
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
