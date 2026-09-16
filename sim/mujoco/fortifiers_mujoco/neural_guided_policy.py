"""Neural task-intent gate for the verified dinner-table controller."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from .imitation_policy import load_intent_checkpoint, model_inputs
from .task import build_dinner_table_plan


class NeuralIntentAdapter:
    """Predict the next task step from the live RGB/state/language frame."""

    name = "camera-state-language-neural-intent-v1"

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        self.device = device
        self.model, self.payload = load_intent_checkpoint(Path(checkpoint), device)
        self.camera_names = tuple(self.payload["camera_names"])
        self.task_step_ids = tuple(self.payload["task_step_ids"])
        expected_steps = tuple(step.id for step in build_dinner_table_plan())
        if self.task_step_ids != expected_steps:
            raise ValueError("intent checkpoint task steps do not match the live dinner-table plan")

    def predict(self, observation: Mapping[str, Any], instruction: str) -> dict[str, Any]:
        import torch

        images, state, language = model_inputs(observation, instruction, self.camera_names)
        with torch.inference_mode():
            logits = self.model(
                torch.from_numpy(images).to(self.device),
                torch.from_numpy(state).to(self.device),
                torch.from_numpy(language).to(self.device),
            )
            probabilities = torch.softmax(logits, dim=1)[0]
            phase_index = int(probabilities.argmax().cpu())
            confidence = float(probabilities[phase_index].cpu())
        return {
            "phase_index": phase_index,
            "step_id": self.task_step_ids[phase_index],
            "confidence": confidence,
            "probabilities": [float(value) for value in probabilities.cpu().tolist()],
        }

    def timed_predict(self, observation: Mapping[str, Any], instruction: str) -> tuple[dict[str, Any], float]:
        started = time.perf_counter_ns()
        prediction = self.predict(observation, instruction)
        return prediction, (time.perf_counter_ns() - started) / 1_000_000
