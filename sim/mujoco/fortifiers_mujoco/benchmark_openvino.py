from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def benchmark(model_path: Path, device: str, iterations: int, warmup: int) -> dict[str, object]:
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError(
            "OpenVINO is not installed. Install the optional edge dependencies before benchmarking."
        ) from exc

    core = ov.Core()
    model = core.read_model(model=str(model_path))
    compiled = core.compile_model(model, device)
    input_specs: list[dict[str, object]] = []
    tensors: dict[object, np.ndarray] = {}
    for input_port in compiled.inputs:
        partial_shape = input_port.partial_shape
        if not partial_shape.is_static:
            raise ValueError("benchmark model inputs must have static shapes")
        shape = tuple(int(dimension) for dimension in partial_shape.to_shape())
        input_specs.append({"input": input_port.any_name, "shape": list(shape), "precision": str(input_port.get_element_type())})
        tensors[input_port] = np.zeros(shape, dtype=np.float32)
    request = compiled.create_infer_request()
    for _ in range(warmup):
        request.infer(tensors)

    samples_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        request.infer(tensors)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    samples_ms.sort()
    mean_ms = sum(samples_ms) / len(samples_ms)
    return {
        "report_schema": "fortifiers.openvino.inference-benchmark.v1",
        "model": str(model_path),
        "device": device,
        "inputs": input_specs,
        "iterations": iterations,
        "warmup": warmup,
        "latency_ms": {
            "mean": mean_ms,
            "p50": samples_ms[len(samples_ms) // 2],
            "p95": samples_ms[min(len(samples_ms) - 1, int(len(samples_ms) * 0.95))],
        },
        "throughput_fps": 1000 / mean_ms,
        "task_success_rate": None,
        "note": "Inference benchmark only; task success requires the connected MuJoCo policy evaluation report.",
        "evidence_boundary": "This is an OpenVINO inference benchmark on the stated device, not a task-success or hardware-robot result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a real OpenVINO model")
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup cannot be negative")
    report = benchmark(args.model, args.device, args.iterations, args.warmup)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
