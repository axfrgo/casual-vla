from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import FortifiersMuJoCoEnv, PerturbationConfig


def run(episodes: int, start_seed: int, render: bool) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    with FortifiersMuJoCoEnv() as env:
        for offset in range(episodes):
            seed = start_seed + offset
            perturbation = PerturbationConfig(
                placement_jitter=0.08 + (offset % 3) * 0.04,
                mass_scale=1.0 + (offset % 2) * 0.1,
                friction_scale=0.9 + (offset % 4) * 0.08,
                object_scale=1.0 + (offset % 3) * 0.03,
                lighting_scale=0.8 + (offset % 3) * 0.15,
                background=("neutral", "warm", "cool")[offset % 3],
            )
            observation = env.reset(seed, perturbation)
            if render:
                env.render()
            rows.append(
                {
                    "seed": seed,
                    "objects": sorted(observation["objects"]),
                    "drawer": observation["drawer"],
                    "contacts": len(observation["contacts"]),
                    "perturbation": observation["perturbation"],
                    "camera": render,
                }
            )
    return {
        "environment": "mujoco-dual-arm-dinner-table",
        "episodes": rows,
        "note": "Scene-load and randomization smoke test only; no policy success score yet.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test randomized MuJoCo dinner-table scenes")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1001)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.episodes > 1000:
        parser.error("--episodes must be between 1 and 1000")
    report = run(args.episodes, args.start_seed, args.render)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
