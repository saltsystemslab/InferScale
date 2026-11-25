from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

PARAMETERS = {
    "launch_folder": "/workspace/results/launches/<memory-launch>",
    "models": ["llama", "mistral", "qwen"],
}


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, format="{message}")
    launch_folder = Path(PARAMETERS["launch_folder"])
    manifest = read_json(launch_folder / "manifest.json")
    model_ids = model_ids_of(manifest)
    runs = {path: read_json(Path(path)) for path in manifest["run_configs"]}
    finished_by_model = {
        model: [
            path
            for path, run in runs.items()
            if run["model"] == model_ids[model] and has_finished(run)
        ]
        for model in PARAMETERS["models"]
    }
    for model, run_config_paths in finished_by_model.items():
        logger.info("{}: {} finished runs", model, len(run_config_paths))
    selected = [path for paths in finished_by_model.values() for path in paths]
    if not selected:
        logger.error("No finished run of these models in {}", launch_folder)
        raise SystemExit(1)
    selected_manifest = launch_folder / f"manifest-{'-'.join(finished_by_model)}.json"
    selected_manifest.write_text(
        json.dumps({**manifest, "run_configs": selected}, indent=2), encoding="utf-8"
    )
    logger.info("Selected {} of {} runs", len(selected), len(runs))
    logger.info('Set "manifest" in configs/launch/memory-judge.json to {}', selected_manifest)


def model_ids_of(manifest: dict[str, Any]) -> dict[str, str]:
    model_ids = read_json(Path(manifest["runtime_config"]))["models"]
    unknown = [model for model in PARAMETERS["models"] if model not in model_ids]
    if unknown:
        logger.error(
            "Unknown models: {}. This launch has: {}", ", ".join(unknown), ", ".join(model_ids)
        )
        raise SystemExit(1)
    return model_ids


def has_finished(run: dict[str, Any]) -> bool:
    return (Path(run["results_dir"]) / run["run_id"] / "summary.json").is_file()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
