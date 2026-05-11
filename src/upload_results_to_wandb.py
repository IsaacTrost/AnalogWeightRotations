#!/usr/bin/env python3
"""Backfill saved evaluation results into Weights & Biases.

This script turns saved `results/**.json` files into W&B runs and uploads all
CSV/JSON/image files from each sweep directory as artifacts. It is intended for
post-hoc logging when experiments were run before W&B logging was enabled.

Example:
  python -m src.upload_results_to_wandb \
      --dirs results/high_ir results/new results/hadamard_grid
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.wandb_config import WANDB_ENTITY, WANDB_MODE, WANDB_PROJECT


RESULT_FILE_SUFFIXES = {".json", ".csv", ".png", ".jpg", ".jpeg", ".svg", ".pdf"}
SUMMARY_NAMES = {"summary.json", "summary.csv", "hadamard_by_cell.csv", "best_seed_by_cell.csv"}


def _sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a JSON object.")
    return data


def _flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in data.items():
        if key == "runs":
            continue
        if key == "grid" and isinstance(value, dict):
            for grid_key, grid_value in value.items():
                config[f"grid/{grid_key}"] = grid_value
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
        elif isinstance(value, list):
            config[key] = ",".join(str(item) for item in value)
        elif isinstance(value, dict):
            config[key] = json.dumps(value, sort_keys=True)
    return config


def _metrics_from_result(data: dict[str, Any]) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {}
    runs = data.get("runs")
    if isinstance(runs, dict):
        for run_label, run_metrics in runs.items():
            if not isinstance(run_metrics, dict):
                continue
            for metric_name in ("nll", "ppl", "tokens"):
                value = run_metrics.get(metric_name)
                if isinstance(value, (int, float)):
                    metrics[f"{run_label}/{metric_name}"] = value

    grid = data.get("grid")
    if isinstance(grid, dict):
        for key in ("run_kind", "ir_drop", "input_bits", "weight_noise", "seed", "checkpoint_label"):
            value = grid.get(key)
            if isinstance(value, (str, int, float)) or value is None:
                metrics[f"grid/{key}"] = "none" if value is None else value
    return metrics


def _add_file_artifact(wandb, run, path: Path, *, artifact_type: str, name_prefix: str) -> None:
    artifact = wandb.Artifact(
        name=f"{_sanitize_name(name_prefix)}-{_sanitize_name(path.stem)}",
        type=artifact_type,
        metadata={"source_path": str(path)},
    )
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def _upload_eval_json(wandb, path: Path, *, sweep_name: str, dry_run: bool) -> None:
    data = _json_load(path)
    run_name = f"{sweep_name}/{path.stem}"
    config = _flatten_config(data)
    metrics = _metrics_from_result(data)

    if dry_run:
        print(f"[dry-run] would upload run {run_name} metrics={sorted(metrics)}")
        return

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        mode=WANDB_MODE,
        name=run_name,
        group=sweep_name,
        job_type="backfilled_eval_result",
        config=config,
        reinit=True,
    )
    if metrics:
        wandb.log(metrics)
        for key, value in metrics.items():
            run.summary[key] = value
    _add_file_artifact(wandb, run, path, artifact_type="eval-json", name_prefix=run_name)
    wandb.finish()


def _upload_summary_run(wandb, directory: Path, *, sweep_name: str, dry_run: bool) -> None:
    csv_paths = sorted(path for path in directory.rglob("*.csv") if path.name in SUMMARY_NAMES)
    result_files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in RESULT_FILE_SUFFIXES
    )

    if dry_run:
        print(f"[dry-run] would upload summary run {sweep_name}: {len(csv_paths)} tables, {len(result_files)} files")
        return

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        mode=WANDB_MODE,
        name=f"{sweep_name}/summary",
        group=sweep_name,
        job_type="backfilled_result_summary",
        config={"result_dir": str(directory)},
        reinit=True,
    )

    for csv_path in csv_paths:
        table = wandb.Table(dataframe=pd.read_csv(csv_path))
        wandb.log({f"tables/{csv_path.stem}": table})

    artifact = wandb.Artifact(
        name=f"{_sanitize_name(sweep_name)}-all-results",
        type="result-directory",
        metadata={"source_dir": str(directory), "file_count": len(result_files)},
    )
    for path in result_files:
        artifact.add_file(str(path), name=str(path.relative_to(directory)))
    run.log_artifact(artifact)
    wandb.finish()


def upload_directory(wandb, directory: Path, *, dry_run: bool) -> None:
    if not directory.exists():
        raise FileNotFoundError(directory)

    sweep_name = directory.name
    eval_jsons = sorted(
        path for path in directory.rglob("*.json")
        if path.name not in SUMMARY_NAMES
    )
    print(f"{sweep_name}: found {len(eval_jsons)} eval JSON files")

    _upload_summary_run(wandb, directory, sweep_name=sweep_name, dry_run=dry_run)
    for path in eval_jsons:
        _upload_eval_json(wandb, path, sweep_name=sweep_name, dry_run=dry_run)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill saved result directories into W&B.")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["results/high_ir", "results/new", "results/hadamard_grid"],
        help="Result directories to upload.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without contacting W&B.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    import wandb

    if not args.dry_run:
        wandb.login(key=os.getenv("WANDB_API_KEY"))

    for raw_dir in args.dirs:
        upload_directory(wandb, Path(raw_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
