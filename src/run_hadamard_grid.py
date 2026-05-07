#!/usr/bin/env python3
"""Run a Hadamard-D seed grid over analog non-ideality levels."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch

from aihwkit.simulator.parameters.enums import WeightNoiseType

from src.eval_analog_perplexity import (
    AnalogPerplexityConfig,
    DEFAULT_ANALOG_TARGETS,
    run_evaluation,
)
from src.hardware_configs import build_rpu_config
from src.llama_model import DEFAULT_MODEL_NAME, TORCH_DTYPE_CHOICES, resolve_torch_dtype


DEFAULT_GRID_TARGETS = DEFAULT_ANALOG_TARGETS


@dataclass(frozen=True)
class GridPoint:
    ir_drop: float
    input_bits: int
    weight_noise: float


@dataclass(frozen=True)
class CheckpointSpec:
    index: int
    path: Path
    label: str


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def input_bits_to_resolution(bits: int) -> float:
    return float(2**bits - 2) if bits > 0 else -1.0


def format_value(value: object) -> str:
    text = f"{value:g}" if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p")


def safe_label(text: str) -> str:
    label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return label.strip("_") or "checkpoint"


def build_grid_rpu_config(point: GridPoint):
    """Build a fresh inference config with only the requested effects enabled."""
    cfg = build_rpu_config("ir_drop_only")
    cfg.forward.ir_drop = point.ir_drop
    cfg.forward.inp_res = input_bits_to_resolution(point.input_bits)
    cfg.forward.out_res = -1.0
    cfg.forward.out_bound = -1.0
    cfg.forward.w_noise = point.weight_noise
    cfg.forward.w_noise_type = (
        WeightNoiseType.ADDITIVE_CONSTANT
        if point.weight_noise > 0
        else WeightNoiseType.NONE
    )
    cfg.forward.inp_noise = 0.0
    cfg.forward.out_noise = 0.0
    cfg.noise_model = None
    cfg.drift_compensation = None
    return cfg


def point_name(point: GridPoint) -> str:
    return (
        f"ir{format_value(point.ir_drop)}_"
        f"bits{format_value(point.input_bits)}_"
        f"wnoise{format_value(point.weight_noise)}"
    )


def json_ready(data):
    if isinstance(data, dict):
        return {str(key): json_ready(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [json_ready(value) for value in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return data
    return data


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(data), f, indent=2)


def run_one(
    *,
    args,
    point: GridPoint,
    seed: Optional[int],
    run_kind: str,
    include_float: bool,
    checkpoint: Optional[CheckpointSpec] = None,
) -> tuple[dict, Path]:
    is_identity = run_kind == "identity"
    is_checkpoint = run_kind == "checkpoint"
    if is_identity:
        run_name = "identity"
    elif is_checkpoint:
        if checkpoint is None:
            raise ValueError("checkpoint run_kind requires a checkpoint spec.")
        run_name = checkpoint.label
    else:
        run_name = f"seed{seed}"
    raw_path = args.output_dir / f"{point_name(point)}_{run_name}.json"
    if args.skip_existing and raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as f:
            return json.load(f), raw_path

    config = AnalogPerplexityConfig(
        model_name=args.model_name,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        device=None if args.device in (None, "auto") else args.device,
        checkpoint_path=str(checkpoint.path) if checkpoint is not None else None,
        rotation_mode="identity" if is_identity else ("learned" if is_checkpoint else "hadamard_D"),
        r2_mode="identity" if is_identity else ("learned" if is_checkpoint else "hadamard_D"),
        seed=0 if seed is None else seed,
        r2_seed_offset=args.r2_seed_offset,
        dataset=args.dataset,
        split=args.split,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_eval_tokens=args.max_eval_tokens,
        hardware_preset="ir_drop_only",
        rpu_config=build_grid_rpu_config(point),
        analog_targets=tuple(args.analog_targets),
        online_hadamards=args.online_hadamards and not is_identity,
        page_analog_tiles=args.page_analog_tiles,
        analog_storage_device=args.analog_storage_device,
        analog_execution_device=args.analog_execution_device,
        cpu_paged_analog_targets=tuple(args.cpu_paged_analog_targets),
        clear_paged_cuda_cache=args.clear_paged_cuda_cache,
        run_float_prepared=include_float,
        run_analog_identity=is_identity,
        run_analog_rotated=not is_identity,
        use_wandb=False,
        progress_every=args.progress_every,
    )
    results = run_evaluation(config)
    results["grid"] = {
        "run_kind": run_kind,
        "seed": seed,
        "checkpoint_path": str(checkpoint.path) if checkpoint is not None else None,
        "checkpoint_label": checkpoint.label if checkpoint is not None else None,
        "ir_drop": point.ir_drop,
        "input_bits": point.input_bits,
        "input_resolution": input_bits_to_resolution(point.input_bits),
        "weight_noise": point.weight_noise,
    }
    write_json(raw_path, results)
    return results, raw_path


def metric_or_nan(results: dict, run_key: str, metric: str) -> float:
    return float(results.get("runs", {}).get(run_key, {}).get(metric, math.nan))


def build_rows(
    *,
    point: GridPoint,
    identity_results: dict,
    identity_path: Path,
    rotated_results: Iterable[tuple[int, dict, Path]],
    checkpoint_results: Iterable[tuple[CheckpointSpec, dict, Path]],
    float_results: Optional[dict],
) -> list[dict]:
    rows = []
    float_ppl = math.nan
    if float_results is not None:
        float_ppl = metric_or_nan(float_results, "float_prepared", "ppl")
    if math.isnan(float_ppl):
        float_ppl = metric_or_nan(identity_results, "float_prepared", "ppl")

    identity_ppl = metric_or_nan(identity_results, "analog_identity", "ppl")
    identity_nll = metric_or_nan(identity_results, "analog_identity", "nll")
    rows.append(
        {
            "run_kind": "identity",
            "seed": "",
            "ir_drop": point.ir_drop,
            "input_bits": point.input_bits,
            "weight_noise": point.weight_noise,
            "run_key": "analog_identity",
            "checkpoint_path": "",
            "checkpoint_label": "",
            "nll": identity_nll,
            "ppl": identity_ppl,
            "tokens": int(metric_or_nan(identity_results, "analog_identity", "tokens")),
            "float_ppl": float_ppl,
            "identity_ppl": identity_ppl,
            "rotated_ppl": "",
            "improvement_ratio": "",
            "hardware_preset": "ir_drop_only+grid_overrides",
            "analog_targets": ",".join(identity_results.get("analog_targets", [])),
            "online_hadamards": identity_results.get("online_hadamards", False),
            "json_path": str(identity_path),
        }
    )

    for seed, results, path in rotated_results:
        rotated_ppl = metric_or_nan(results, "analog_rotated", "ppl")
        improvement = identity_ppl / rotated_ppl if rotated_ppl > 0 else math.nan
        rows.append(
            {
                "run_kind": "hadamard_D",
                "seed": seed,
                "ir_drop": point.ir_drop,
                "input_bits": point.input_bits,
                "weight_noise": point.weight_noise,
                "run_key": "analog_rotated",
                "checkpoint_path": "",
                "checkpoint_label": "",
                "nll": metric_or_nan(results, "analog_rotated", "nll"),
                "ppl": rotated_ppl,
                "tokens": int(metric_or_nan(results, "analog_rotated", "tokens")),
                "float_ppl": float_ppl,
                "identity_ppl": identity_ppl,
                "rotated_ppl": rotated_ppl,
                "improvement_ratio": improvement,
                "hardware_preset": "ir_drop_only+grid_overrides",
                "analog_targets": ",".join(results.get("analog_targets", [])),
                "online_hadamards": results.get("online_hadamards", True),
                "json_path": str(path),
            }
        )
    for checkpoint, results, path in checkpoint_results:
        learned_ppl = metric_or_nan(results, "analog_rotated", "ppl")
        improvement = identity_ppl / learned_ppl if learned_ppl > 0 else math.nan
        rows.append(
            {
                "run_kind": "checkpoint",
                "seed": "",
                "ir_drop": point.ir_drop,
                "input_bits": point.input_bits,
                "weight_noise": point.weight_noise,
                "run_key": "analog_rotated",
                "checkpoint_path": str(checkpoint.path),
                "checkpoint_label": checkpoint.label,
                "nll": metric_or_nan(results, "analog_rotated", "nll"),
                "ppl": learned_ppl,
                "tokens": int(metric_or_nan(results, "analog_rotated", "tokens")),
                "float_ppl": float_ppl,
                "identity_ppl": identity_ppl,
                "rotated_ppl": learned_ppl,
                "improvement_ratio": improvement,
                "hardware_preset": "ir_drop_only+grid_overrides",
                "analog_targets": ",".join(results.get("analog_targets", [])),
                "online_hadamards": results.get("online_hadamards", True),
                "json_path": str(path),
            }
        )
    return rows


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    write_json(json_path, {"rows": rows})
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


def iter_grid(args) -> list[GridPoint]:
    points = [
        GridPoint(ir_drop=ir, input_bits=bits, weight_noise=noise)
        for ir, bits, noise in product(
            args.ir_drop_values,
            args.input_bits_values,
            args.weight_noise_values,
        )
    ]
    if args.limit is not None:
        points = points[: args.limit]
    return points


def build_checkpoint_specs(paths: Sequence[Path]) -> list[CheckpointSpec]:
    return [
        CheckpointSpec(index=index, path=path, label=f"ckpt{index}_{safe_label(path.stem)}")
        for index, path in enumerate(paths)
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grid search Hadamard-D seeds over analog hardware levels.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/hadamard_grid"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", *TORCH_DTYPE_CHOICES.keys()])
    parser.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-eval-tokens", type=int, default=8192)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        type=Path,
        default=[],
        help="Optional learned rotation checkpoints to evaluate at each hardware grid point.",
    )
    parser.add_argument("--ir-drop-values", type=parse_float_list, default=parse_float_list("0,0.1,0.25,0.5,1.0"))
    parser.add_argument("--input-bits-values", type=parse_int_list, default=parse_int_list("-1,10,8,6"))
    parser.add_argument("--weight-noise-values", type=parse_float_list, default=parse_float_list("0,0.005,0.01,0.02"))
    parser.add_argument("--analog-targets", nargs="+", default=list(DEFAULT_GRID_TARGETS))
    parser.add_argument("--no-online-hadamards", action="store_true")
    parser.add_argument("--r2-seed-offset", type=int, default=1)
    parser.add_argument("--page-analog-tiles", action="store_true")
    parser.add_argument("--analog-storage-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--analog-execution-device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--cpu-paged-analog-targets", nargs="*", default=[])
    parser.add_argument("--clear-paged-cuda-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit hardware grid points for smoke tests.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.seeds = parse_int_list(args.seeds)
    args.checkpoints = build_checkpoint_specs(args.checkpoints)
    args.online_hadamards = not args.no_online_hadamards
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    float_results = None
    first_run = True
    points = iter_grid(args)
    total_runs = len(points) * (1 + len(args.seeds) + len(args.checkpoints))
    run_idx = 0

    for point in points:
        run_idx += 1
        print(f"[{run_idx}/{total_runs}] identity {point_name(point)}", flush=True)
        identity_results, identity_path = run_one(
            args=args,
            point=point,
            seed=None,
            run_kind="identity",
            include_float=first_run,
        )
        if first_run:
            float_results = identity_results
            first_run = False

        rotated = []
        for seed in args.seeds:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] hadamard_D seed={seed} {point_name(point)}", flush=True)
            rotated_results, rotated_path = run_one(
                args=args,
                point=point,
                seed=seed,
                run_kind="hadamard_D",
                include_float=False,
            )
            rotated.append((seed, rotated_results, rotated_path))

        checkpoint_runs = []
        for checkpoint in args.checkpoints:
            run_idx += 1
            print(
                f"[{run_idx}/{total_runs}] checkpoint {checkpoint.path} {point_name(point)}",
                flush=True,
            )
            checkpoint_results, checkpoint_path = run_one(
                args=args,
                point=point,
                seed=None,
                run_kind="checkpoint",
                include_float=False,
                checkpoint=checkpoint,
            )
            checkpoint_runs.append((checkpoint, checkpoint_results, checkpoint_path))

        rows.extend(
            build_rows(
                point=point,
                identity_results=identity_results,
                identity_path=identity_path,
                rotated_results=rotated,
                checkpoint_results=checkpoint_runs,
                float_results=float_results,
            )
        )
        write_summary(args.output_dir, rows)

    write_summary(args.output_dir, rows)


if __name__ == "__main__":
    main()
