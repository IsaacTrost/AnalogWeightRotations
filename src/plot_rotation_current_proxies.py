"""Plot activation-weight current proxies for base vs. rotated checkpoints.

This tests whether a rotation helps IR drop by reducing peaky joint current
patterns, even when activations/weights are not shifted toward particular IDs.

Example:
  python src/plot_rotation_current_proxies.py \
      --config configs/eval_high_ir_8bit.json \
      --checkpoint checkpoints/large_ir_8_bit.pt \
      --layers 3 20 21 \
      --output results/figures/large_ir_8_bit_current_proxy.png
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.llama_model import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXTS,
    is_llama_like_model,
    load_model_and_tokenizer,
    resolve_torch_dtype,
)
from src.llama_prepare import prepare_model_for_rotation
from src.llama_rotation import bake_rotation_state_into_model
from src.plot_rotation_weight_heatmaps import MODULE_SPECS, _get_linear


DEFAULT_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _load_json_config(path: Optional[str]) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise TypeError(f"Config file {path} must contain a JSON object.")
    return config


def _load_checkpoint(path: str) -> Mapping[str, object]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint {path} did not contain a dict-like object.")
    return state


def _concentration(values: np.ndarray) -> dict[str, float]:
    values = np.abs(np.asarray(values, dtype=np.float64))
    values = values[np.isfinite(values)]
    total = float(values.sum())
    if values.size == 0 or total == 0.0:
        return {
            "max_over_mean": 0.0,
            "top1pct_share": 0.0,
            "top5pct_share": 0.0,
            "effective_frac": 0.0,
        }

    mean = float(values.mean())
    probs = values / total
    sorted_values = np.sort(values)[::-1]
    top1_count = max(1, int(math.ceil(0.01 * len(sorted_values))))
    top5_count = max(1, int(math.ceil(0.05 * len(sorted_values))))
    return {
        "max_over_mean": float(values.max() / (mean + 1e-30)),
        "top1pct_share": float(sorted_values[:top1_count].sum() / (total + 1e-30)),
        "top5pct_share": float(sorted_values[:top5_count].sum() / (total + 1e-30)),
        "effective_frac": float((1.0 / float(np.sum(probs * probs))) / len(values)),
    }


def _get_module(layer: torch.nn.Module, module_name: str) -> torch.nn.Linear:
    return _get_linear(layer, MODULE_SPECS[module_name])


@torch.inference_mode()
def _capture_model_current_inputs(
    *,
    model_name: str,
    torch_dtype: Optional[torch.dtype],
    device: Optional[str],
    rotation_state: Optional[Mapping[str, object]],
    layers: Sequence[int],
    modules: Sequence[str],
    max_length: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
    )
    if not is_llama_like_model(model):
        raise ValueError(f"{model_name} does not expose a LLaMA-style layout.")
    prepare_model_for_rotation(model)
    if rotation_state is not None:
        bake_rotation_state_into_model(model, rotation_state)

    captures: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        for module_name in modules:
            key = f"L{layer_idx}.{module_name}"
            module = _get_module(layer, module_name)
            weights[key] = module.weight.detach().to(device="cpu", dtype=torch.float32).numpy()

            def capture_pre_hook(
                _module: torch.nn.Module,
                inputs: tuple[torch.Tensor, ...],
                *,
                hook_key: str = key,
            ) -> None:
                activation = inputs[0].detach().to(device="cpu", dtype=torch.float32)
                flattened = activation.reshape(-1, activation.shape[-1])
                captures[hook_key] = flattened.abs().mean(dim=0).numpy()

            handles.append(module.register_forward_pre_hook(capture_pre_hook))

    try:
        encoded = tokenizer(
            list(DEFAULT_TEXTS),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        target_device = next(model.parameters()).device
        encoded = {name: value.to(target_device) for name, value in encoded.items()}
        model.eval()
        model(**encoded)
    finally:
        for handle in handles:
            handle.remove()
        del model, tokenizer
        gc.collect()

    return captures, weights


def _current_proxy_metrics(abs_activation: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    abs_weight = np.abs(weight.astype(np.float64))
    abs_activation = np.abs(abs_activation.astype(np.float64))
    if abs_weight.shape[1] != abs_activation.shape[0]:
        raise ValueError(
            f"Activation dim {abs_activation.shape[0]} does not match weight input dim {abs_weight.shape[1]}."
        )

    # Column proxy: input-wire demand from activation magnitude times total column conductance.
    column_current = abs_activation * abs_weight.sum(axis=0)
    # Row proxy: output-row demand induced by the activation pattern.
    row_current = abs_weight @ abs_activation

    col = _concentration(column_current)
    row = _concentration(row_current)
    return {
        "column_max_over_mean": col["max_over_mean"],
        "column_top1pct_share": col["top1pct_share"],
        "column_top5pct_share": col["top5pct_share"],
        "column_effective_frac": col["effective_frac"],
        "row_max_over_mean": row["max_over_mean"],
        "row_top1pct_share": row["top1pct_share"],
        "row_top5pct_share": row["top5pct_share"],
        "row_effective_frac": row["effective_frac"],
    }


def _analyze_current_proxies(
    base_acts: Mapping[str, np.ndarray],
    base_weights: Mapping[str, np.ndarray],
    rotated_acts: Mapping[str, np.ndarray],
    rotated_weights: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(base_acts):
        base_metrics = _current_proxy_metrics(base_acts[key], base_weights[key])
        rotated_metrics = _current_proxy_metrics(rotated_acts[key], rotated_weights[key])
        row: dict[str, object] = {"point": key}
        row.update({f"base_{name}": value for name, value in base_metrics.items()})
        row.update(rotated_metrics)
        row.update(
            {
                f"ratio_{name}": rotated_metrics[name] / base_metrics[name]
                if base_metrics[name]
                else 0.0
                for name in rotated_metrics
            }
        )
        rows.append(row)
    return rows


def _plot_current_proxies(rows: Sequence[Mapping[str, object]], output: str, *, title: str, dpi: int) -> None:
    labels = [str(row["point"]) for row in rows]
    x_pos = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(nrows=2, figsize=(max(16, 0.62 * len(labels)), 13), constrained_layout=True)
    specs = (
        ("column_max_over_mean", "Column Current Peakiness\nmax / mean"),
        ("row_max_over_mean", "Row Current Peakiness\nmax / mean"),
    )

    for axis, (metric, ylabel) in zip(axes, specs):
        base = np.array([float(row[f"base_{metric}"]) for row in rows])
        rotated = np.array([float(row[metric]) for row in rows])
        ratio = rotated / np.maximum(base, 1e-30)

        axis.bar(x_pos - width / 2, base, width, label="Base", color="#7f849c", alpha=0.9)
        colors = np.where(ratio < 1.0, "#4daf4a", "#e41a1c")
        axis.bar(x_pos + width / 2, rotated, width, label="Rotated", color=colors, alpha=0.9)

        for idx, value in enumerate(rotated):
            axis.text(
                idx + width / 2,
                value,
                f"{ratio[idx]:.2f}x",
                ha="center",
                va="bottom",
                fontsize=12,
                rotation=90,
            )

        axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.45)
        axis.set_ylabel(ylabel, fontsize=18)
        axis.tick_params(axis="y", labelsize=14)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=15, loc="upper right")

    axes[-1].set_xticks(x_pos)
    axes[-1].set_xticklabels(labels, rotation=60, ha="right", fontsize=13)
    fig.suptitle(title, fontsize=22)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _print_summary(rows: Sequence[Mapping[str, object]], limit: int) -> None:
    ranked = sorted(rows, key=lambda row: float(row["ratio_column_max_over_mean"]))
    print("\nBiggest reductions in column current peakiness:")
    for row in ranked[:limit]:
        print(
            f"{str(row['point']):18s} "
            f"column max/mean {float(row['base_column_max_over_mean']):.2f}->{float(row['column_max_over_mean']):.2f} "
            f"ratio={float(row['ratio_column_max_over_mean']):.2f}; "
            f"top1 {float(row['base_column_top1pct_share']):.3f}->{float(row['column_top1pct_share']):.3f}"
        )

    ranked_bad = sorted(rows, key=lambda row: float(row["ratio_column_max_over_mean"]), reverse=True)
    print("\nBiggest increases in column current peakiness:")
    for row in ranked_bad[:limit]:
        print(
            f"{str(row['point']):18s} "
            f"column max/mean {float(row['base_column_max_over_mean']):.2f}->{float(row['column_max_over_mean']):.2f} "
            f"ratio={float(row['ratio_column_max_over_mean']):.2f}; "
            f"top1 {float(row['base_column_top1pct_share']):.3f}->{float(row['column_top1pct_share']):.3f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/figures/current_proxy.png")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float16", "bfloat16", "float32", "float64"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 20, 21])
    parser.add_argument("--modules", nargs="+", default=list(DEFAULT_MODULES), choices=list(DEFAULT_MODULES))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--top", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = _load_json_config(args.config)
    checkpoint = args.checkpoint or config.get("checkpoint_path") or config.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Provide --checkpoint or a config containing checkpoint_path.")

    model_name = str(config.get("model_name", args.model_name))
    torch_dtype = resolve_torch_dtype(str(config.get("torch_dtype", args.torch_dtype)))
    device = None if args.device == "auto" else args.device
    rotation_state = _load_checkpoint(str(checkpoint))

    print("Capturing base current inputs...")
    base_acts, base_weights = _capture_model_current_inputs(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device=device,
        rotation_state=None,
        layers=args.layers,
        modules=args.modules,
        max_length=args.max_length,
    )
    print("Capturing rotated current inputs...")
    rotated_acts, rotated_weights = _capture_model_current_inputs(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device=device,
        rotation_state=rotation_state,
        layers=args.layers,
        modules=args.modules,
        max_length=args.max_length,
    )

    rows = _analyze_current_proxies(base_acts, base_weights, rotated_acts, rotated_weights)
    _print_summary(rows, args.top)
    _plot_current_proxies(
        rows,
        args.output,
        title=f"Activation x Weight Current Proxy - {os.path.basename(str(checkpoint))}",
        dpi=args.dpi,
    )
    print(f"\nSaved current proxy plot to {args.output}")

    json_output = args.json_output or os.path.splitext(args.output)[0] + ".json"
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "checkpoint": str(checkpoint),
                "layers": args.layers,
                "modules": args.modules,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"Saved current proxy metrics to {json_output}")


if __name__ == "__main__":
    main()
