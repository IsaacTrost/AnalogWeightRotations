"""Measure whether rotations concentrate weight/activation magnitudes by channel.

Example:
  python src/analyze_rotation_channel_concentration.py \
      --config configs/eval_high_ir_8bit.json \
      --checkpoint checkpoints/large_ir_8_bit.pt \
      --layers 3 20 21 \
      --output results/figures/large_ir_8_bit_channel_concentration.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Optional

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
from src.plot_rotation_weight_heatmaps import MODULE_SPECS, _get_linear, _rotated_weight


DEFAULT_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEFAULT_ACTIVATION_POINTS = ("q_proj_input", "o_proj_input", "mlp_input", "down_proj_input")


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
            "p99_over_mean": 0.0,
            "top1pct_share": 0.0,
            "top5pct_share": 0.0,
            "cv": 0.0,
            "effective_frac": 0.0,
            "index_center": 0.5,
            "top1pct_index_center": 0.5,
            "low_quarter_share": 0.25,
            "low_half_share": 0.5,
            "high_half_share": 0.5,
            "index_corr": 0.0,
        }

    mean = float(values.mean())
    probs = values / total
    effective_channels = 1.0 / float(np.sum(probs * probs))
    sorted_values = np.sort(values)[::-1]
    sorted_indices = np.argsort(values)[::-1]
    top1_count = max(1, int(math.ceil(0.01 * len(sorted_values))))
    top5_count = max(1, int(math.ceil(0.05 * len(sorted_values))))
    norm_indices = np.linspace(0.0, 1.0, len(values))
    low_quarter_count = max(1, int(math.ceil(0.25 * len(values))))
    low_half_count = max(1, int(math.ceil(0.5 * len(values))))
    top1_indices = sorted_indices[:top1_count]
    top1_weights = values[top1_indices]
    if len(values) > 1 and float(values.std()) > 0.0:
        index_corr = float(np.corrcoef(norm_indices, values)[0, 1])
    else:
        index_corr = 0.0

    return {
        "max_over_mean": float(values.max() / (mean + 1e-30)),
        "p99_over_mean": float(np.percentile(values, 99.0) / (mean + 1e-30)),
        "top1pct_share": float(sorted_values[:top1_count].sum() / (total + 1e-30)),
        "top5pct_share": float(sorted_values[:top5_count].sum() / (total + 1e-30)),
        "cv": float(values.std() / (mean + 1e-30)),
        "effective_frac": float(effective_channels / len(values)),
        "index_center": float(np.sum(norm_indices * values) / (total + 1e-30)),
        "top1pct_index_center": float(
            np.sum(norm_indices[top1_indices] * top1_weights) / (top1_weights.sum() + 1e-30)
        ),
        "low_quarter_share": float(values[:low_quarter_count].sum() / (total + 1e-30)),
        "low_half_share": float(values[:low_half_count].sum() / (total + 1e-30)),
        "high_half_share": float(values[low_half_count:].sum() / (total + 1e-30)),
        "index_corr": index_corr,
    }


def _summarize_change(base_values: np.ndarray, rotated_values: np.ndarray) -> dict[str, float]:
    base = _concentration(base_values)
    rotated = _concentration(rotated_values)
    summary = {key: rotated[key] for key in rotated}
    summary.update({f"base_{key}": base[key] for key in base})
    summary.update(
        {
            f"ratio_{key}": rotated[key] / base[key] if base[key] else 0.0
            for key in rotated
        }
    )
    return summary


def _weight_channel_scores(weight: torch.Tensor, axis: str) -> np.ndarray:
    abs_weight = weight.detach().to(device="cpu", dtype=torch.float32).abs()
    if axis == "input":
        return abs_weight.mean(dim=0).numpy()
    if axis == "output":
        return abs_weight.mean(dim=1).numpy()
    raise ValueError(f"Unknown weight axis: {axis}")


def _analyze_weights(
    model: torch.nn.Module,
    rotation_state: Mapping[str, object],
    modules: Sequence[str],
) -> list[dict[str, object]]:
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    rows: list[dict[str, object]] = []
    for layer_idx, layer in enumerate(model.model.layers):
        for module_name in modules:
            spec = MODULE_SPECS[module_name]
            linear = _get_linear(layer, spec)
            base_weight = linear.weight.detach().to(device="cpu", dtype=torch.float32)
            rotated_weight = _rotated_weight(linear, spec, rotation_state, layer_idx, head_dim)

            for axis in ("input", "output"):
                summary = _summarize_change(
                    _weight_channel_scores(base_weight, axis),
                    _weight_channel_scores(rotated_weight, axis),
                )
                rows.append(
                    {
                        "layer": layer_idx,
                        "module": module_name,
                        "axis": axis,
                        **summary,
                    }
                )
    return rows


def _register_activation_hook(
    layer: torch.nn.Module,
    point: str,
    name: str,
    captures: dict[str, np.ndarray],
    handles: list[torch.utils.hooks.RemovableHandle],
) -> None:
    if point == "q_proj_input":
        target = layer.self_attn.q_proj
    elif point == "o_proj_input":
        target = layer.self_attn.o_proj
    elif point == "mlp_input":
        target = layer.mlp
    elif point == "down_proj_input":
        target = layer.mlp.down_proj
    else:
        raise ValueError(f"Unknown activation point: {point}")

    def capture_pre_hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        activation = inputs[0].detach().to(device="cpu", dtype=torch.float32)
        flattened = activation.reshape(-1, activation.shape[-1])
        captures[name] = flattened.abs().mean(dim=0).numpy()

    handles.append(target.register_forward_pre_hook(capture_pre_hook))


@torch.inference_mode()
def _capture_activation_scores(
    *,
    model_name: str,
    torch_dtype: Optional[torch.dtype],
    device: Optional[str],
    rotation_state: Optional[Mapping[str, object]],
    layers: Sequence[int],
    activation_points: Sequence[str],
    texts: Sequence[str],
    max_length: int,
) -> dict[str, np.ndarray]:
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
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        for point in activation_points:
            _register_activation_hook(
                layer,
                point,
                f"layer_{layer_idx}.{point}",
                captures,
                handles,
            )

    try:
        encoded = tokenizer(
            list(texts),
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

    return captures


def _analyze_activations(
    *,
    model_name: str,
    torch_dtype: Optional[torch.dtype],
    device: Optional[str],
    rotation_state: Mapping[str, object],
    layers: Sequence[int],
    activation_points: Sequence[str],
    texts: Sequence[str],
    max_length: int,
) -> list[dict[str, object]]:
    print("Capturing base activations...")
    base = _capture_activation_scores(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device=device,
        rotation_state=None,
        layers=layers,
        activation_points=activation_points,
        texts=texts,
        max_length=max_length,
    )
    print("Capturing rotated activations...")
    rotated = _capture_activation_scores(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device=device,
        rotation_state=rotation_state,
        layers=layers,
        activation_points=activation_points,
        texts=texts,
        max_length=max_length,
    )

    rows: list[dict[str, object]] = []
    for key in sorted(base):
        rows.append({"point": key, **_summarize_change(base[key], rotated[key])})
    return rows


def _print_top_weight_rows(rows: Sequence[Mapping[str, object]], limit: int) -> None:
    ranked = sorted(rows, key=lambda row: float(row["ratio_top1pct_share"]), reverse=True)
    print("\nTop rotated/base increases in weight top-1%-channel share:")
    for row in ranked[:limit]:
        print(
            f"L{int(row['layer']):02d} {str(row['module']):9s} {str(row['axis']):6s} "
            f"top1 {float(row['base_top1pct_share']):.3f}->{float(row['top1pct_share']):.3f} "
            f"ratio={float(row['ratio_top1pct_share']):.2f} "
            f"max/mean {float(row['base_max_over_mean']):.2f}->{float(row['max_over_mean']):.2f} "
            f"eff_frac {float(row['base_effective_frac']):.2f}->{float(row['effective_frac']):.2f}"
        )


def _print_activation_rows(rows: Sequence[Mapping[str, object]]) -> None:
    print("\nActivation channel concentration, rotated vs. base:")
    for row in rows:
        print(
            f"{str(row['point']):28s} "
            f"top1 {float(row['base_top1pct_share']):.3f}->{float(row['top1pct_share']):.3f} "
            f"ratio={float(row['ratio_top1pct_share']):.2f} "
            f"max/mean {float(row['base_max_over_mean']):.2f}->{float(row['max_over_mean']):.2f} "
            f"eff_frac {float(row['base_effective_frac']):.2f}->{float(row['effective_frac']):.2f}"
        )


def _print_positional_rows(rows: Sequence[Mapping[str, object]], label: str, limit: int) -> None:
    ranked = sorted(
        rows,
        key=lambda row: abs(float(row["index_center"]) - float(row["base_index_center"])),
        reverse=True,
    )
    print(f"\nLargest positional channel shifts ({label}):")
    for row in ranked[:limit]:
        name = (
            f"L{int(row['layer']):02d} {str(row['module']):9s} {str(row['axis']):6s}"
            if "layer" in row
            else str(row["point"])
        )
        shift = float(row["index_center"]) - float(row["base_index_center"])
        top_shift = float(row["top1pct_index_center"]) - float(row["base_top1pct_index_center"])
        print(
            f"{name:28s} center {float(row['base_index_center']):.3f}->{float(row['index_center']):.3f} "
            f"shift={shift:+.3f} top1_center "
            f"{float(row['base_top1pct_index_center']):.3f}->{float(row['top1pct_index_center']):.3f} "
            f"top_shift={top_shift:+.3f} low_half "
            f"{float(row['base_low_half_share']):.3f}->{float(row['low_half_share']):.3f} "
            f"corr {float(row['base_index_corr']):+.3f}->{float(row['index_corr']):+.3f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/figures/channel_concentration.json")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float16", "bfloat16", "float32", "float64"])
    parser.add_argument("--device", default="cpu", help="Use cpu for reproducible low-memory analysis; use cuda if desired.")
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 20, 21])
    parser.add_argument("--modules", nargs="+", default=list(DEFAULT_MODULES), choices=list(DEFAULT_MODULES))
    parser.add_argument(
        "--activation-points",
        nargs="+",
        default=list(DEFAULT_ACTIVATION_POINTS),
        choices=list(DEFAULT_ACTIVATION_POINTS),
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--top", type=int, default=12)
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

    print(f"Loading model for weight analysis: {model_name}")
    model, _ = load_model_and_tokenizer(model_name=model_name, device="cpu", torch_dtype=torch_dtype)
    if not is_llama_like_model(model):
        raise ValueError(f"{model_name} does not expose a LLaMA-style layout.")
    prepare_model_for_rotation(model)
    weight_rows = _analyze_weights(model, rotation_state, args.modules)
    del model
    gc.collect()

    activation_rows = _analyze_activations(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device=device,
        rotation_state=rotation_state,
        layers=args.layers,
        activation_points=args.activation_points,
        texts=DEFAULT_TEXTS,
        max_length=args.max_length,
    )

    _print_top_weight_rows(weight_rows, args.top)
    _print_activation_rows(activation_rows)
    _print_positional_rows(weight_rows, "weights", args.top)
    _print_positional_rows(activation_rows, "activations", args.top)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "checkpoint": str(checkpoint),
                "weights": weight_rows,
                "activations": activation_rows,
            },
            f,
            indent=2,
        )
    print(f"\nSaved channel concentration analysis to {args.output}")


if __name__ == "__main__":
    main()
