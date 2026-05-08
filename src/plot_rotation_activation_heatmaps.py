"""Plot FFN activation histograms under base/generated/trained rotation regimes.

Examples:
  python src/plot_rotation_activation_heatmaps.py \
      --config configs/train_full_hadamard_d_ir0p5_bits8_steps30.json \
      --layer-idx 3 \
      --output results/figures/layer3_ffn_activation_histograms.png

  python src/plot_rotation_activation_heatmaps.py \
      --config configs/train_full_hadamard_d_ir0p5_bits8_steps30.json \
      --layer-idx 3 \
      --ffn-point down_proj_input \
      --output results/figures/layer3_down_proj_input_histograms.png
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Optional, Sequence

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
from src.llama_rotation import bake_rotation_state_into_model, generated_rotation_state
from src.rotation_utils import hadamard_matrix, largest_power_of_two_divisor


ROTATION_MODES = (
    "identity",
    "sign_flip",
    "random",
    "hadamard",
    "block_hadamard",
    "hadamard_D",
)
FFN_POINTS = ("mlp_input", "down_proj_input")


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


def _load_dataset_text(dataset: str, split: str, max_eval_tokens: int) -> str:
    from datasets import load_dataset

    if dataset == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    elif dataset == "wikitext-103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    chunks: list[str] = []
    approx_chars = max_eval_tokens * 6
    total_chars = 0
    for example in ds:
        text = example["text"].strip()
        if not text:
            continue
        chunks.append(text)
        total_chars += len(text)
        if total_chars >= approx_chars:
            break
    return "\n\n".join(chunks)


def _sample_texts(args: argparse.Namespace, config: Mapping[str, object]) -> list[str]:
    if args.text:
        return [" ".join(args.text)]
    if args.sample_source == "dataset":
        dataset = str(config.get("dataset", args.dataset))
        split = str(config.get("split", args.split))
        return [_load_dataset_text(dataset, split, args.max_eval_tokens)]
    return list(DEFAULT_TEXTS)


def _hist_values(activation: torch.Tensor, max_values: int) -> tuple[np.ndarray, int]:
    values = activation.detach().to(device="cpu", dtype=torch.float32).numpy().ravel()
    values = values[np.isfinite(values)]
    if max_values <= 0 or values.size <= max_values:
        return values, 1

    stride = int(np.ceil(values.size / max_values))
    return values[::stride], stride


def _apply_block_hadamard(x: torch.Tensor, hadamard_block: torch.Tensor) -> torch.Tensor:
    block_size = hadamard_block.shape[0]
    dim = x.shape[-1]
    if dim % block_size != 0:
        raise ValueError(f"Last dim {dim} not divisible by Hadamard block size {block_size}.")
    leading = x.shape[:-1]
    blocks = x.reshape(*leading, dim // block_size, block_size)
    h = hadamard_block.to(device=x.device, dtype=x.dtype)
    return (blocks @ h).reshape(*leading, dim)


def _symmetric_limit(arrays: Sequence[np.ndarray], percentile: float) -> float:
    values = np.concatenate([array for array in arrays if array.size])
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(values), percentile))
    return limit if limit > 0 else 1.0


def _abs_limit(arrays: Sequence[np.ndarray], percentile: float) -> float:
    values = np.concatenate([np.abs(array) for array in arrays if array.size])
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(values, percentile))
    return limit if limit > 0 else 1.0


def _activation_stats(values: np.ndarray) -> dict[str, float]:
    abs_values = np.abs(values)
    if abs_values.size == 0:
        return {"p99": 0.0, "p999": 0.0, "max": 0.0}
    return {
        "p99": float(np.percentile(abs_values, 99.0)),
        "p999": float(np.percentile(abs_values, 99.9)),
        "max": float(abs_values.max()),
    }


def _plot_histograms(
    histograms: Mapping[str, tuple[np.ndarray, int, tuple[int, ...]]],
    output: str,
    *,
    layer_idx: int,
    ffn_point: str,
    percentile: float,
    tail_percentile: float,
    bins: int,
    dpi: int,
) -> None:
    variants = list(histograms)
    arrays = [histograms[variant][0] for variant in variants]
    signed_limit = _symmetric_limit(arrays, percentile)
    tail_limit = _abs_limit(arrays, tail_percentile)

    fig, axes = plt.subplots(ncols=2, figsize=(13, 5.2), constrained_layout=True)
    signed_axis, tail_axis = axes

    max_stride = 1
    shape_notes = []
    for variant in variants:
        values, stride, original_shape = histograms[variant]
        max_stride = max(max_stride, stride)
        shape_notes.append(f"{variant.replace(chr(10), ' ')}: {original_shape}")
        label = variant.replace("\n", " ")
        stats = _activation_stats(values)

        signed_axis.hist(
            values,
            bins=bins,
            range=(-signed_limit, signed_limit),
            density=True,
            histtype="step",
            linewidth=1.8,
            label=label,
        )
        tail_axis.hist(
            np.abs(values),
            bins=bins,
            range=(0.0, tail_limit),
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"{label} p99={stats['p99']:.3g} p99.9={stats['p999']:.3g}",
        )
        tail_axis.axvline(stats["p99"], linewidth=1.0, alpha=0.65)
        tail_axis.axvline(stats["p999"], linewidth=1.0, linestyle="--", alpha=0.65)
        print(
            f"{label}: p99_abs={stats['p99']:.6g} "
            f"p99.9_abs={stats['p999']:.6g} max_abs={stats['max']:.6g}"
        )

    sample_note = f" (sample stride {max_stride})" if max_stride > 1 else ""
    signed_axis.set_title(f"Signed activations, p{percentile:g} x-limit")
    signed_axis.set_xlabel("activation value")
    signed_axis.set_ylabel("density")
    signed_axis.set_xlim(-signed_limit, signed_limit)
    signed_axis.grid(alpha=0.25)
    signed_axis.legend(fontsize=8)
    signed_axis.text(
        0.01,
        0.99,
        "\n".join(shape_notes),
        transform=signed_axis.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    tail_axis.set_title(f"|activation| tails, p{tail_percentile:g} x-limit")
    tail_axis.set_xlabel("|activation value|")
    tail_axis.set_ylabel("density (log)")
    tail_axis.set_xlim(0.0, tail_limit)
    tail_axis.set_yscale("log")
    tail_axis.grid(alpha=0.25)
    tail_axis.legend(fontsize=8)

    fig.suptitle(f"Layer {layer_idx} {ffn_point} activation distributions{sample_note}", fontsize=13)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


@torch.inference_mode()
def _capture_ffn_activation(
    *,
    variant: str,
    rotation_state: Optional[Mapping[str, object]],
    texts: Sequence[str],
    model_name: str,
    torch_dtype: Optional[torch.dtype],
    device: Optional[str],
    layer_idx: int,
    ffn_point: str,
    max_length: int,
    prepare_model: bool,
    online_hadamards: bool,
) -> torch.Tensor:
    print(f"Capturing {variant} activations...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
    )
    if not is_llama_like_model(model):
        raise ValueError(f"{model_name} does not expose a LLaMA-style layout.")
    if layer_idx < 0 or layer_idx >= len(model.model.layers):
        raise ValueError(f"--layer-idx must be in [0, {len(model.model.layers) - 1}], got {layer_idx}.")
    if prepare_model:
        prepare_model_for_rotation(model)
    if rotation_state is not None:
        bake_rotation_state_into_model(model, rotation_state)

    captured: list[torch.Tensor] = []
    layer = model.model.layers[layer_idx]
    target = layer.mlp if ffn_point == "mlp_input" else layer.mlp.down_proj
    r4_hadamard = None
    if rotation_state is not None and online_hadamards and ffn_point == "down_proj_input":
        block_size = largest_power_of_two_divisor(model.config.intermediate_size)
        if block_size < 2:
            raise ValueError(
                f"intermediate_size {model.config.intermediate_size} has no power-of-two factor for R4."
            )
        r4_hadamard = hadamard_matrix(
            block_size,
            device=next(model.parameters()).device.type,
            dtype=torch.float32,
        )
        print(f"  Capturing down_proj_input after online R4 Hadamard block={block_size}.")
    elif rotation_state is not None and online_hadamards and ffn_point == "mlp_input":
        print("  mlp_input includes baked R1/R2 context; R4 applies later at down_proj_input.")

    def capture_pre_hook(_module, inputs):
        activation = inputs[0]
        if r4_hadamard is not None:
            activation = _apply_block_hadamard(activation, r4_hadamard)
        captured.append(activation.detach().cpu())

    handle = target.register_forward_pre_hook(capture_pre_hook)
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
        handle.remove()

    if not captured:
        raise RuntimeError(f"No activation was captured for layer {layer_idx} {ffn_point}.")
    return captured[0]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot FFN activation histograms for base, generated random-Hadamard, and checkpoint rotations."
    )
    parser.add_argument("--config", default=None, help="Optional JSON config to read model/checkpoint settings from.")
    parser.add_argument("--checkpoint", default=None, help="Rotation checkpoint containing R1 and R2/layers tensors.")
    parser.add_argument("--output", default="results/figures/rotation_ffn_activation_histograms.png")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float16", "bfloat16", "float32", "float64"])
    parser.add_argument("--device", default="cpu", help="Device for loading the model; use 'auto' for CUDA if available.")
    parser.add_argument("--prepare-model", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--online-hadamards",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Apply online R4 in the captured down_proj_input for rotated regimes. Defaults to config value, else true.",
    )
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--ffn-point", default="mlp_input", choices=FFN_POINTS)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-source", default="prompt", choices=["prompt", "dataset"])
    parser.add_argument("--text", nargs="+", default=None, help="Prompt text to use instead of the default sample prompts.")
    parser.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-eval-tokens", type=int, default=1024)
    parser.add_argument("--rotation-mode", default="hadamard_D", choices=ROTATION_MODES)
    parser.add_argument("--r2-mode", default=None, choices=ROTATION_MODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r2-seed-offset", type=int, default=1)
    parser.add_argument("--max-display-elements", type=int, default=2_000_000,
                        help="Maximum activation values sampled for histogramming. Use 0 for all values.")
    parser.add_argument("--percentile", type=float, default=99.0, help="Percentile used for signed x-axis scaling.")
    parser.add_argument("--tail-percentile", type=float, default=99.9,
                        help="Percentile used for the absolute-value tail x-axis.")
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = _load_json_config(args.config)

    checkpoint = args.checkpoint or config.get("checkpoint_path") or config.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Provide --checkpoint or a config with checkpoint_path.")

    model_name = str(config.get("model_name", args.model_name))
    torch_dtype_name = str(config.get("torch_dtype", args.torch_dtype))
    torch_dtype = resolve_torch_dtype(torch_dtype_name)
    device = None if args.device == "auto" else args.device
    texts = _sample_texts(args, config)
    online_hadamards = (
        bool(config["online_hadamards"])
        if args.online_hadamards is None and "online_hadamards" in config
        else (True if args.online_hadamards is None else args.online_hadamards)
    )

    checkpoint_state = _load_checkpoint(str(checkpoint))

    variants: dict[str, Optional[Mapping[str, object]]] = {
        "Base": None,
        f"Random Hadamard\n({args.rotation_mode})": "generated",
        "Trained\ncheckpoint": checkpoint_state,
    }

    histograms: dict[str, tuple[np.ndarray, int, tuple[int, ...]]] = {}
    for variant, state in variants.items():
        if state == "generated":
            # Generate against a temporary model so the dimensions match the checkpoint/model exactly.
            temp_model, _ = load_model_and_tokenizer(
                model_name=model_name,
                device=device,
                torch_dtype=torch_dtype,
            )
            try:
                state = generated_rotation_state(
                    temp_model,
                    rotate_mode=args.rotation_mode,
                    r2_mode=args.r2_mode,
                    seed=args.seed,
                    r2_seed_offset=args.r2_seed_offset,
                )
            finally:
                del temp_model

        activation = _capture_ffn_activation(
            variant=variant,
            rotation_state=state,
            texts=texts,
            model_name=model_name,
            torch_dtype=torch_dtype,
            device=device,
            layer_idx=args.layer_idx,
            ffn_point=args.ffn_point,
            max_length=args.max_length,
            prepare_model=args.prepare_model,
            online_hadamards=online_hadamards,
        )
        values, stride = _hist_values(activation, args.max_display_elements)
        histograms[variant] = (values, stride, tuple(activation.shape))

    _plot_histograms(
        histograms,
        args.output,
        layer_idx=args.layer_idx,
        ffn_point=args.ffn_point,
        percentile=args.percentile,
        tail_percentile=args.tail_percentile,
        bins=args.bins,
        dpi=args.dpi,
    )
    print(f"Saved activation histograms to {args.output}")


if __name__ == "__main__":
    main()
