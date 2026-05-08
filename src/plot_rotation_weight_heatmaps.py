"""Plot base/generated/trained rotated LLaMA layer weight histograms.

Example:
  python src/plot_rotation_weight_heatmaps.py \
      --config configs/train_full_hadamard_d_ir0p5_bits8_steps30.json \
      --layer-idx 0 \
      --output results/figures/layer0_rotation_weight_histograms.png
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.llama_model import DEFAULT_MODEL_NAME, is_llama_like_model, load_model_and_tokenizer, resolve_torch_dtype
from src.llama_prepare import prepare_model_for_rotation
from src.llama_rotation import generated_rotation_state
from src.runtime_rotation import build_runtime_linear_weight_and_bias


DEFAULT_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    parent: str
    attr: str
    apply_r1: str
    apply_r2: Optional[str] = None


MODULE_SPECS = {
    "q_proj": ModuleSpec("q_proj", "self_attn", "q_proj", apply_r1="input"),
    "k_proj": ModuleSpec("k_proj", "self_attn", "k_proj", apply_r1="input"),
    "v_proj": ModuleSpec("v_proj", "self_attn", "v_proj", apply_r1="input", apply_r2="output"),
    "o_proj": ModuleSpec("o_proj", "self_attn", "o_proj", apply_r1="output", apply_r2="input"),
    "gate_proj": ModuleSpec("gate_proj", "mlp", "gate_proj", apply_r1="input"),
    "up_proj": ModuleSpec("up_proj", "mlp", "up_proj", apply_r1="input"),
    "down_proj": ModuleSpec("down_proj", "mlp", "down_proj", apply_r1="output"),
}


def _load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise TypeError(f"Config file {path} must contain a JSON object.")
    return config


def _get_linear(layer: torch.nn.Module, spec: ModuleSpec) -> torch.nn.Linear:
    parent = getattr(layer, spec.parent)
    linear = getattr(parent, spec.attr)
    if not isinstance(linear, torch.nn.Linear):
        raise TypeError(f"{spec.parent}.{spec.attr} is {type(linear).__name__}, expected torch.nn.Linear.")
    return linear


def _coerce_r2_layers(state: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    r2_source = state.get("R2", state.get("layers", {}))
    if not isinstance(r2_source, Mapping):
        raise TypeError("Rotation checkpoint R2/layers entry must be a mapping.")
    return r2_source


def _layer_r2(
    state: Mapping[str, object],
    layer_idx: int,
    head_dim: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    r2_layers = _coerce_r2_layers(state)
    candidates = (
        f"layer_{layer_idx}",
        f"model.layers.{layer_idx}.self_attn.R2",
    )
    for key in candidates:
        value = r2_layers.get(key)
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
    return torch.eye(head_dim, device=device, dtype=torch.float32)


def _rotation_r1(state: Mapping[str, object], *, device: torch.device) -> torch.Tensor:
    r1 = state.get("R1")
    if not isinstance(r1, torch.Tensor):
        raise TypeError("Rotation state must contain an R1 tensor.")
    return r1.to(device=device, dtype=torch.float32)


def _load_checkpoint(path: str) -> Mapping[str, object]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint {path} did not contain a dict-like object.")
    return state


@torch.inference_mode()
def _rotated_weight(
    linear: torch.nn.Linear,
    spec: ModuleSpec,
    state: Mapping[str, object],
    layer_idx: int,
    head_dim: int,
) -> torch.Tensor:
    r1 = _rotation_r1(state, device=linear.weight.device)
    r2 = (
        _layer_r2(state, layer_idx, head_dim, device=linear.weight.device)
        if spec.apply_r2 is not None
        else None
    )
    weight, _ = build_runtime_linear_weight_and_bias(
        linear.weight,
        linear.bias,
        r1=r1,
        apply_r1=spec.apply_r1,
        r2=r2,
        apply_r2=spec.apply_r2,
        head_dim=head_dim if spec.apply_r2 is not None else None,
    )
    return weight.detach()


def _hist_values(weight: torch.Tensor, max_values: int) -> tuple[np.ndarray, int]:
    values = weight.detach().to(device="cpu", dtype=torch.float32).numpy().ravel()
    values = values[np.isfinite(values)]
    if max_values <= 0 or values.size <= max_values:
        return values, 1

    stride = int(np.ceil(values.size / max_values))
    return values[::stride], stride


def _symmetric_limit(arrays: Sequence[np.ndarray], percentile: float) -> float:
    values = np.concatenate([array for array in arrays if array.size])
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(values), percentile))
    return limit if limit > 0 else 1.0


def _plot_histograms(
    histograms: Mapping[str, Mapping[str, tuple[np.ndarray, int, tuple[int, int]]]],
    modules: Sequence[str],
    output: str,
    *,
    layer_idx: int,
    percentile: float,
    bins: int,
    dpi: int,
) -> None:
    variants = list(histograms.keys())
    arrays = [histograms[variant][module][0] for variant in variants for module in modules]
    limit = _symmetric_limit(arrays, percentile)

    fig, axes = plt.subplots(
        ncols=len(modules),
        figsize=(3.4 * len(modules), 3.0),
        squeeze=False,
        constrained_layout=True,
    )

    max_stride = 1
    for col_idx, module in enumerate(modules):
        axis = axes[0][col_idx]
        for variant in variants:
            values, stride, original_shape = histograms[variant][module]
            max_stride = max(max_stride, stride)
            axis.hist(
                values,
                bins=bins,
                range=(-limit, limit),
                density=True,
                histtype="step",
                linewidth=1.5,
                label=variant.replace("\n", " "),
            )
        axis.set_title(f"{module}\n{original_shape[0]}x{original_shape[1]}", fontsize=9)
        axis.set_xlabel("weight value")
        if col_idx == 0:
            axis.set_ylabel("density")
        axis.set_xlim(-limit, limit)
        axis.grid(alpha=0.25)

    axes[0][0].legend(fontsize=8)

    sample_note = f" (sample stride {max_stride})" if max_stride > 1 else ""
    fig.suptitle(f"Layer {layer_idx} weight histograms{sample_note}", fontsize=13)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot base, generated random-Hadamard, and trained-checkpoint layer weight histograms."
    )
    parser.add_argument("--config", default=None, help="Optional JSON config to read model/checkpoint settings from.")
    parser.add_argument("--checkpoint", default=None, help="Rotation checkpoint containing R1 and R2/layers tensors.")
    parser.add_argument("--output", default="results/figures/rotation_weight_histograms.png")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float16", "bfloat16", "float32", "float64"])
    parser.add_argument("--device", default="cpu", help="Device for loading the model; use 'auto' for CUDA if available.")
    parser.add_argument("--prepare-model", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--modules", nargs="+", default=list(DEFAULT_MODULES), choices=sorted(MODULE_SPECS))
    parser.add_argument("--rotation-mode", default="hadamard_D", choices=[
        "identity",
        "sign_flip",
        "random",
        "hadamard",
        "block_hadamard",
        "hadamard_D",
    ])
    parser.add_argument("--r2-mode", default=None, choices=[
        "identity",
        "sign_flip",
        "random",
        "hadamard",
        "block_hadamard",
        "hadamard_D",
    ])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r2-seed-offset", type=int, default=1)
    parser.add_argument("--max-display-elements", type=int, default=2_000_000,
                        help="Maximum values sampled per matrix for histogramming. Use 0 for all values.")
    parser.add_argument("--percentile", type=float, default=99.5, help="Percentile used for symmetric x-axis scaling.")
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = _load_json_config(args.config) if args.config else {}

    checkpoint = args.checkpoint or config.get("checkpoint_path") or config.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Provide --checkpoint or a config with checkpoint_path.")

    model_name = config.get("model_name", args.model_name)
    torch_dtype_name = config.get("torch_dtype", args.torch_dtype)
    device = None if args.device == "auto" else args.device

    torch_dtype = resolve_torch_dtype(torch_dtype_name)
    model, _ = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
    )
    if not is_llama_like_model(model):
        raise ValueError(f"{model_name} does not expose a LLaMA-style layout.")
    if args.prepare_model:
        prepare_model_for_rotation(model)

    if args.layer_idx < 0 or args.layer_idx >= len(model.model.layers):
        raise ValueError(f"--layer-idx must be in [0, {len(model.model.layers) - 1}], got {args.layer_idx}.")

    layer = model.model.layers[args.layer_idx]
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    generated_state = generated_rotation_state(
        model,
        rotate_mode=args.rotation_mode,
        r2_mode=args.r2_mode,
        seed=args.seed,
        r2_seed_offset=args.r2_seed_offset,
    )
    checkpoint_state = _load_checkpoint(checkpoint)

    variants = {
        "Base": None,
        f"Random Hadamard\n({args.rotation_mode})": generated_state,
        "Trained\ncheckpoint": checkpoint_state,
    }
    histograms: dict[str, dict[str, tuple[np.ndarray, int, tuple[int, int]]]] = {}
    for variant, state in variants.items():
        histograms[variant] = {}
        for module in args.modules:
            spec = MODULE_SPECS[module]
            linear = _get_linear(layer, spec)
            weight = linear.weight.detach() if state is None else _rotated_weight(linear, spec, state, args.layer_idx, head_dim)
            values, stride = _hist_values(weight, args.max_display_elements)
            histograms[variant][module] = (values, stride, tuple(weight.shape))

    _plot_histograms(
        histograms,
        args.modules,
        args.output,
        layer_idx=args.layer_idx,
        percentile=args.percentile,
        bins=args.bins,
        dpi=args.dpi,
    )
    print(f"Saved histograms to {args.output}")


if __name__ == "__main__":
    main()
