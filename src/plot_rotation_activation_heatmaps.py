"""Plot FFN activation heatmaps under base/generated/trained rotation regimes.

Examples:
  python src/plot_rotation_activation_heatmaps.py \
      --config configs/train_full_hadamard_d_ir0p5_bits8_steps30.json \
      --layer-idx 3 \
      --output results/figures/layer3_ffn_activation_heatmaps.png

  python src/plot_rotation_activation_heatmaps.py \
      --config configs/train_full_hadamard_d_ir0p5_bits8_steps30.json \
      --layer-idx 3 \
      --ffn-point down_proj_input \
      --output results/figures/layer3_down_proj_input_heatmaps.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import TwoSlopeNorm
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


def _display_array(activation: torch.Tensor, max_display_elements: int) -> tuple[np.ndarray, int]:
    array = activation.detach().to(device="cpu", dtype=torch.float32).numpy()
    if array.ndim == 3:
        array = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
    elif array.ndim != 2:
        array = array.reshape(-1, array.shape[-1])

    if max_display_elements <= 0 or array.size <= max_display_elements:
        return array, 1

    stride = max(1, math.ceil(math.sqrt(array.size / max_display_elements)))
    return array[::stride, ::stride], stride


def _symmetric_limit(arrays: Sequence[np.ndarray], percentile: float) -> float:
    values = np.concatenate([np.ravel(array[np.isfinite(array)]) for array in arrays if array.size])
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(values), percentile))
    return limit if limit > 0 else 1.0


def _plot_heatmaps(
    images: Mapping[str, tuple[np.ndarray, int, tuple[int, ...]]],
    output: str,
    *,
    layer_idx: int,
    ffn_point: str,
    cmap: str,
    percentile: float,
    dpi: int,
) -> None:
    variants = list(images)
    arrays = [images[variant][0] for variant in variants]
    limit = _symmetric_limit(arrays, percentile)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(
        nrows=len(variants),
        ncols=1,
        figsize=(10, 2.7 * len(variants)),
        squeeze=False,
        constrained_layout=True,
    )

    last_image = None
    max_stride = 1
    for row_idx, variant in enumerate(variants):
        array, stride, original_shape = images[variant]
        max_stride = max(max_stride, stride)
        axis = axes[row_idx][0]
        last_image = axis.imshow(array, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        axis.set_ylabel(variant, fontsize=10)
        axis.set_xlabel("hidden / intermediate channel")
        axis.set_title(f"captured shape={original_shape}", fontsize=9)
        axis.set_yticks([])

    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.8, label="activation value")

    downsample_note = f" (display stride {max_stride})" if max_stride > 1 else ""
    fig.suptitle(f"Layer {layer_idx} {ffn_point} activation heatmaps{downsample_note}", fontsize=13)
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

    def capture_pre_hook(_module, inputs):
        captured.append(inputs[0].detach().cpu())

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
        description="Plot FFN activations for base, generated random-Hadamard, and checkpoint rotations."
    )
    parser.add_argument("--config", default=None, help="Optional JSON config to read model/checkpoint settings from.")
    parser.add_argument("--checkpoint", default=None, help="Rotation checkpoint containing R1 and R2/layers tensors.")
    parser.add_argument("--output", default="results/figures/rotation_ffn_activation_heatmaps.png")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", "float16", "bfloat16", "float32", "float64"])
    parser.add_argument("--device", default="cpu", help="Device for loading the model; use 'auto' for CUDA if available.")
    parser.add_argument("--prepare-model", default=True, action=argparse.BooleanOptionalAction)
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
    parser.add_argument("--max-display-elements", type=int, default=2_000_000)
    parser.add_argument("--percentile", type=float, default=99.5, help="Percentile used for symmetric color scaling.")
    parser.add_argument("--cmap", default="coolwarm")
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

    checkpoint_state = _load_checkpoint(str(checkpoint))

    variants: dict[str, Optional[Mapping[str, object]]] = {
        "Base": None,
        f"Random Hadamard\n({args.rotation_mode})": "generated",
        "Trained\ncheckpoint": checkpoint_state,
    }

    images: dict[str, tuple[np.ndarray, int, tuple[int, ...]]] = {}
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
        )
        array, stride = _display_array(activation, args.max_display_elements)
        images[variant] = (array, stride, tuple(activation.shape))

    _plot_heatmaps(
        images,
        args.output,
        layer_idx=args.layer_idx,
        ffn_point=args.ffn_point,
        cmap=args.cmap,
        percentile=args.percentile,
        dpi=args.dpi,
    )
    print(f"Saved activation heatmaps to {args.output}")


if __name__ == "__main__":
    main()
