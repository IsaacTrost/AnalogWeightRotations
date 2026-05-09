"""Evaluate a trained rotation checkpoint against the identity baseline.

Tests:
  1. Task loss (LM perplexity) stays identical — invariance check.
  2. Simulated INT4 quantization error on effective weights.
  3. Setpoint repulsion score at 0.0 — fraction of weights within sigma of zero.

Usage:
    python -m src.eval_rotation --checkpoint path/to/checkpoint.pt
    python -m src.eval_rotation  # identity baseline only
"""
import argparse
from typing import Optional

import torch

from src.llama_model import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXTS,
    build_inputs,
    get_default_device,
    load_model_and_tokenizer,
)
from src.llama_prepare import prepare_model_for_rotation
from src.rotation_losses import setpoint_repulsion_loss, simulated_quant_error
from src.runtime_rotation import (
    RotationParameters,
    RuntimeRotatedLinear,
    enable_runtime_attention_rotations,
)

TARGET_SUFFIXES = ("down_proj", "up_proj", "gate_proj")


def _collect_effective_weights(model):
    return [
        module.effective_weight()
        for name, module in model.named_modules()
        if isinstance(module, RuntimeRotatedLinear)
        and name.endswith(TARGET_SUFFIXES)
    ]


def _lm_loss(model, tokenizer, device: str, max_length: int = 128) -> float:
    text = DEFAULT_TEXTS[0]
    enc = build_inputs(tokenizer, texts=[text], device=device, max_length=max_length)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    labels = input_ids.clone()
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return float(out.loss)


def evaluate(
    model_name: str = DEFAULT_MODEL_NAME,
    checkpoint_path: Optional[str] = None,
    num_bits: int = 4,
    sigma: float = 0.05,
) -> dict:
    device = get_default_device()
    results = {}

    for label, rotate_mode, ckpt in [
        ("identity", "identity", None),
        ("trained" if checkpoint_path else "random", "random", checkpoint_path),
    ]:
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name, device=device, torch_dtype=torch.float32
        )
        model.eval()
        prepare_model_for_rotation(model)

        if ckpt is not None:
            state = torch.load(ckpt, map_location=device)
            rotation_params = RotationParameters.for_model(model, rotate_mode="random")
            rotation_params.R1 = torch.nn.Parameter(state["R1"].to(device))
            for k, v in state.get("R2", {}).items():
                rotation_params.layer_R2[k] = torch.nn.Parameter(v.to(device))
            enable_runtime_attention_rotations(model, rotation_parameters=rotation_params)
        else:
            enable_runtime_attention_rotations(model, rotate_mode=rotate_mode, seed=0)

        task_loss = _lm_loss(model, tokenizer, device)

        with torch.no_grad():
            weights = _collect_effective_weights(model)
            q_err = float(simulated_quant_error(weights, num_bits=num_bits))
            sp_loss = float(
                setpoint_repulsion_loss(weights, set_points=[0.0], sigma=sigma, kind="gaussian")
            )
            # fraction of weight entries within 2*sigma of zero
            all_w = torch.cat([w.flatten() for w in weights])
            near_zero_frac = float((all_w.abs() < 2 * sigma).float().mean())

        results[label] = {
            "task_loss": task_loss,
            "quant_error": q_err,
            "setpoint_loss": sp_loss,
            "near_zero_frac": near_zero_frac,
        }
        del model

    # Print comparison
    id_r = results["identity"]
    tr_r = results[list(results.keys())[1]]
    trained_label = list(results.keys())[1]

    print(f"\n{'Metric':<22} {'identity':>12} {trained_label:>12} {'delta':>12}")
    print("-" * 60)
    for key, fmt in [
        ("task_loss",      "{:12.6f}"),
        ("quant_error",    "{:12.6f}"),
        ("setpoint_loss",  "{:12.6f}"),
        ("near_zero_frac", "{:12.4%}"),
    ]:
        iv = id_r[key]
        tv = tr_r[key]
        delta = tv - iv
        sign = "+" if delta >= 0 else ""
        print(f"{key:<22} {fmt.format(iv)} {fmt.format(tv)} {sign}{fmt.format(delta).strip()}")

    print()
    print("task_loss invariant:", abs(tr_r["task_loss"] - id_r["task_loss"]) < 0.01)
    print("quant_error improved:", tr_r["quant_error"] < id_r["quant_error"])
    print("near_zero reduced:  ", tr_r["near_zero_frac"] < id_r["near_zero_frac"])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint", default=None, help="Path to saved rotation checkpoint .pt")
    parser.add_argument("--num-bits", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.05)
    args = parser.parse_args()
    evaluate(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        num_bits=args.num_bits,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()
