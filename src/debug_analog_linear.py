"""Debug AIHWKit AnalogLinear conversion for one LLaMA projection."""
import argparse

import torch

from src.analog_llama import apply_block_hadamard, linear_to_analog
from src.llama_model import DEFAULT_MODEL_NAME, get_default_device, load_model_and_tokenizer
from src.rotation_precision import ROTATION_COMPUTE_DTYPE
from src.rotation_utils import hadamard_matrix, largest_power_of_two_divisor


def _resolve_module(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    module = model
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def _summarize(label: str, tensor: torch.Tensor) -> None:
    t = tensor.detach().float()
    print(
        f"{label}: shape={tuple(t.shape)} "
        f"min={t.min().item():.4e} max={t.max().item():.4e} "
        f"mean={t.mean().item():.4e} std={t.std().item():.4e}"
    )


def _compare_outputs(label: str, reference: torch.Tensor, candidate: torch.Tensor) -> None:
    delta = (candidate - reference).detach().float()
    denom = reference.detach().float().norm() + 1e-12
    print(
        f"{label}: max_abs={delta.abs().max().item():.4e} "
        f"mean_abs={delta.abs().mean().item():.4e} "
        f"rel_l2={(delta.norm() / denom).item():.4e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--module-name", default="model.layers.0.mlp.down_proj")
    parser.add_argument("--hardware-preset", default="ideal_analog")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=4)
    parser.add_argument("--hadamard", action="store_true")
    args = parser.parse_args()

    device = get_default_device()
    model, _ = load_model_and_tokenizer(args.model_name, device=device, torch_dtype=torch.float32)
    linear = _resolve_module(model, args.module_name)
    if not isinstance(linear, torch.nn.Linear):
        raise TypeError(f"{args.module_name} is not a torch.nn.Linear")

    test_linear = torch.nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        device=device,
        dtype=linear.weight.dtype,
    )
    test_linear.load_state_dict(linear.state_dict())

    x = torch.randn(args.batch, args.seq, test_linear.in_features, device=device, dtype=test_linear.weight.dtype)
    if args.hadamard:
        block = largest_power_of_two_divisor(test_linear.in_features)
        h = hadamard_matrix(block, device=device, dtype=ROTATION_COMPUTE_DTYPE)
        test_linear.weight.data = apply_block_hadamard(test_linear.weight.data, h).to(test_linear.weight.dtype)
        x_for_analog = apply_block_hadamard(x, h)
    else:
        x_for_analog = x

    analog = linear_to_analog(test_linear, hardware_preset=args.hardware_preset)
    y_float = test_linear(x_for_analog)
    y_analog = analog(x_for_analog)
    _summarize("weight", test_linear.weight)
    _summarize("input", x_for_analog)
    _compare_outputs("analog_vs_float", y_float, y_analog)

    try:
        analog_weight, analog_bias = analog.get_weights()
        _compare_outputs("weight_roundtrip", test_linear.weight.detach().cpu(), analog_weight.detach().cpu())
        if test_linear.bias is not None and analog_bias is not None:
            _compare_outputs("bias_roundtrip", test_linear.bias.detach().cpu(), analog_bias.detach().cpu())
    except Exception as exc:
        print(f"get_weights failed: {exc}")


if __name__ == "__main__":
    main()
