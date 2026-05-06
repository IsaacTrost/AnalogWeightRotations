"""Evaluate end-to-end perplexity with lightweight PyTorch activation quantization."""
import argparse
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from src.analog_llama import apply_block_hadamard
from src.eval_analog_perplexity import (
    DEFAULT_ANALOG_TARGETS,
    build_packed_token_batches,
    evaluate_perplexity,
)
from src.llama_model import (
    DEFAULT_MODEL_NAME,
    TORCH_DTYPE_CHOICES,
    get_default_device,
    is_llama_like_model,
    load_model_and_tokenizer,
    resolve_torch_dtype,
)
from src.llama_prepare import prepare_model_for_rotation
from src.llama_rotation import bake_rotation_state_into_model, generated_rotation_state, identity_rotation_state
from src.rotation_precision import ROTATION_COMPUTE_DTYPE
from src.rotation_utils import hadamard_matrix, largest_power_of_two_divisor


@dataclass
class FakeQuantConfig:
    model_name: str = DEFAULT_MODEL_NAME
    torch_dtype: Optional[torch.dtype] = torch.float32
    device: Optional[str] = None
    dataset: str = "wikitext-2"
    split: str = "validation"
    max_length: int = 128
    batch_size: int = 1
    max_eval_tokens: Optional[int] = 512
    quant_bits: int = 10
    analog_targets: Sequence[str] = DEFAULT_ANALOG_TARGETS[:-1]
    rotation_mode: str = "hadamard_D"
    r2_mode: Optional[str] = None
    seed: int = 0
    r2_seed_offset: int = 1
    online_hadamards: bool = True


class FakeQuantLinear(torch.nn.Module):
    """Linear with dynamic symmetric input/output activation quantization."""

    def __init__(
        self,
        base_linear: torch.nn.Linear,
        quant_bits: int,
        online_hadamard: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.quant_bits = quant_bits
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        if online_hadamard is not None:
            self.register_buffer("online_hadamard", online_hadamard.detach().clone())
        else:
            self.online_hadamard = None

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias

    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        levels = 2 ** (self.quant_bits - 1) - 1
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / levels
        return (x / scale).round().clamp(-levels, levels) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.base_linear.weight
        if self.online_hadamard is not None:
            x = apply_block_hadamard(x, self.online_hadamard)
            weight = apply_block_hadamard(weight, self.online_hadamard)
        x = self._quantize_activation(x)
        y = F.linear(x, weight, self.base_linear.bias)
        return self._quantize_activation(y)


def _split_parent_name(module_name: str) -> tuple[str, str]:
    if "." not in module_name:
        return "", module_name
    return module_name.rsplit(".", 1)


def _get_submodule(root: torch.nn.Module, module_name: str) -> torch.nn.Module:
    module = root
    if not module_name:
        return module
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def _find_linear_modules(model: torch.nn.Module, suffixes: Sequence[str]) -> list[str]:
    suffix_set = tuple(suffixes)
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.endswith(suffix_set)
    ]


def enable_fake_quant_linears(
    model: torch.nn.Module,
    target_suffixes: Sequence[str],
    quant_bits: int,
    online_hadamards: bool,
) -> list[str]:
    converted = []
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    for module_name in _find_linear_modules(model, target_suffixes):
        parent_name, leaf_name = _split_parent_name(module_name)
        parent = _get_submodule(model, parent_name)
        linear = getattr(parent, leaf_name)
        online_hadamard = None
        if online_hadamards and leaf_name == "o_proj":
            online_hadamard = hadamard_matrix(
                head_dim,
                device=linear.weight.device,
                dtype=ROTATION_COMPUTE_DTYPE,
            )
        elif online_hadamards and leaf_name == "down_proj":
            block_size = largest_power_of_two_divisor(linear.in_features)
            online_hadamard = hadamard_matrix(
                block_size,
                device=linear.weight.device,
                dtype=ROTATION_COMPUTE_DTYPE,
            )
        setattr(parent, leaf_name, FakeQuantLinear(linear, quant_bits, online_hadamard))
        converted.append(module_name)
    return converted


def _load_prepared_model(config: FakeQuantConfig):
    device = config.device or get_default_device()
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model_name,
        device=device,
        torch_dtype=config.torch_dtype,
    )
    if not is_llama_like_model(model):
        raise ValueError(f"{config.model_name} does not expose a LLaMA-style layout.")
    prepare_model_for_rotation(model)
    return model, tokenizer, device


def _build_quant_model(config: FakeQuantConfig, *, rotated: bool):
    model, tokenizer, device = _load_prepared_model(config)
    if rotated:
        state = generated_rotation_state(
            model,
            rotate_mode=config.rotation_mode,
            r2_mode=config.r2_mode,
            seed=config.seed,
            r2_seed_offset=config.r2_seed_offset,
        )
        bake_rotation_state_into_model(model, state)
    else:
        bake_rotation_state_into_model(model, identity_rotation_state(model))
    converted = enable_fake_quant_linears(
        model,
        config.analog_targets,
        quant_bits=config.quant_bits,
        online_hadamards=config.online_hadamards if rotated else False,
    )
    return model, tokenizer, device, converted


def run_evaluation(config: FakeQuantConfig) -> dict:
    model, tokenizer, device = _load_prepared_model(config)
    batches, loaded_tokens = build_packed_token_batches(
        tokenizer,
        dataset=config.dataset,
        split=config.split,
        max_length=config.max_length,
        batch_size=config.batch_size,
        max_eval_tokens=config.max_eval_tokens,
        device=device,
    )
    results = {
        "model_name": config.model_name,
        "dataset": config.dataset,
        "split": config.split,
        "loaded_tokens": loaded_tokens,
        "quant_bits": config.quant_bits,
        "targets": list(config.analog_targets),
        "runs": {
            "float_prepared": evaluate_perplexity(model, batches),
        },
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, _, _, converted = _build_quant_model(config, rotated=False)
    identity = evaluate_perplexity(model, batches)
    identity["converted_layers"] = converted
    results["runs"]["quant_identity"] = identity
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, _, _, converted = _build_quant_model(config, rotated=True)
    rotated = evaluate_perplexity(model, batches)
    rotated["converted_layers"] = converted
    results["runs"]["quant_rotated"] = rotated
    return results


def print_results(results: dict) -> None:
    print(
        f"Evaluated {results['model_name']} on {results['dataset']}:{results['split']} "
        f"({results['loaded_tokens']} loaded tokens), fake {results['quant_bits']}-bit activations"
    )
    print(f"targets={','.join(results['targets'])}")
    print()
    print(f"{'run':<18} {'nll':>12} {'ppl':>12} {'tokens':>12}")
    print("-" * 58)
    for name, metrics in results["runs"].items():
        print(
            f"{name:<18} "
            f"{metrics['nll']:12.6f} "
            f"{metrics['ppl']:12.4f} "
            f"{metrics['tokens']:12d}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PyTorch-only activation quantized perplexity.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", *TORCH_DTYPE_CHOICES.keys()])
    parser.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-eval-tokens", type=int, default=512)
    parser.add_argument("--quant-bits", type=int, default=10)
    parser.add_argument("--rotation-mode", default="hadamard_D")
    parser.add_argument("--r2-mode", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r2-seed-offset", type=int, default=1)
    parser.add_argument("--no-online-hadamards", action="store_true")
    parser.add_argument("--analog-targets", nargs="+", default=list(DEFAULT_ANALOG_TARGETS[:-1]))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = FakeQuantConfig(
        model_name=args.model_name,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        device=None if args.device in (None, "auto") else args.device,
        dataset=args.dataset,
        split=args.split,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_eval_tokens=args.max_eval_tokens,
        quant_bits=args.quant_bits,
        analog_targets=tuple(args.analog_targets),
        rotation_mode=args.rotation_mode,
        r2_mode=args.r2_mode,
        seed=args.seed,
        r2_seed_offset=args.r2_seed_offset,
        online_hadamards=not args.no_online_hadamards,
    )
    print_results(run_evaluation(config))


if __name__ == "__main__":
    main()
