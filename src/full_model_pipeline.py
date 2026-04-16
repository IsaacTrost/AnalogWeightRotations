import argparse
from dataclasses import dataclass
from typing import Optional, Sequence
import torch

from src.analog_llama import convert_llama_linears_to_analog
from src.llama_model import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXTS,
    TORCH_DTYPE_CHOICES,
    is_llama_like_model,
    load_model_and_tokenizer,
    resolve_torch_dtype,
)
from src.llama_prepare import prepare_model_for_rotation
from src.llama_rotation import rotate_model
from src.llama_verify import compare_verification_runs, run_verification_forward
from src.rotation_utils import orthogonality_error


@dataclass
class PipelineConfig:
    """Bundle the knobs used by the float-first LLaMA rotation pipeline."""

    model_name: str = DEFAULT_MODEL_NAME
    rotate_mode: str = "random"
    seed: int = 0
    max_length: int = 128
    texts: Optional[Sequence[str]] = None
    torch_dtype: Optional[torch.dtype] = torch.float32
    prepare_model: bool = True
    convert_analog: bool = False
    analog_targets: Sequence[str] = ("down_proj",)


def run_pipeline(config: PipelineConfig) -> dict:
    """Execute the staged SpinQuant-style flow on a LLaMA-like checkpoint."""
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model_name,
        torch_dtype=config.torch_dtype,
    )
    if not is_llama_like_model(model):
        raise ValueError(f"Model {config.model_name} does not expose a LLaMA-style module layout.")

    texts = config.texts or DEFAULT_TEXTS
    baseline = run_verification_forward(
        model,
        tokenizer,
        texts=texts,
        max_length=config.max_length,
    )

    if config.prepare_model:
        prepare_model_for_rotation(model)

    prepared = run_verification_forward(
        model,
        tokenizer,
        texts=texts,
        max_length=config.max_length,
    )

    rotation_state = rotate_model(
        model,
        rotate_mode=config.rotate_mode,
        seed=config.seed,
    )
    rotated = run_verification_forward(
        model,
        tokenizer,
        texts=texts,
        max_length=config.max_length,
    )

    results = {
        "model_name": config.model_name,
        "rotate_mode": config.rotate_mode,
        "rotation_error": orthogonality_error(rotation_state["R1"]),
        "prep_equivalence": compare_verification_runs(baseline, prepared),
        "rotation_equivalence": compare_verification_runs(prepared, rotated),
        "float_equivalence": compare_verification_runs(baseline, rotated),
        "analog_targets": [],
    }

    if config.convert_analog:
        converted = convert_llama_linears_to_analog(model, target_suffixes=config.analog_targets)
        analog_outputs = run_verification_forward(
            model,
            tokenizer,
            texts=texts,
            max_length=config.max_length,
        )
        results["analog_targets"] = converted
        results["analog_comparison"] = compare_verification_runs(baseline, analog_outputs)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Expose a simple CLI for running the float-first pipeline from the repo root."""
    parser = argparse.ArgumentParser(description="Run the LLaMA full-model rotation pipeline.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--rotate-mode",
        default="random",
        choices=["identity", "sign_flip", "random", "hadamard", "block_hadamard", "hadamard_D"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["auto", *TORCH_DTYPE_CHOICES.keys()],
        help="Model weight dtype for float-side verification. Use float32 for the cleanest equivalence check.",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--convert-analog", action="store_true")
    parser.add_argument(
        "--analog-targets",
        nargs="+",
        default=["down_proj"],
        help="Module suffixes to swap to AnalogLinear when the analog stage is enabled.",
    )
    return parser


def main() -> None:
    """Run the pipeline and print the key verification metrics in a readable form."""
    def print_logits_summary(label: str, metrics: dict) -> None:
        """Render the top-line logits comparison and argmax stability for one stage."""
        logits = metrics["logits"]
        print(
            f"{label}:",
            f"max_abs={logits['max_abs']:.3e}",
            f"mean_abs={logits['mean_abs']:.3e}",
            f"rel_l2={logits['rel_l2']:.3e}",
        )
        print(f"{label} next-token match: {metrics['next_token_match']}")

    args = build_arg_parser().parse_args()
    results = run_pipeline(
        PipelineConfig(
            model_name=args.model_name,
            rotate_mode=args.rotate_mode,
            seed=args.seed,
            max_length=args.max_length,
            torch_dtype=resolve_torch_dtype(args.torch_dtype),
            prepare_model=not args.skip_prepare,
            convert_analog=args.convert_analog,
            analog_targets=tuple(args.analog_targets),
        )
    )

    print(f"Model: {results['model_name']}")
    print(f"Rotation mode: {results['rotate_mode']}")
    print(f"Model dtype: {args.torch_dtype}")
    print(f"Orthogonality error: {results['rotation_error']:.3e}")
    print_logits_summary("Prep logits diff", results["prep_equivalence"])
    print_logits_summary("Rotation logits diff", results["rotation_equivalence"])
    print_logits_summary("Overall logits diff", results["float_equivalence"])

    if results["analog_targets"]:
        analog_metrics = results["analog_comparison"]["logits"]
        print("Analog targets:", ", ".join(results["analog_targets"]))
        print(
            "Analog logits diff:",
            f"max_abs={analog_metrics['max_abs']:.3e}",
            f"mean_abs={analog_metrics['mean_abs']:.3e}",
            f"rel_l2={analog_metrics['rel_l2']:.3e}",
        )


if __name__ == "__main__":
    main()
