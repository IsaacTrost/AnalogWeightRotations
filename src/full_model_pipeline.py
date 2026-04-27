import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

# Add the repo root so the CLI can be run directly from the repository.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analog_llama import prepare_analog_model
from src.hardware_configs import supported_hardware_presets
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
from src.runtime_rotation import enable_runtime_attention_rotations


@dataclass
class PipelineConfig:
    """Bundle the knobs used by the float-first LLaMA rotation pipeline."""

    model_name: str = DEFAULT_MODEL_NAME
    rotate_mode: str = "random"
    r2_mode: Optional[str] = None
    seed: int = 0
    r2_seed_offset: int = 1
    rotation_backend: str = "static"
    max_length: int = 128
    texts: Optional[Sequence[str]] = None
    torch_dtype: Optional[torch.dtype] = torch.float32
    prepare_model: bool = True
    convert_analog: bool = False
    analog_targets: Sequence[str] = ("down_proj",)
    hardware_preset: str = "ideal_analog"


def summarize_rotation_state(rotation_state: dict) -> dict:
    """Split the checkpoint-shaped rotation state into readable `R1` and `R2` summaries."""
    metadata = rotation_state["metadata"]
    r1 = rotation_state["R1"]
    r2_layers = rotation_state["layers"]
    r2_errors = {name: orthogonality_error(r2) for name, r2 in r2_layers.items()}
    r2_error_values = list(r2_errors.values())
    r2_shapes = sorted({tuple(r2.shape) for r2 in r2_layers.values()})

    return {
        "R1": {
            "mode": metadata["rotate_mode"],
            "shape": tuple(r1.shape),
            "orthogonality_error": orthogonality_error(r1),
        },
        "R2": {
            "mode": metadata["r2_mode"],
            "count": len(r2_layers),
            "head_dim": metadata["head_dim"],
            "matrix_shapes": r2_shapes,
            "mean_orthogonality_error": (
                sum(r2_error_values) / len(r2_error_values) if r2_error_values else 0.0
            ),
            "min_orthogonality_error": min(r2_error_values) if r2_error_values else 0.0,
            "max_orthogonality_error": max(r2_error_values) if r2_error_values else 0.0,
        },
    }


def run_pipeline(config: PipelineConfig) -> dict:
    """Execute the staged SpinQuant-style flow on a LLaMA-like checkpoint."""
    if config.rotation_backend not in {"static", "runtime"}:
        raise ValueError(f"Unsupported rotation backend: {config.rotation_backend}")

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

    if config.rotation_backend == "static":
        rotation_state = rotate_model(
            model,
            rotate_mode=config.rotate_mode,
            seed=config.seed,
            r2_mode=config.r2_mode,
            r2_seed_offset=config.r2_seed_offset,
        )
    else:
        rotation_state = enable_runtime_attention_rotations(
            model,
            rotate_mode=config.rotate_mode,
            seed=config.seed,
            r2_mode=config.r2_mode,
            r2_seed_offset=config.r2_seed_offset,
        )

    rotated = run_verification_forward(
        model,
        tokenizer,
        texts=texts,
        max_length=config.max_length,
    )
    rotation_summary = summarize_rotation_state(rotation_state)

    results = {
        "model_name": config.model_name,
        "rotation_backend": config.rotation_backend,
        "rotate_mode": config.rotate_mode,
        "r2_mode": config.r2_mode or config.rotate_mode,
        "rotation_state": rotation_state,
        "rotation_summary": rotation_summary,
        "r1_rotation": rotation_summary["R1"],
        "r2_rotation": rotation_summary["R2"],
        "prep_equivalence": compare_verification_runs(baseline, prepared),
        "rotation_equivalence": compare_verification_runs(prepared, rotated),
        "float_equivalence": compare_verification_runs(baseline, rotated),
        "analog_targets": [],
        "hardware_preset": None,
    }

    if config.convert_analog:
        if config.rotation_backend != "static":
            raise ValueError("Analog conversion currently supports only the static rotation backend.")
        converted = prepare_analog_model(
            model, 
            target_suffixes=config.analog_targets,
            hardware_preset=config.hardware_preset,
        )
        analog_outputs = run_verification_forward(
            model,
            tokenizer,
            texts=texts,
            max_length=config.max_length,
        )
        results["analog_targets"] = converted
        results["hardware_preset"] = config.hardware_preset

        # This compares original float baseline to analog-after-rotation.
        results["baseline_to_analog_comparison"] = compare_verification_runs(
            baseline, 
            analog_outputs,
        )

        # This better isolates hardware error introduced by analog conversion.
        results["rotated_float_to_analog_comparison"] = compare_verification_runs(
            rotated,
            analog_outputs,
        )

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
    parser.add_argument(
        "--r2-mode",
        default=None,
        choices=["identity", "sign_flip", "random", "hadamard", "block_hadamard", "hadamard_D"],
        help="Per-layer OV rotation mode. Defaults to the same mode used for R1.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--r2-seed-offset",
        type=int,
        default=1,
        help="Offset added before deriving deterministic per-layer R2 seeds from the main seed.",
    )
    parser.add_argument(
        "--rotation-backend",
        default="static",
        choices=["static", "runtime"],
        help="Choose whether rotations are baked into the weights or applied at runtime in attention.",
    )
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
    parser.add_argument(
        "--hardware-preset",
        default="ideal_analog",
        choices=list(supported_hardware_presets()),
        help="AIHWKit hardware-loss preset to use when --convert-analog is enabled.",
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

    def print_rotation_summary(label: str, metrics: dict) -> None:
        """Render one rotation family summary so `R1` and `R2` are easy to distinguish."""
        summary = ", ".join(f"{key}={value}" for key, value in metrics.items())
        print(f"{label}: {summary}")

    args = build_arg_parser().parse_args()
    results = run_pipeline(
        PipelineConfig(
            model_name=args.model_name,
            rotate_mode=args.rotate_mode,
            r2_mode=args.r2_mode,
            seed=args.seed,
            r2_seed_offset=args.r2_seed_offset,
            rotation_backend=args.rotation_backend,
            max_length=args.max_length,
            torch_dtype=resolve_torch_dtype(args.torch_dtype),
            prepare_model=not args.skip_prepare,
            convert_analog=args.convert_analog,
            analog_targets=tuple(args.analog_targets),
            hardware_preset=args.hardware_preset,
        )
    )

    print(f"Model: {results['model_name']}")
    print(f"Model dtype: {args.torch_dtype}")
    print(f"Rotation backend: {results['rotation_backend']}")
    print_rotation_summary("R1 rotation", results["r1_rotation"])
    print_rotation_summary("R2 rotation", results["r2_rotation"])
    print_logits_summary("Prep logits diff", results["prep_equivalence"])
    print_logits_summary("Rotation logits diff", results["rotation_equivalence"])
    print_logits_summary("Overall logits diff", results["float_equivalence"])

    if results["analog_targets"]:
        print("Hardware preset:", results["hardware_preset"])
        print("Analog targets:", ", ".join(results["analog_targets"]))
        baseline_metrics  = results["analog_comparison"]["logits"]
        print(
            "Analog logits diff:",
            f"max_abs={baseline_metrics ['max_abs']:.3e}",
            f"mean_abs={baseline_metrics ['mean_abs']:.3e}",
            f"rel_l2={baseline_metrics ['rel_l2']:.3e}",
        )
        isolated_metrics = results["rotated_float_to_analog_comparison"]["logits"]
        print(
            "Rotated float -> analog logits diff:",
            f"max_abs={isolated_metrics['max_abs']:.3e}",
            f"mean_abs={isolated_metrics['mean_abs']:.3e}",
            f"rel_l2={isolated_metrics['rel_l2']:.3e}",
        )


if __name__ == "__main__":
    main()
