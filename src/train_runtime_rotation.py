import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

# Add the repo root so the CLI can be run directly from the repository.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration_data import build_calibration_batches, load_calibration_texts
from src.cayley_optimizer import CayleySGDG
from src.full_model_pipeline import summarize_rotation_state
from src.llama_model import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXTS,
    TORCH_DTYPE_CHOICES,
    is_llama_like_model,
    load_model_and_tokenizer,
    resolve_torch_dtype,
)
from src.llama_prepare import prepare_model_for_rotation
from src.llama_verify import compare_verification_runs, run_verification_forward
from src.runtime_rotation import enable_runtime_attention_rotations


@dataclass
class RuntimeRotationTrainingConfig:
    """Bundle the knobs used to optimize runtime `R1` and `R2` on calibration text."""

    model_name: str = DEFAULT_MODEL_NAME
    rotate_mode: str = "random"
    r2_mode: Optional[str] = None
    seed: int = 0
    r2_seed_offset: int = 1
    torch_dtype: Optional[torch.dtype] = torch.float32
    max_length: int = 128
    calibration_texts: Optional[Sequence[str]] = None
    calibration_path: Optional[str] = None
    batch_size: int = 2
    num_steps: int = 10
    learning_rate: float = 1e-3
    momentum: float = 0.0
    eval_every: int = 0
    prepare_model: bool = True
    save_rotation_path: Optional[str] = None


def freeze_non_rotation_parameters(model: torch.nn.Module) -> list[str]:
    """Freeze the backbone and leave only the runtime rotation parameters trainable."""
    trainable_names = []
    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.runtime_rotation_parameters.named_parameters():
        parameter.requires_grad = True
        trainable_names.append(f"runtime_rotation_parameters.{name}")

    return trainable_names


def _verification_texts(config: RuntimeRotationTrainingConfig) -> list[str]:
    """Pick a small deterministic verification slice from the calibration input."""
    texts = load_calibration_texts(texts=config.calibration_texts, data_path=config.calibration_path)
    return texts[: min(4, len(texts))]


def _loss_value(outputs, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Use the model-provided causal-LM loss when available and fall back to manual CE."""
    if getattr(outputs, "loss", None) is not None:
        return outputs.loss

    logits = outputs.logits
    labels = batch["labels"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def summarize_loss_history(history: list[dict[str, float]]) -> dict[str, Any]:
    """Condense the step-wise loss trace into a small training-progress summary."""
    if not history:
        return {
            "num_steps": 0,
            "first_loss": None,
            "last_loss": None,
            "best_loss": None,
            "absolute_improvement": None,
            "relative_improvement": None,
            "monotonic_nonincreasing": None,
        }

    losses = [entry["loss"] for entry in history]
    first_loss = losses[0]
    last_loss = losses[-1]
    absolute_improvement = first_loss - last_loss
    relative_improvement = None if first_loss == 0 else absolute_improvement / first_loss
    monotonic_nonincreasing = all(current <= previous for previous, current in zip(losses, losses[1:]))
    return {
        "num_steps": len(history),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "best_loss": min(losses),
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "monotonic_nonincreasing": monotonic_nonincreasing,
    }


def build_cli_results(results: dict[str, Any], include_rotation_state: bool = False) -> dict[str, Any]:
    """Keep the CLI output readable while still exposing dense tensors on demand."""
    printable_results = dict(results)
    printable_results["history_summary"] = summarize_loss_history(printable_results["history"])
    if not include_rotation_state:
        printable_results.pop("rotation_state", None)
    return printable_results


def run_runtime_rotation_training(config: RuntimeRotationTrainingConfig) -> dict:
    """Optimize runtime `R1` and `R2` with causal-LM loss on calibration batches."""
    if config.num_steps <= 0:
        raise ValueError(f"Number of optimization steps must be positive, got {config.num_steps}.")
    if config.eval_every < 0:
        raise ValueError("Evaluation frequency cannot be negative.")

    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model_name,
        torch_dtype=config.torch_dtype,
    )
    if not is_llama_like_model(model):
        raise ValueError(f"Model {config.model_name} does not expose a LLaMA-style module layout.")

    if config.prepare_model:
        prepare_model_for_rotation(model)

    verification_texts = _verification_texts(config)
    reference_outputs = run_verification_forward(
        model,
        tokenizer,
        texts=verification_texts,
        max_length=config.max_length,
    )

    enable_runtime_attention_rotations(
        model,
        rotate_mode=config.rotate_mode,
        seed=config.seed,
        r2_mode=config.r2_mode,
        r2_seed_offset=config.r2_seed_offset,
    )
    trainable_parameter_names = freeze_non_rotation_parameters(model)

    calibration_batches = build_calibration_batches(
        tokenizer,
        texts=config.calibration_texts,
        data_path=config.calibration_path,
        batch_size=config.batch_size,
        device=next(model.parameters()).device.type,
        max_length=config.max_length,
    )
    if not calibration_batches:
        raise ValueError("Calibration data did not produce any batches.")

    optimizer = CayleySGDG(
        model.runtime_rotation_parameters.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        stiefel=True,
    )
    history = []
    evaluation_history = []
    model.train()

    initial_outputs = run_verification_forward(
        model,
        tokenizer,
        texts=verification_texts,
        max_length=config.max_length,
    )

    for step_idx in range(config.num_steps):
        batch = calibration_batches[step_idx % len(calibration_batches)]
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = _loss_value(outputs, batch)
        loss.backward()
        optimizer.step()

        history.append(
            {
                "step": step_idx + 1,
                "loss": float(loss.detach().cpu().item()),
            }
        )

        if config.eval_every and (step_idx + 1) % config.eval_every == 0:
            model.eval()
            current_outputs = run_verification_forward(
                model,
                tokenizer,
                texts=verification_texts,
                max_length=config.max_length,
            )
            evaluation_history.append(
                {
                    "step": step_idx + 1,
                    "float_equivalence": compare_verification_runs(reference_outputs, current_outputs),
                }
            )
            model.train()

    model.eval()
    final_outputs = run_verification_forward(
        model,
        tokenizer,
        texts=verification_texts,
        max_length=config.max_length,
    )
    rotation_state = model.runtime_rotation_parameters.export_state()
    results = {
        "model_name": config.model_name,
        "rotate_mode": config.rotate_mode,
        "r2_mode": config.r2_mode or config.rotate_mode,
        "trainable_parameter_names": trainable_parameter_names,
        "history": history,
        "initial_equivalence": compare_verification_runs(reference_outputs, initial_outputs),
        "final_equivalence": compare_verification_runs(reference_outputs, final_outputs),
        "evaluation_history": evaluation_history,
        "rotation_state": rotation_state,
        "rotation_summary": summarize_rotation_state(rotation_state),
    }

    if config.save_rotation_path is not None:
        target_path = pathlib.Path(config.save_rotation_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rotation_state, target_path)
        results["saved_rotation_path"] = str(target_path)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Expose a CLI for runtime-rotation training on calibration data."""
    parser = argparse.ArgumentParser(description="Train runtime LLaMA rotation parameters.")
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
    parser.add_argument("--r2-seed-offset", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["auto", *TORCH_DTYPE_CHOICES.keys()],
        help="Model weight dtype used during runtime-rotation training.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--include-rotation-state",
        action="store_true",
        help="Print the full dense rotation tensors in the final JSON output.",
    )
    parser.add_argument("--save-rotation-path", default=None)
    parser.add_argument(
        "--calibration-path",
        default=None,
        help="Optional text or JSONL file used to build calibration batches.",
    )
    parser.add_argument(
        "--calibration-text",
        nargs="*",
        default=None,
        help="Optional inline calibration texts. Defaults to the repo verification texts.",
    )
    return parser


def main() -> None:
    """Run runtime-rotation training and print the key metrics as JSON."""
    args = build_arg_parser().parse_args()
    results = run_runtime_rotation_training(
        RuntimeRotationTrainingConfig(
            model_name=args.model_name,
            rotate_mode=args.rotate_mode,
            r2_mode=args.r2_mode,
            seed=args.seed,
            r2_seed_offset=args.r2_seed_offset,
            torch_dtype=resolve_torch_dtype(args.torch_dtype),
            max_length=args.max_length,
            calibration_texts=tuple(args.calibration_text) if args.calibration_text is not None else None,
            calibration_path=args.calibration_path,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            eval_every=args.eval_every,
            prepare_model=not args.skip_prepare,
            save_rotation_path=args.save_rotation_path,
        )
    )
    printable_results = build_cli_results(
        results,
        include_rotation_state=args.include_rotation_state,
    )
    print(
        json.dumps(
            printable_results,
            indent=2,
            default=lambda value: value.tolist() if isinstance(value, torch.Tensor) else str(value),
        )
    )


if __name__ == "__main__":
    main()
