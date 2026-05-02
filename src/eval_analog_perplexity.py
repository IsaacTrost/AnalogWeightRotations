"""Evaluate trained rotations on AIHWKit analog perplexity.

Example:
    python -m src.eval_analog_perplexity \
        --checkpoint checkpoints/full_analog.pt \
        --hardware-preset full_pcm \
        --online-hadamards \
        --max-eval-tokens 8192
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from src.analog_llama import prepare_analog_model
from src.hardware_configs import supported_hardware_presets
from src.llama_model import (
    DEFAULT_MODEL_NAME,
    TORCH_DTYPE_CHOICES,
    get_default_device,
    is_llama_like_model,
    load_model_and_tokenizer,
    resolve_torch_dtype,
)
from src.llama_prepare import prepare_model_for_rotation
from src.llama_rotation import (
    bake_rotation_state_into_model,
    generated_rotation_state,
    identity_rotation_state,
)
from src.wandb_config import WANDB_ENTITY, WANDB_MODE, WANDB_PROJECT


DEFAULT_ANALOG_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
    "lm_head",
)


@dataclass
class AnalogPerplexityConfig:
    model_name: str = DEFAULT_MODEL_NAME
    torch_dtype: Optional[torch.dtype] = torch.float32
    device: Optional[str] = None
    checkpoint_path: Optional[str] = None
    identity_r1_r2: bool = False
    rotation_mode: str = "identity"
    r2_mode: Optional[str] = None
    seed: int = 0
    r2_seed_offset: int = 1
    dataset: str = "wikitext-2"
    split: str = "validation"
    max_length: int = 128
    batch_size: int = 1
    max_eval_tokens: Optional[int] = 8192
    hardware_preset: str = "ideal_analog"
    analog_targets: Sequence[str] = DEFAULT_ANALOG_TARGETS
    online_hadamards: bool = False
    run_float_prepared: bool = True
    run_analog_identity: bool = True
    run_analog_rotated: bool = True
    use_wandb: bool = False
    wandb_name: Optional[str] = None


def _load_dataset_text(dataset: str, split: str) -> str:
    from datasets import load_dataset

    if dataset == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    elif dataset == "wikitext-103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return "\n\n".join(t for t in (ex["text"] for ex in ds) if t.strip())


def build_packed_token_batches(
    tokenizer,
    *,
    dataset: str,
    split: str,
    max_length: int,
    batch_size: int,
    max_eval_tokens: Optional[int],
    device: str,
) -> tuple[list[dict], int]:
    """Tokenize a dataset split into fixed-length causal-LM batches."""
    text = _load_dataset_text(dataset, split)
    saved_max = tokenizer.model_max_length
    tokenizer.model_max_length = 10**9
    try:
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    finally:
        tokenizer.model_max_length = saved_max

    tokens = torch.tensor(ids, dtype=torch.long)
    if max_eval_tokens is not None:
        tokens = tokens[:max_eval_tokens]

    n_chunks = tokens.numel() // max_length
    if n_chunks == 0:
        raise ValueError(
            f"Need at least {max_length} tokens after filtering; got {tokens.numel()}."
        )
    chunks = tokens[: n_chunks * max_length].view(n_chunks, max_length)

    batches = []
    for start in range(0, n_chunks, batch_size):
        batch = chunks[start : start + batch_size].to(device)
        attention_mask = torch.ones_like(batch)
        batches.append(
            {
                "input_ids": batch,
                "attention_mask": attention_mask,
                "labels": batch.clone(),
            }
        )
    return batches, int(n_chunks * max_length)


def evaluate_perplexity(model: torch.nn.Module, batches: Sequence[dict]) -> dict:
    """Run causal-LM NLL over pre-packed batches and return NLL/perplexity."""
    total_nll = 0.0
    total_tokens = 0
    model.eval()

    with torch.no_grad():
        for batch in batches:
            outputs = model(**batch)
            batch_size, seq_len = batch["input_ids"].shape
            target_tokens = batch_size * max(seq_len - 1, 0)
            total_nll += float(outputs.loss.detach()) * target_tokens
            total_tokens += target_tokens

    if total_tokens == 0:
        raise ValueError("No target tokens were evaluated.")
    mean_nll = total_nll / total_tokens
    return {
        "nll": mean_nll,
        "ppl": math.exp(mean_nll) if mean_nll < 100 else float("inf"),
        "tokens": total_tokens,
    }


def _load_rotation_state(config: AnalogPerplexityConfig, model: torch.nn.Module) -> dict:
    if config.identity_r1_r2:
        return identity_rotation_state(model)
    path = config.checkpoint_path
    if path is None:
        if config.rotation_mode == "identity":
            return identity_rotation_state(model)
        return generated_rotation_state(
            model,
            rotate_mode=config.rotation_mode,
            r2_mode=config.r2_mode,
            seed=config.seed,
            r2_seed_offset=config.r2_seed_offset,
        )
    state = torch.load(path, map_location=model.model.embed_tokens.weight.device)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} did not contain a dict.")
    return state


def _load_prepared_model(config: AnalogPerplexityConfig):
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


def _build_float_prepared(config: AnalogPerplexityConfig):
    model, tokenizer, device = _load_prepared_model(config)
    return model, tokenizer, device


def _build_analog_model(
    config: AnalogPerplexityConfig,
    *,
    force_identity: bool,
    online_hadamards: bool,
):
    model, tokenizer, device = _load_prepared_model(config)
    state = identity_rotation_state(model) if force_identity else _load_rotation_state(config, model)
    rotation_state = bake_rotation_state_into_model(model, state)
    converted = prepare_analog_model(
        model,
        target_suffixes=config.analog_targets,
        hardware_preset=config.hardware_preset,
        online_hadamards=online_hadamards,
    )
    return model, tokenizer, device, rotation_state, converted


def run_evaluation(config: AnalogPerplexityConfig) -> dict:
    """Evaluate requested float and analog baselines with shared dataset settings."""
    results = {
        "model_name": config.model_name,
        "dataset": config.dataset,
        "split": config.split,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "max_eval_tokens": config.max_eval_tokens,
        "hardware_preset": config.hardware_preset,
        "analog_targets": list(config.analog_targets),
        "online_hadamards": config.online_hadamards,
        "rotation_mode": "identity" if config.identity_r1_r2 else config.rotation_mode,
        "r2_mode": "identity" if config.identity_r1_r2 else config.r2_mode or config.rotation_mode,
        "seed": config.seed,
        "r2_seed_offset": config.r2_seed_offset,
        "runs": {},
    }

    batches = None

    def get_batches(tokenizer, device):
        nonlocal batches
        if batches is None:
            batches, loaded_tokens = build_packed_token_batches(
                tokenizer,
                dataset=config.dataset,
                split=config.split,
                max_length=config.max_length,
                batch_size=config.batch_size,
                max_eval_tokens=config.max_eval_tokens,
                device=device,
            )
            results["loaded_tokens"] = loaded_tokens
        return batches

    if config.run_float_prepared:
        model, tokenizer, device = _build_float_prepared(config)
        results["runs"]["float_prepared"] = evaluate_perplexity(
            model,
            get_batches(tokenizer, device),
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if config.run_analog_identity:
        model, tokenizer, device, rotation_state, converted = _build_analog_model(
            config,
            force_identity=True,
            online_hadamards=False,
        )
        run = evaluate_perplexity(model, get_batches(tokenizer, device))
        run["converted_layers"] = converted
        run["rotation_mode"] = rotation_state["metadata"]["rotate_mode"]
        results["runs"]["analog_identity"] = run
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if config.run_analog_rotated:
        model, tokenizer, device, rotation_state, converted = _build_analog_model(
            config,
            force_identity=False,
            online_hadamards=config.online_hadamards,
        )
        run = evaluate_perplexity(model, get_batches(tokenizer, device))
        run["converted_layers"] = converted
        run["rotation_mode"] = rotation_state["metadata"]["rotate_mode"]
        run["checkpoint_path"] = None if config.identity_r1_r2 else config.checkpoint_path
        results["runs"]["analog_rotated"] = run
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def _print_results(results: dict) -> None:
    print(
        f"Evaluated {results['model_name']} on {results['dataset']}:{results['split']} "
        f"({results.get('loaded_tokens', 0)} loaded tokens)"
    )
    print(
        f"hardware={results['hardware_preset']} "
        f"rotation={results['rotation_mode']} "
        f"online_hadamards={results['online_hadamards']} "
        f"targets={','.join(results['analog_targets'])}"
    )
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


def _log_wandb(results: dict, run_name: Optional[str] = None) -> None:
    import wandb

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    config = {key: value for key, value in results.items() if key != "runs"}
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        mode=WANDB_MODE,
        name=run_name
        or (
            f"eval_{results['hardware_preset']}_"
            f"{results['rotation_mode']}_tok={results.get('loaded_tokens', 0)}"
        ),
        config=config,
        job_type="analog_perplexity_eval",
    )
    for run_label, metrics in results["runs"].items():
        wandb.log(
            {
                f"{run_label}/nll": metrics["nll"],
                f"{run_label}/ppl": metrics["ppl"],
                f"{run_label}/tokens": metrics["tokens"],
            }
        )
    summary = run.summary
    for run_label, metrics in results["runs"].items():
        summary[f"{run_label}_nll"] = metrics["nll"]
        summary[f"{run_label}_ppl"] = metrics["ppl"]
        summary[f"{run_label}_tokens"] = metrics["tokens"]
    wandb.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate trained R1/R2 rotations on AIHWKit analog perplexity."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Force model and AIHWKit tiles onto a device. Use cpu for full analog models that do not fit in VRAM.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["auto", *TORCH_DTYPE_CHOICES.keys()],
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--identity-r1-r2",
        action="store_true",
        help="Ignore any checkpoint and evaluate identity R1/R2 in the analog_rotated slot.",
    )
    parser.add_argument(
        "--rotation-mode",
        default="identity",
        choices=["identity", "sign_flip", "random", "hadamard", "block_hadamard", "hadamard_D"],
        help="Generated R1/R2 mode to use when --checkpoint is not provided.",
    )
    parser.add_argument(
        "--r2-mode",
        default=None,
        choices=["identity", "sign_flip", "random", "hadamard", "block_hadamard", "hadamard_D"],
        help="Generated per-layer R2 mode. Defaults to --rotation-mode.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r2-seed-offset", type=int, default=1)
    parser.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-eval-tokens", type=int, default=8192)
    parser.add_argument(
        "--hardware-preset",
        default="ideal_analog",
        choices=list(supported_hardware_presets()),
    )
    parser.add_argument(
        "--analog-targets",
        nargs="+",
        default=list(DEFAULT_ANALOG_TARGETS),
        help="Linear module suffixes to convert to AIHWKit AnalogLinear.",
    )
    parser.add_argument(
        "--online-hadamards",
        action="store_true",
        help="Enable R3 on o_proj and R4 on down_proj.",
    )
    parser.add_argument("--skip-float-prepared", action="store_true")
    parser.add_argument("--skip-analog-identity", action="store_true")
    parser.add_argument("--skip-analog-rotated", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--json-output", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = AnalogPerplexityConfig(
        model_name=args.model_name,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        device=None if args.device in (None, "auto") else args.device,
        checkpoint_path=args.checkpoint,
        identity_r1_r2=args.identity_r1_r2,
        rotation_mode=args.rotation_mode,
        r2_mode=args.r2_mode,
        seed=args.seed,
        r2_seed_offset=args.r2_seed_offset,
        dataset=args.dataset,
        split=args.split,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_eval_tokens=args.max_eval_tokens,
        hardware_preset=args.hardware_preset,
        analog_targets=tuple(args.analog_targets),
        online_hadamards=args.online_hadamards,
        run_float_prepared=not args.skip_float_prepared,
        run_analog_identity=not args.skip_analog_identity,
        run_analog_rotated=not args.skip_analog_rotated,
        use_wandb=args.use_wandb,
        wandb_name=args.wandb_name,
    )
    results = run_evaluation(config)
    _print_results(results)
    if config.use_wandb:
        _log_wandb(results, run_name=config.wandb_name)
    if args.json_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
