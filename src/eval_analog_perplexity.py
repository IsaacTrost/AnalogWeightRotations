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
import multiprocessing as mp
import os
import queue
import traceback
from dataclasses import dataclass, fields
from typing import Optional, Sequence

import torch

from src.analog_llama import prepare_analog_model
from src.hardware_configs import build_rpu_config, supported_hardware_presets
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
    rpu_config: Optional[object] = None
    rpu_overrides: Optional[dict] = None
    analog_targets: Sequence[str] = DEFAULT_ANALOG_TARGETS
    online_hadamards: bool = False
    page_analog_tiles: bool = False
    analog_storage_device: str = "cpu"
    analog_execution_device: Optional[str] = None
    cpu_paged_analog_targets: Sequence[str] = ()
    clear_paged_cuda_cache: bool = False
    run_float_prepared: bool = True
    run_analog_identity: bool = True
    run_analog_rotated: bool = True
    use_wandb: bool = False
    wandb_name: Optional[str] = None
    progress_every: int = 1
    json_output_path: Optional[str] = None


def _load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    if not isinstance(raw_config, dict):
        raise TypeError(f"Config file {path} must contain a JSON object.")

    aliases = {
        "checkpoint": "checkpoint_path",
        "json_output": "json_output_path",
    }
    inverted_aliases = {
        "skip_float_prepared": "run_float_prepared",
        "skip_analog_identity": "run_analog_identity",
        "skip_analog_rotated": "run_analog_rotated",
    }
    field_names = {field.name for field in fields(AnalogPerplexityConfig)}
    config = {}
    for raw_key, value in raw_config.items():
        normalized_key = raw_key.replace("-", "_")
        if normalized_key in inverted_aliases:
            key = inverted_aliases[normalized_key]
            value = not value
        else:
            key = aliases.get(normalized_key, normalized_key)
        if key not in field_names:
            valid = ", ".join(sorted(field_names | set(aliases) | set(inverted_aliases)))
            raise ValueError(f"Unknown config key {raw_key!r} in {path}. Valid keys: {valid}")
        config[key] = value

    if "torch_dtype" in config:
        config["torch_dtype"] = resolve_torch_dtype(config["torch_dtype"])
    if config.get("device") == "auto":
        config["device"] = None

    return config


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


def evaluate_perplexity(
    model: torch.nn.Module,
    batches: Sequence[dict],
    *,
    run_label: str = "eval",
    progress_every: int = 1,
) -> dict:
    """Run causal-LM NLL over pre-packed batches and return NLL/perplexity."""
    total_nll = 0.0
    total_tokens = 0
    model.eval()

    with torch.no_grad():
        num_batches = len(batches)
        for batch_idx, batch in enumerate(batches, start=1):
            if progress_every > 0 and (batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == num_batches):
                batch_size, seq_len = batch["input_ids"].shape
                print(
                    f"[{run_label}] batch {batch_idx}/{num_batches} "
                    f"(batch_size={batch_size}, seq_len={seq_len})",
                    flush=True,
                )
            outputs = model(**batch)
            batch_size, seq_len = batch["input_ids"].shape
            target_tokens = batch_size * max(seq_len - 1, 0)
            total_nll += float(outputs.loss.detach()) * target_tokens
            total_tokens += target_tokens
            if progress_every > 0 and (batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == num_batches):
                mean_nll = total_nll / max(total_tokens, 1)
                print(
                    f"[{run_label}] completed {batch_idx}/{num_batches} "
                    f"tokens={total_tokens} nll={mean_nll:.6f}",
                    flush=True,
                )

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
    rpu_config = config.rpu_config
    if rpu_config is None and config.rpu_overrides:
        rpu_config = build_rpu_config(
            config.hardware_preset,
            overrides=config.rpu_overrides,
        )
    converted = prepare_analog_model(
        model,
        target_suffixes=config.analog_targets,
        hardware_preset=config.hardware_preset,
        rpu_config=rpu_config,
        online_hadamards=online_hadamards,
        page_analog_tiles=config.page_analog_tiles,
        analog_storage_device=config.analog_storage_device,
        analog_execution_device=config.analog_execution_device or device,
        cpu_paged_analog_targets=config.cpu_paged_analog_targets,
        clear_paged_cuda_cache=config.clear_paged_cuda_cache,
    )
    return model, tokenizer, device, rotation_state, converted


def _evaluate_single_run(config: AnalogPerplexityConfig, run_label: str) -> tuple[dict, int]:
    """Run one eval slot inside one Python process."""
    if run_label == "float_prepared":
        model, tokenizer, device = _build_float_prepared(config)
        batches, loaded_tokens = build_packed_token_batches(
            tokenizer,
            dataset=config.dataset,
            split=config.split,
            max_length=config.max_length,
            batch_size=config.batch_size,
            max_eval_tokens=config.max_eval_tokens,
            device=device,
        )
        return (
            evaluate_perplexity(
                model,
                batches,
                run_label=run_label,
                progress_every=config.progress_every,
            ),
            loaded_tokens,
        )

    if run_label == "analog_identity":
        model, tokenizer, device, rotation_state, converted = _build_analog_model(
            config,
            force_identity=True,
            online_hadamards=False,
        )
        batches, loaded_tokens = build_packed_token_batches(
            tokenizer,
            dataset=config.dataset,
            split=config.split,
            max_length=config.max_length,
            batch_size=config.batch_size,
            max_eval_tokens=config.max_eval_tokens,
            device=device,
        )
        run = evaluate_perplexity(
            model,
            batches,
            run_label=run_label,
            progress_every=config.progress_every,
        )
        run["converted_layers"] = converted
        run["rotation_mode"] = "identity"
        return run, loaded_tokens

    if run_label == "analog_rotated":
        model, tokenizer, device, rotation_state, converted = _build_analog_model(
            config,
            force_identity=False,
            online_hadamards=config.online_hadamards,
        )
        batches, loaded_tokens = build_packed_token_batches(
            tokenizer,
            dataset=config.dataset,
            split=config.split,
            max_length=config.max_length,
            batch_size=config.batch_size,
            max_eval_tokens=config.max_eval_tokens,
            device=device,
        )
        run = evaluate_perplexity(
            model,
            batches,
            run_label=run_label,
            progress_every=config.progress_every,
        )
        run["converted_layers"] = converted
        run["rotation_mode"] = (
            "learned"
            if config.checkpoint_path and not config.identity_r1_r2
            else rotation_state["metadata"].get("rotate_mode", "checkpoint")
        )
        run["checkpoint_path"] = None if config.identity_r1_r2 else config.checkpoint_path
        return run, loaded_tokens

    raise ValueError(f"Unknown eval run label: {run_label}")


def _run_single_evaluation_child(
    config: AnalogPerplexityConfig,
    run_label: str,
    result_queue,
) -> None:
    try:
        run, loaded_tokens = _evaluate_single_run(config, run_label)
        result_queue.put(
            {
                "ok": True,
                "run_label": run_label,
                "run": run,
                "loaded_tokens": loaded_tokens,
            }
        )
    except BaseException:
        result_queue.put(
            {
                "ok": False,
                "run_label": run_label,
                "traceback": traceback.format_exc(),
            }
        )


def _run_single_evaluation_subprocess(config: AnalogPerplexityConfig, run_label: str) -> tuple[dict, int]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_run_single_evaluation_child,
        args=(config, run_label, result_queue),
    )
    process.start()
    process.join()

    try:
        message = result_queue.get(timeout=5)
    except queue.Empty as exc:
        raise RuntimeError(
            f"{run_label} subprocess exited with code {process.exitcode} without returning results."
        ) from exc

    if not message["ok"]:
        raise RuntimeError(
            f"{run_label} subprocess failed with exit code {process.exitcode}:\n"
            f"{message['traceback']}"
        )
    if process.exitcode != 0:
        raise RuntimeError(f"{run_label} subprocess exited with code {process.exitcode}.")

    return message["run"], message["loaded_tokens"]


def run_evaluation(config: AnalogPerplexityConfig) -> dict:
    """Evaluate requested float and analog baselines with shared dataset settings."""
    # Set rotation_mode to 'learned' if a checkpoint is provided and not using identity
    rotation_mode = (
        "identity" if config.identity_r1_r2 else ("learned" if config.checkpoint_path else config.rotation_mode)
    )
    r2_mode = (
        "identity" if config.identity_r1_r2 else config.r2_mode or config.rotation_mode
    )
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
        "page_analog_tiles": config.page_analog_tiles,
        "analog_storage_device": config.analog_storage_device,
        "analog_execution_device": config.analog_execution_device,
        "cpu_paged_analog_targets": list(config.cpu_paged_analog_targets),
        "rotation_mode": rotation_mode,
        "r2_mode": r2_mode,
        "seed": config.seed,
        "r2_seed_offset": config.r2_seed_offset,
        "runs": {},
    }

    requested_runs = []
    if config.run_float_prepared:
        requested_runs.append("float_prepared")
    if config.run_analog_identity:
        requested_runs.append("analog_identity")
    if config.run_analog_rotated:
        requested_runs.append("analog_rotated")

    for run_label in requested_runs:
        print(f"Starting {run_label} in a subprocess.", flush=True)
        run, loaded_tokens = _run_single_evaluation_subprocess(config, run_label)
        results["loaded_tokens"] = loaded_tokens
        results["runs"][run_label] = run

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
            f"{metrics.get('nll', float('nan')):12.6f} "
            f"{metrics.get('ppl', float('nan')):12.4f} "
            f"{metrics.get('tokens', 0):12d}"
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
            f"{'learned' if results.get('rotation_mode') == 'learned' else results['rotation_mode']}_tok={results.get('loaded_tokens', 0)}"
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
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON config file whose keys match AnalogPerplexityConfig fields.",
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
    parser.add_argument(
        "--page-analog-tiles",
        action="store_true",
        help="Store AIHWKit analog tiles on CPU and move one converted module to the execution device for each forward.",
    )
    parser.add_argument(
        "--analog-storage-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Resident device for analog tiles when --page-analog-tiles is enabled.",
    )
    parser.add_argument(
        "--analog-execution-device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device used for each paged analog forward. Defaults to --device/auto resolved device.",
    )
    parser.add_argument(
        "--cpu-paged-analog-targets",
        nargs="*",
        default=[],
        help="Paged analog target suffixes to execute on CPU even when the rest page to CUDA.",
    )
    parser.add_argument(
        "--clear-paged-cuda-cache",
        action="store_true",
        help="Call torch.cuda.empty_cache() after each paged analog module returns to CPU.",
    )
    parser.add_argument("--skip-float-prepared", action="store_true")
    parser.add_argument("--skip-analog-identity", action="store_true")
    parser.add_argument("--skip-analog-rotated", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print eval progress every N batches. Use 0 to disable.",
    )
    parser.add_argument("--json-output", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.config:
        config = AnalogPerplexityConfig(**_load_json_config(args.config))
    else:
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
            page_analog_tiles=args.page_analog_tiles,
            analog_storage_device=args.analog_storage_device,
            analog_execution_device=args.analog_execution_device,
            cpu_paged_analog_targets=tuple(args.cpu_paged_analog_targets),
            clear_paged_cuda_cache=args.clear_paged_cuda_cache,
            run_float_prepared=not args.skip_float_prepared,
            run_analog_identity=not args.skip_analog_identity,
            run_analog_rotated=not args.skip_analog_rotated,
            use_wandb=args.use_wandb,
            wandb_name=args.wandb_name,
            progress_every=args.progress_every,
            json_output_path=args.json_output,
        )
    results = run_evaluation(config)
    _print_results(results)
    if config.use_wandb:
        _log_wandb(results, run_name=config.wandb_name)
    if config.json_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(config.json_output_path)), exist_ok=True)
        with open(config.json_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
