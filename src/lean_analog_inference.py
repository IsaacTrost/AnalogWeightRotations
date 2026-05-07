import argparse
import gc
import math
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
    TORCH_DTYPE_CHOICES,
    is_llama_like_model,
    load_model_and_tokenizer,
    resolve_torch_dtype,
)
from src.llama_prepare import prepare_model_for_rotation
from src.llama_rotation import rotate_model


DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog."
ROTATION_MODES = ("identity", "sign_flip", "random", "hadamard", "block_hadamard", "hadamard_D")


@dataclass
class LeanAnalogConfig:
    """Configuration for memory-conscious static-rotation analog inference."""

    model_name: str = DEFAULT_MODEL_NAME
    rotate_mode: str = "random"
    r2_mode: Optional[str] = None
    seed: int = 0
    r2_seed_offset: int = 1
    max_length: int = 64
    prompts: Sequence[str] = (DEFAULT_PROMPT,)
    torch_dtype: Optional[torch.dtype] = torch.float32
    hardware_preset: str = "ideal_analog"
    analog_targets: Sequence[str] = ("down_proj",)
    run_baseline: bool = False
    keep_full_logits: bool = False
    print_top_k: int = 5
    eval_perplexity: bool = False
    dataset: str = "wikitext-2"
    split: str = "validation"
    batch_size: int = 1
    max_eval_tokens: Optional[int] = 8192
    perplexity_from_prompts: bool = False
    memory_log_interval: int = 0
    memory_log_modules: int = 0


def cleanup_cuda() -> None:
    """Release Python and CUDA caches between memory-heavy inference stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_peak_memory_mb() -> Optional[float]:
    """Return the current CUDA peak allocation in MiB when CUDA is active."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**2)


def _cuda_memory_summary(prefix: str) -> str:
    """Format PyTorch and driver-level CUDA memory counters for debug logs."""
    if not torch.cuda.is_available():
        return f"{prefix}: CUDA unavailable"
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated_mb = torch.cuda.memory_allocated() / (1024**2)
    reserved_mb = torch.cuda.memory_reserved() / (1024**2)
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    free_mb = free_bytes / (1024**2)
    total_mb = total_bytes / (1024**2)
    used_mb = total_mb - free_mb
    return (
        f"{prefix}: "
        f"allocated={allocated_mb:.1f} MiB "
        f"reserved={reserved_mb:.1f} MiB "
        f"peak_allocated={peak_mb:.1f} MiB "
        f"driver_used={used_mb:.1f}/{total_mb:.1f} MiB "
        f"driver_free={free_mb:.1f} MiB"
    )


def _resolve_submodule(root: torch.nn.Module, module_name: str) -> torch.nn.Module:
    """Resolve a dotted module path for registering focused debug hooks."""
    module = root
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def _decode_token(tokenizer, token_id: int) -> str:
    """Decode one token for readable summaries while keeping token IDs authoritative."""
    return tokenizer.decode([token_id])


def build_inputs(tokenizer, prompts: Sequence[str], device: torch.device, max_length: int) -> dict:
    """Tokenize prompts directly so no verification helper needs to capture hidden states."""
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def _load_dataset_text(dataset: str, split: str) -> str:
    """Load the text corpus used for causal-LM perplexity evaluation."""
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Dataset perplexity requires the `datasets` package. Install it in the "
            "runtime environment, or pass --perplexity-from-prompts to evaluate the "
            "provided --prompt text instead."
        ) from exc

    if dataset == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    elif dataset == "wikitext-103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return "\n\n".join(t for t in (ex["text"] for ex in ds) if t.strip())


def build_packed_token_chunks(
    tokenizer,
    *,
    dataset: str,
    split: str,
    max_length: int,
    max_eval_tokens: Optional[int],
) -> tuple[torch.Tensor, int]:
    """Tokenize a dataset split into fixed-length CPU chunks for lean perplexity."""
    text = _load_dataset_text(dataset, split)
    return build_packed_text_chunks(
        tokenizer,
        text,
        max_length=max_length,
        max_eval_tokens=max_eval_tokens,
    )


def build_packed_text_chunks(
    tokenizer,
    text: str,
    *,
    max_length: int,
    max_eval_tokens: Optional[int],
) -> tuple[torch.Tensor, int]:
    """Tokenize text into fixed-length CPU chunks for causal-LM perplexity."""
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
    return chunks, int(n_chunks * max_length)


def evaluate_perplexity(
    model: torch.nn.Module,
    chunks: torch.Tensor,
    *,
    batch_size: int,
    memory_log_interval: int = 0,
    memory_log_label: str = "perplexity",
    memory_log_module_names: Sequence[str] = (),
    memory_log_module_batches: int = 0,
) -> dict:
    """Run causal-LM NLL over CPU token chunks while retaining only scalar metrics."""
    total_nll = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    model.eval()
    total_chunks = chunks.shape[0]
    hook_state = {"batch_index": -1}
    hooks = []

    if memory_log_module_batches > 0:
        for module_name in memory_log_module_names:
            module = _resolve_submodule(model, module_name)

            def pre_hook(_module, _inputs, name=module_name):
                if hook_state["batch_index"] < memory_log_module_batches:
                    print(_cuda_memory_summary(f"{memory_log_label} {name} before"))

            def post_hook(_module, _inputs, _output, name=module_name):
                if hook_state["batch_index"] < memory_log_module_batches:
                    print(_cuda_memory_summary(f"{memory_log_label} {name} after"))

            hooks.append(module.register_forward_pre_hook(pre_hook))
            hooks.append(module.register_forward_hook(post_hook))

    try:
        with torch.inference_mode():
            for start in range(0, chunks.shape[0], batch_size):
                batch_index = start // batch_size
                hook_state["batch_index"] = batch_index
                should_log_memory = (
                    memory_log_interval > 0
                    and (batch_index % memory_log_interval == 0 or start + batch_size >= total_chunks)
                )
                if should_log_memory:
                    print(_cuda_memory_summary(f"{memory_log_label} batch {batch_index} before"))
                input_ids = chunks[start : start + batch_size].to(device)
                attention_mask = torch.ones_like(input_ids)
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                        use_cache=False,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(_cuda_memory_summary(f"{memory_log_label} batch {batch_index} OOM"))
                    raise
                current_batch, seq_len = input_ids.shape
                target_tokens = current_batch * max(seq_len - 1, 0)
                total_nll += float(outputs.loss.detach()) * target_tokens
                total_tokens += target_tokens
                del outputs, input_ids, attention_mask
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                if should_log_memory:
                    print(_cuda_memory_summary(f"{memory_log_label} batch {batch_index} after"))
    finally:
        for hook in hooks:
            hook.remove()

    if total_tokens == 0:
        raise ValueError("No target tokens were evaluated.")
    mean_nll = total_nll / total_tokens
    cleanup_cuda()
    return {
        "nll": mean_nll,
        "ppl": math.exp(mean_nll) if mean_nll < 100 else float("inf"),
        "tokens": total_tokens,
    }


def run_final_logits_forward(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
    *,
    keep_full_logits: bool = False,
) -> dict:
    """Run inference and retain only final-token logits unless full logits are requested."""
    device = next(model.parameters()).device
    inputs = build_inputs(tokenizer, prompts, device, max_length)
    attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
    final_positions = attention_mask.sum(dim=-1).sub(1).clamp(min=0)
    batch_indices = torch.arange(inputs["input_ids"].shape[0], device=device)

    with torch.inference_mode():
        outputs = model(**inputs)
        final_logits = outputs.logits[batch_indices, final_positions, :].detach().cpu().contiguous()
        full_logits = outputs.logits.detach().cpu() if keep_full_logits else None

    result = {
        "input_ids": inputs["input_ids"].detach().cpu(),
        "attention_mask": attention_mask.detach().cpu(),
        "final_logits": final_logits,
        "next_token_ids": final_logits.argmax(dim=-1),
    }
    if keep_full_logits:
        result["logits"] = full_logits

    del outputs, inputs, attention_mask, final_positions, batch_indices, full_logits
    cleanup_cuda()
    return result


def _load_perplexity_chunks(config: LeanAnalogConfig, tokenizer) -> tuple[torch.Tensor, int]:
    """Build shared packed chunks once so baseline and analog evaluate identical tokens."""
    if config.perplexity_from_prompts:
        return build_packed_text_chunks(
            tokenizer,
            "\n\n".join(config.prompts),
            max_length=config.max_length,
            max_eval_tokens=config.max_eval_tokens,
        )
    return build_packed_token_chunks(
        tokenizer,
        dataset=config.dataset,
        split=config.split,
        max_length=config.max_length,
        max_eval_tokens=config.max_eval_tokens,
    )


def compare_final_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Compare final-token logits with the same scale-aware metrics used elsewhere."""
    delta = (candidate - reference).float()
    reference_float = reference.float()
    reference_norm = torch.linalg.norm(reference_float)
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rel_l2": (torch.linalg.norm(delta) / (reference_norm + 1e-12)).item(),
        "next_token_match": bool(torch.equal(reference.argmax(dim=-1), candidate.argmax(dim=-1))),
    }


def top_k_tokens(logits: torch.Tensor, tokenizer, k: int) -> list[list[dict]]:
    """Return top-k final-token candidates for each prompt without storing extra model outputs."""
    if k <= 0:
        return []
    values, indices = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1)
    rows = []
    for token_values, token_indices in zip(values, indices):
        rows.append(
            [
                {
                    "token_id": int(token_id),
                    "token": _decode_token(tokenizer, int(token_id)),
                    "logit": float(logit),
                }
                for logit, token_id in zip(token_values, token_indices)
            ]
        )
    return rows


def run_lean_analog_inference(config: LeanAnalogConfig) -> dict:
    """Load, rotate, optionally baseline, convert to analog, and run final-logit inference."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model_name,
        torch_dtype=config.torch_dtype,
    )
    if config.memory_log_interval > 0:
        print(_cuda_memory_summary("after model load"))
    if not is_llama_like_model(model):
        raise ValueError(f"Model {config.model_name} does not expose a LLaMA-style module layout.")

    prepare_model_for_rotation(model)
    rotation_state = rotate_model(
        model,
        rotate_mode=config.rotate_mode,
        seed=config.seed,
        r2_mode=config.r2_mode,
        r2_seed_offset=config.r2_seed_offset,
    )
    rotation_metadata = dict(rotation_state["metadata"])
    del rotation_state
    cleanup_cuda()
    if config.memory_log_interval > 0:
        print(_cuda_memory_summary("after rotation"))

    ppl_chunks = None
    loaded_ppl_tokens = None
    if config.eval_perplexity:
        ppl_chunks, loaded_ppl_tokens = _load_perplexity_chunks(config, tokenizer)

    baseline = None
    baseline_perplexity = None
    if config.run_baseline:
        baseline = run_final_logits_forward(
            model,
            tokenizer,
            prompts=config.prompts,
            max_length=config.max_length,
            keep_full_logits=config.keep_full_logits,
        )
        if config.eval_perplexity:
            baseline_perplexity = evaluate_perplexity(
                model,
                ppl_chunks,
                batch_size=config.batch_size,
                memory_log_interval=config.memory_log_interval,
                memory_log_label="baseline perplexity",
            )
        cleanup_cuda()

    converted_modules = prepare_analog_model(
        model,
        target_suffixes=config.analog_targets,
        hardware_preset=config.hardware_preset,
    )
    cleanup_cuda()
    if config.memory_log_interval > 0:
        print(_cuda_memory_summary("after analog conversion"))

    analog = run_final_logits_forward(
        model,
        tokenizer,
        prompts=config.prompts,
        max_length=config.max_length,
        keep_full_logits=config.keep_full_logits,
    )
    if config.memory_log_interval > 0:
        print(_cuda_memory_summary("after analog prompt forward"))
    analog_perplexity = None
    if config.eval_perplexity:
        analog_perplexity = evaluate_perplexity(
            model,
            ppl_chunks,
            batch_size=config.batch_size,
            memory_log_interval=config.memory_log_interval,
            memory_log_label="analog perplexity",
            memory_log_module_names=converted_modules,
            memory_log_module_batches=config.memory_log_modules,
        )

    comparison = None
    if baseline is not None:
        comparison = compare_final_logits(baseline["final_logits"], analog["final_logits"])

    return {
        "model_name": config.model_name,
        "torch_dtype": str(config.torch_dtype).replace("torch.", "") if config.torch_dtype else "auto",
        "rotate_mode": config.rotate_mode,
        "r2_mode": config.r2_mode or config.rotate_mode,
        "rotation_metadata": rotation_metadata,
        "hardware_preset": config.hardware_preset,
        "analog_targets": tuple(config.analog_targets),
        "converted_modules": converted_modules,
        "baseline": baseline,
        "analog": analog,
        "comparison": comparison,
        "baseline_perplexity": baseline_perplexity,
        "analog_perplexity": analog_perplexity,
        "perplexity_tokens": loaded_ppl_tokens,
        "analog_top_k": top_k_tokens(analog["final_logits"], tokenizer, config.print_top_k),
        "peak_cuda_memory_mb": _cuda_peak_memory_mb(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Expose a small CLI for final-logit analog inference under tight memory budgets."""
    parser = argparse.ArgumentParser(description="Run lean static-rotation analog inference.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--rotate-mode", default="random", choices=ROTATION_MODES)
    parser.add_argument(
        "--r2-mode",
        default=None,
        choices=ROTATION_MODES,
        help="Per-layer OV rotation mode. Defaults to the same mode used for R1.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r2-seed-offset", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to run. Pass multiple times for a small batch.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["auto", *TORCH_DTYPE_CHOICES.keys()],
        help="Model dtype. Defaults to float32 for AIHWKit compatibility.",
    )
    parser.add_argument(
        "--hardware-preset",
        default="ideal_analog",
        choices=list(supported_hardware_presets()),
        help="AIHWKit hardware-loss preset to use for analog conversion.",
    )
    parser.add_argument(
        "--analog-targets",
        nargs="+",
        default=["down_proj"],
        help="Module suffixes to convert to AnalogLinear.",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Run rotated-float inference before analog conversion and compare final logits.",
    )
    parser.add_argument(
        "--keep-full-logits",
        action="store_true",
        help="Keep full sequence logits on CPU instead of only final-token logits.",
    )
    parser.add_argument("--print-top-k", type=int, default=5)
    parser.add_argument(
        "--eval-perplexity",
        action="store_true",
        help="Evaluate causal-LM perplexity on packed dataset chunks.",
    )
    parser.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-eval-tokens", type=int, default=8192)
    parser.add_argument(
        "--perplexity-from-prompts",
        action="store_true",
        help="Compute perplexity from --prompt text instead of loading a dataset split.",
    )
    parser.add_argument(
        "--memory-log-interval",
        type=int,
        default=0,
        help="Print CUDA memory before/after every N perplexity batches. Use 1 for each batch.",
    )
    parser.add_argument(
        "--memory-log-modules",
        type=int,
        default=0,
        help="Print CUDA memory around each converted analog module for the first N perplexity batches.",
    )
    return parser


def _print_result_summary(results: dict) -> None:
    """Render the memory-light analog inference summary for shell usage."""
    analog = results["analog"]
    comparison = results["comparison"]
    next_ids = [int(token_id) for token_id in analog["next_token_ids"]]

    print(f"Model: {results['model_name']}")
    print(f"Model dtype: {results['torch_dtype']}")
    print(f"Rotation: R1={results['rotate_mode']} R2={results['r2_mode']}")
    print(f"Hardware preset: {results['hardware_preset']}")
    print(f"Analog target suffixes: {', '.join(results['analog_targets'])}")
    print(f"Converted analog modules: {len(results['converted_modules'])}")
    print(f"Analog next token ids: {next_ids}")

    if results["peak_cuda_memory_mb"] is not None:
        print(f"Peak CUDA memory: {results['peak_cuda_memory_mb']:.1f} MiB")

    if comparison is not None:
        print(
            "Final logits diff:",
            f"max_abs={comparison['max_abs']:.3e}",
            f"mean_abs={comparison['mean_abs']:.3e}",
            f"rel_l2={comparison['rel_l2']:.3e}",
        )
        print(f"Final next-token match: {comparison['next_token_match']}")

    if results["analog_perplexity"] is not None:
        print(
            "Analog perplexity:",
            f"nll={results['analog_perplexity']['nll']:.6f}",
            f"ppl={results['analog_perplexity']['ppl']:.4f}",
            f"tokens={results['analog_perplexity']['tokens']}",
        )

    if results["baseline_perplexity"] is not None:
        print(
            "Baseline perplexity:",
            f"nll={results['baseline_perplexity']['nll']:.6f}",
            f"ppl={results['baseline_perplexity']['ppl']:.4f}",
            f"tokens={results['baseline_perplexity']['tokens']}",
        )
        print(
            "Perplexity delta:",
            f"nll={results['analog_perplexity']['nll'] - results['baseline_perplexity']['nll']:.6f}",
            f"ppl={results['analog_perplexity']['ppl'] - results['baseline_perplexity']['ppl']:.4f}",
        )

    for prompt_idx, row in enumerate(results["analog_top_k"]):
        print(f"Analog top tokens for prompt {prompt_idx}:")
        for item in row:
            print(f"  id={item['token_id']} logit={item['logit']:.4f} token={item['token']!r}")


def main() -> None:
    """Parse CLI arguments and run the lean analog inference flow."""
    args = build_arg_parser().parse_args()
    config = LeanAnalogConfig(
        model_name=args.model_name,
        rotate_mode=args.rotate_mode,
        r2_mode=args.r2_mode,
        seed=args.seed,
        r2_seed_offset=args.r2_seed_offset,
        max_length=args.max_length,
        prompts=tuple(args.prompts or [DEFAULT_PROMPT]),
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        hardware_preset=args.hardware_preset,
        analog_targets=tuple(args.analog_targets),
        run_baseline=args.run_baseline,
        keep_full_logits=args.keep_full_logits,
        print_top_k=args.print_top_k,
        eval_perplexity=args.eval_perplexity,
        dataset=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        max_eval_tokens=args.max_eval_tokens,
        perplexity_from_prompts=args.perplexity_from_prompts,
        memory_log_interval=args.memory_log_interval,
        memory_log_modules=args.memory_log_modules,
    )
    results = run_lean_analog_inference(config)
    _print_result_summary(results)


if __name__ == "__main__":
    main()
