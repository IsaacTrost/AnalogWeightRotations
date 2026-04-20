from typing import Dict, Iterable, Optional, Sequence

import torch

from src.llama_model import DEFAULT_TEXTS, build_inputs, get_module_names_for_verification


def _capture_module_outputs(
    model: torch.nn.Module,
    module_names: Iterable[str],
) -> tuple[Dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    """Register forward hooks on a small set of modules for equivalence checks."""
    captured: Dict[str, torch.Tensor] = {}
    handles = []
    modules = dict(model.named_modules())

    for module_name in module_names:
        module = modules.get(module_name)
        if module is None:
            continue

        def hook(_module, _inputs, output, name=module_name):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)):
                for value in output:
                    if isinstance(value, torch.Tensor):
                        captured[name] = value.detach().cpu()
                        break

        handles.append(module.register_forward_hook(hook))

    return captured, handles


def run_verification_forward(
    model: torch.nn.Module,
    tokenizer,
    texts: Optional[Sequence[str]] = None,
    max_length: int = 128,
    module_names: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Run a forward pass that captures logits, hidden states, and selected module outputs."""
    inputs = build_inputs(
        tokenizer,
        texts=texts or DEFAULT_TEXTS,
        device=next(model.parameters()).device.type,
        max_length=max_length,
    )

    selected_modules = list(module_names or get_module_names_for_verification(model))
    module_outputs, handles = _capture_module_outputs(model, selected_modules)
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    return {
        "inputs": {name: tensor.detach().cpu() for name, tensor in inputs.items()},
        "logits": outputs.logits.detach().cpu(),
        "hidden_states": tuple(hidden.detach().cpu() for hidden in outputs.hidden_states),
        "module_outputs": module_outputs,
    }


def _tensor_diff(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    """Summarize the distance between two tensors with scale-aware metrics."""
    delta = (candidate - reference).float()
    ref = reference.float()
    ref_norm = torch.linalg.norm(ref)
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rel_l2": (torch.linalg.norm(delta) / (ref_norm + 1e-12)).item(),
    }


def compare_verification_runs(reference: Dict[str, object], candidate: Dict[str, object]) -> Dict[str, object]:
    """Compare two captured forwards and report exactness at several points in the model."""
    logits_ref = reference["logits"]
    logits_candidate = candidate["logits"]
    summary = {
        "logits": _tensor_diff(logits_ref, logits_candidate),
        "next_token_match": bool(
            torch.equal(logits_ref.argmax(dim=-1), logits_candidate.argmax(dim=-1))
        ),
    }

    hidden_metrics = []
    for idx, (hidden_ref, hidden_candidate) in enumerate(
        zip(reference["hidden_states"], candidate["hidden_states"])
    ):
        diff = _tensor_diff(hidden_ref, hidden_candidate)
        diff["layer_index"] = idx
        hidden_metrics.append(diff)
    summary["hidden_states"] = hidden_metrics

    module_metrics = {}
    for name, ref_value in reference["module_outputs"].items():
        if name not in candidate["module_outputs"]:
            continue
        module_metrics[name] = _tensor_diff(ref_value, candidate["module_outputs"][name])
    summary["module_outputs"] = module_metrics
    return summary
