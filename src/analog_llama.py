import copy
from typing import List, Optional, Sequence

import torch


def _require_aihwkit():
    """Import AIHWKit lazily so the float-only path works without analog deps."""
    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.configs import InferenceRPUConfig

    return AnalogLinear, InferenceRPUConfig


def default_inference_rpu_config():
    """Create the baseline inference config used for the first analog conversion stage."""
    _, InferenceRPUConfig = _require_aihwkit()
    return InferenceRPUConfig()


def _split_parent_name(module_name: str) -> tuple[str, str]:
    """Split a dotted module path into parent path and leaf attribute."""
    if "." not in module_name:
        return "", module_name
    parent_name, leaf_name = module_name.rsplit(".", 1)
    return parent_name, leaf_name


def _get_submodule(root: torch.nn.Module, module_name: str) -> torch.nn.Module:
    """Resolve a dotted module path without relying on newer PyTorch helpers."""
    module = root
    if not module_name:
        return module
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def _copy_linear_to_analog(
    linear: torch.nn.Linear,
    rpu_config,
) -> torch.nn.Module:
    """Mirror a float linear layer into an AnalogLinear module with the same parameters."""
    AnalogLinear, _ = _require_aihwkit()
    analog = AnalogLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        rpu_config=rpu_config,
    )

    weight = linear.weight.detach().float().cpu()
    bias = linear.bias.detach().float().cpu() if linear.bias is not None else None
    for _, tile in analog.named_analog_layers():
        if bias is None:
            tile.set_weights(weight)
        else:
            tile.set_weights(weight, bias)

    analog.to(device=linear.weight.device, dtype=linear.weight.dtype)
    analog.eval()
    return analog


def find_llama_linear_modules(
    model: torch.nn.Module,
    target_suffixes: Sequence[str],
) -> List[str]:
    """Find LLaMA projection modules by their suffix names."""
    names = []
    suffix_set = tuple(target_suffixes)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.endswith(suffix_set):
            names.append(name)
    return names


def convert_llama_linears_to_analog(
    model: torch.nn.Module,
    target_suffixes: Optional[Sequence[str]] = None,
    rpu_config=None,
) -> List[str]:
    """Replace selected LLaMA linear projections with AnalogLinear modules in place."""
    suffixes = tuple(target_suffixes or ("down_proj",))
    config = rpu_config or default_inference_rpu_config()
    converted = []

    for module_name in find_llama_linear_modules(model, suffixes):
        parent_name, leaf_name = _split_parent_name(module_name)
        parent = _get_submodule(model, parent_name)
        linear = getattr(parent, leaf_name)
        setattr(parent, leaf_name, _copy_linear_to_analog(linear, copy.deepcopy(config)))
        converted.append(module_name)

    return converted
