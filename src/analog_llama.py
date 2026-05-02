import copy
from typing import List, Optional, Sequence
import torch
from src.hardware_configs import build_rpu_config, requires_program_analog_weights
from src.rotation_precision import ROTATION_COMPUTE_DTYPE
from src.rotation_utils import hadamard_matrix, largest_power_of_two_divisor

def _require_aihwkit():
    """Import AIHWKit lazily so the float-only path works without analog deps."""
    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.configs import InferenceRPUConfig

    return AnalogLinear, InferenceRPUConfig


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


def apply_block_hadamard(x: torch.Tensor, hadamard_block: torch.Tensor) -> torch.Tensor:
    """Apply a block-diagonal Hadamard transform to the last dimension."""
    block_size = hadamard_block.shape[0]
    dim = x.shape[-1]
    if dim % block_size != 0:
        raise ValueError(f"Last dim {dim} not divisible by Hadamard block size {block_size}.")
    leading = x.shape[:-1]
    num_blocks = dim // block_size
    x_blocks = x.reshape(*leading, num_blocks, block_size)
    h = hadamard_block.to(device=x.device, dtype=x.dtype)
    return (x_blocks @ h).reshape(*leading, dim)


class AnalogOnlineHadamardLinear(torch.nn.Module):
    """Wrap an analog linear with the activation side of an online Hadamard."""

    def __init__(self, analog_linear: torch.nn.Module, hadamard_block: torch.Tensor) -> None:
        super().__init__()
        self.analog_linear = analog_linear
        self.register_buffer("hadamard_block", hadamard_block.detach().clone())
        self.in_features = analog_linear.in_features
        self.out_features = analog_linear.out_features

    @property
    def weight(self):
        return getattr(self.analog_linear, "weight", None)

    @property
    def bias(self):
        return getattr(self.analog_linear, "bias", None)

    def named_analog_layers(self, *args, **kwargs):
        return self.analog_linear.named_analog_layers(*args, **kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.analog_linear(apply_block_hadamard(inputs, self.hadamard_block))


def _rotate_linear_input_with_hadamard(
    linear: torch.nn.Linear,
    hadamard_block: torch.Tensor,
) -> None:
    """Bake the weight side of an online input Hadamard into a linear layer."""
    weight_dtype = linear.weight.dtype
    rotated = apply_block_hadamard(
        linear.weight.data.to(dtype=ROTATION_COMPUTE_DTYPE),
        hadamard_block.to(device=linear.weight.device, dtype=ROTATION_COMPUTE_DTYPE),
    )
    linear.weight.data = rotated.to(dtype=weight_dtype)


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
    if bias is None:
        analog.set_weights(weight)
    else:
        analog.set_weights(weight, bias)

    analog.to(device=linear.weight.device, dtype=linear.weight.dtype)
    analog.eval()
    return analog


def linear_to_analog(
    linear: torch.nn.Linear,
    *,
    hardware_preset: str = "ideal_analog",
    rpu_config=None,
) -> torch.nn.Module:
    """
    Convert one torch.nn.Linear into an AIHWKit AnalogLinear.

    Either pass an explicit rpu_config or choose one by hardware_preset.
    """
    config = rpu_config if rpu_config is not None else build_rpu_config(hardware_preset)
    analog = _copy_linear_to_analog(linear, copy.deepcopy(config))

    if requires_program_analog_weights(hardware_preset):
        analog.program_analog_weights()

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
    *,
    hardware_preset: str = "ideal_analog",
    rpu_config=None,
    online_hadamards: bool = False,
) -> List[str]:
    """
    Replace selected LLaMA linear projections with AnalogLinear modules in place.

    If rpu_config is provided, it is used as the base config.
    Otherwise, hardware_preset is resolved through src.hardware_configs.
    """
    suffixes = tuple(target_suffixes or ("down_proj",))
    base_config = rpu_config if rpu_config is not None else build_rpu_config(hardware_preset)
    converted = []
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    for module_name in find_llama_linear_modules(model, suffixes):
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
            if block_size < 2:
                raise ValueError(
                    f"{module_name} input dim {linear.in_features} has no power-of-two factor for R4."
                )
            online_hadamard = hadamard_matrix(
                block_size,
                device=linear.weight.device,
                dtype=ROTATION_COMPUTE_DTYPE,
            )

        if online_hadamard is not None:
            _rotate_linear_input_with_hadamard(linear, online_hadamard)

        analog = _copy_linear_to_analog(linear, copy.deepcopy(base_config))

        if requires_program_analog_weights(hardware_preset):
            analog.program_analog_weights()

        if online_hadamard is not None:
            online_hadamard = online_hadamard.to(
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
            analog = AnalogOnlineHadamardLinear(analog, online_hadamard)

        setattr(parent, leaf_name, analog)
        converted.append(module_name)

    return converted


def prepare_analog_model(
    model: torch.nn.Module,
    target_suffixes: Optional[Sequence[str]] = None,
    *,
    hardware_preset: str = "ideal_analog",
    rpu_config=None,
    online_hadamards: bool = False,
) -> List[str]:
    """
    Public model-level analog preparation API.

    This is the function full_model_pipeline.py should call.
    """
    return convert_llama_linears_to_analog(
        model,
        target_suffixes=target_suffixes,
        hardware_preset=hardware_preset,
        rpu_config=rpu_config,
        online_hadamards=online_hadamards,
    )