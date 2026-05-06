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


def _is_cuda_device(device: torch.device) -> bool:
    return device.type == "cuda"


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


class AnalogLinearSequenceWrapper(torch.nn.Module):
    """Flatten sequence/batch dimensions before AIHWKit and restore them after.

    AIHWKit's torch IR-drop tile expects a 2D activation matrix. Hugging Face
    transformer blocks call Linear modules with [batch, seq, hidden], so this
    wrapper keeps the module shape-compatible with standard LLaMA forwards.
    """

    def __init__(self, analog_linear: torch.nn.Module) -> None:
        super().__init__()
        self.analog_linear = analog_linear
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
        if inputs.ndim <= 2:
            return self.analog_linear(inputs)
        leading_shape = inputs.shape[:-1]
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_outputs = self.analog_linear(flat_inputs)
        return flat_outputs.reshape(*leading_shape, flat_outputs.shape[-1])


class PagedAnalogModuleWrapper(torch.nn.Module):
    """Keep an analog module on CPU and move it to the input device for forward."""

    def __init__(
        self,
        analog_module: torch.nn.Module,
        *,
        module_name: str = "analog_module",
        storage_device: str = "cpu",
        execution_device: Optional[str] = None,
        compute_dtype: torch.dtype = torch.float32,
        clear_cuda_cache: bool = False,
        log_cuda_memory: bool = False,
    ) -> None:
        super().__init__()
        self.analog_module = analog_module
        self.module_name = module_name
        self.storage_device = torch.device(storage_device)
        self.execution_device = torch.device(execution_device) if execution_device else None
        self.compute_dtype = compute_dtype
        self.clear_cuda_cache = clear_cuda_cache
        self.log_cuda_memory = log_cuda_memory
        self.in_features = analog_module.in_features
        self.out_features = analog_module.out_features
        self.analog_module.to(self.storage_device)

    @property
    def weight(self):
        return getattr(self.analog_module, "weight", None)

    @property
    def bias(self):
        return getattr(self.analog_module, "bias", None)

    def named_analog_layers(self, *args, **kwargs):
        return self.analog_module.named_analog_layers(*args, **kwargs)

    def _log_cuda_memory(self, label: str, device: torch.device) -> None:
        if not self.log_cuda_memory or device.type != "cuda":
            return
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        free, total = torch.cuda.mem_get_info(device)
        print(
            f"[cuda_mem:{self.module_name}:{label}] "
            f"allocated={allocated:.1f}MiB reserved={reserved:.1f}MiB "
            f"max_allocated={max_allocated:.1f}MiB "
            f"driver_free={free / 1024**2:.1f}MiB driver_total={total / 1024**2:.1f}MiB",
            flush=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_device = inputs.device
        execution_device = self.execution_device or inputs.device
        self._log_cuda_memory("before_to_execution", execution_device)
        with torch.profiler.record_function(f"analog_page:{self.module_name}:to_execution"):
            self.analog_module.to(device=execution_device, dtype=self.compute_dtype)
        self._log_cuda_memory("after_to_execution", execution_device)
        try:
            with torch.profiler.record_function(f"analog_page:{self.module_name}:forward"):
                outputs = self.analog_module(
                    inputs.to(device=execution_device, dtype=self.compute_dtype)
                )
            if _is_cuda_device(execution_device):
                torch.cuda.synchronize(execution_device)
        finally:
            self._log_cuda_memory("before_to_storage", execution_device)
            with torch.profiler.record_function(f"analog_page:{self.module_name}:to_storage"):
                self.analog_module.to(self.storage_device)
            if _is_cuda_device(execution_device) and self.clear_cuda_cache:
                torch.cuda.empty_cache()
            self._log_cuda_memory("after_to_storage", execution_device)
        return outputs.to(device=original_device, dtype=inputs.dtype)


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
    *,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
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

    target_device = device if device is not None else linear.weight.device
    target_dtype = dtype if dtype is not None else linear.weight.dtype
    analog.to(device=target_device, dtype=target_dtype)
    analog.eval()
    return analog


def linear_to_analog(
    linear: torch.nn.Linear,
    *,
    hardware_preset: str = "ideal_analog",
    rpu_config=None,
    analog_device: Optional[str] = None,
) -> torch.nn.Module:
    """
    Convert one torch.nn.Linear into an AIHWKit AnalogLinear.

    Either pass an explicit rpu_config or choose one by hardware_preset.
    """
    config = rpu_config if rpu_config is not None else build_rpu_config(hardware_preset)
    analog = _copy_linear_to_analog(linear, copy.deepcopy(config), device=analog_device)

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
    page_analog_tiles: bool = False,
    analog_storage_device: str = "cpu",
    analog_execution_device: Optional[str] = None,
    cpu_paged_analog_targets: Sequence[str] = (),
    clear_paged_cuda_cache: bool = False,
    log_paged_cuda_memory: bool = False,
) -> List[str]:
    """
    Replace selected LLaMA linear projections with AnalogLinear modules in place.

    If rpu_config is provided, it is used as the base config.
    Otherwise, hardware_preset is resolved through src.hardware_configs.
    """
    suffixes = tuple(target_suffixes or ("down_proj",))
    cpu_paged_suffixes = tuple(cpu_paged_analog_targets)
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

        analog_device = analog_storage_device if page_analog_tiles else None
        analog = _copy_linear_to_analog(
            linear,
            copy.deepcopy(base_config),
            device=analog_device,
            dtype=torch.float32 if page_analog_tiles else None,
        )

        if requires_program_analog_weights(hardware_preset):
            analog.program_analog_weights()

        if online_hadamard is not None:
            online_hadamard = online_hadamard.to(
                device=analog_storage_device if page_analog_tiles else linear.weight.device,
                dtype=linear.weight.dtype,
            )
            analog = AnalogOnlineHadamardLinear(analog, online_hadamard)

        if page_analog_tiles:
            paged_execution_device = (
                "cpu" if leaf_name in cpu_paged_suffixes else analog_execution_device
            )
            analog = PagedAnalogModuleWrapper(
                analog,
                module_name=module_name,
                storage_device=analog_storage_device,
                execution_device=paged_execution_device,
                compute_dtype=torch.float32,
                clear_cuda_cache=clear_paged_cuda_cache,
                log_cuda_memory=log_paged_cuda_memory,
            )

        analog = AnalogLinearSequenceWrapper(analog)

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
    page_analog_tiles: bool = False,
    analog_storage_device: str = "cpu",
    analog_execution_device: Optional[str] = None,
    cpu_paged_analog_targets: Sequence[str] = (),
    clear_paged_cuda_cache: bool = False,
    log_paged_cuda_memory: bool = False,
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
        page_analog_tiles=page_analog_tiles,
        analog_storage_device=analog_storage_device,
        analog_execution_device=analog_execution_device,
        cpu_paged_analog_targets=cpu_paged_analog_targets,
        clear_paged_cuda_cache=clear_paged_cuda_cache,
        log_paged_cuda_memory=log_paged_cuda_memory,
    )