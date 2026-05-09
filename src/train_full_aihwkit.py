"""Train R1/R2 rotations against full-model LM loss using AIHWKIT AnalogLinear.

This mirrors src.train_full_analog, but the analog forward is provided by
AIHWKIT tiles instead of the hand-written differentiable simulator. Each wrapped
linear builds the current rotated weight inside forward(), injects it into the
AIHWKIT tile without detaching the autograd graph, and then runs the analog
layer as part of the normal LLaMA forward pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import wandb

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
from src.optimizer import SGDG
from src.rotation_precision import ROTATION_COMPUTE_DTYPE
from src.rotation_utils import hadamard_matrix, largest_power_of_two_divisor
from src.runtime_rotation import (
    RotationParameters,
    RuntimeRotatedEmbedding,
    build_runtime_linear_weight_and_bias,
)
from src.wandb_config import WANDB_ENTITY, WANDB_MODE, WANDB_PROJECT


def _apply_block_hadamard(x: torch.Tensor, hadamard_block: torch.Tensor) -> torch.Tensor:
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


def _load_weight_into_tile(analog_layer: nn.Module, weight: torch.Tensor) -> None:
    """Replace AIHWKIT tile weights while preserving the autograd chain to weight."""
    analog_module = analog_layer.analog_module

    if "shared_weights" in getattr(analog_module, "_parameters", {}):
        current = analog_module._parameters["shared_weights"]
        if current.shape == weight.shape:
            analog_module._parameters["shared_weights"] = weight.contiguous()
        elif current.shape == weight.T.shape:
            analog_module._parameters["shared_weights"] = weight.T.contiguous()
        else:
            raise ValueError(
                f"Cannot inject weight with shape {tuple(weight.shape)} into "
                f"AIHWKIT shared_weights shape {tuple(current.shape)}."
            )
        return

    tile_h = tile_w = None
    for param_name, param in list(analog_module.named_parameters(recurse=True)):
        if param_name.endswith("shared_weights"):
            parts = param_name.split(".")
        elif param_name.endswith("tile.weight"):
            parts = param_name.split(".")
        else:
            continue

        try:
            row_idx, col_idx = int(parts[1]), int(parts[2])
        except (IndexError, ValueError):
            continue

        if tile_h is None:
            tile_h, tile_w = param.shape

        row_start, col_start = row_idx * tile_h, col_idx * tile_w
        row_end = min(row_start + tile_h, weight.shape[0])
        col_end = min(col_start + tile_w, weight.shape[1])
        weight_slice = weight[row_start:row_end, col_start:col_end]

        if weight_slice.shape != (tile_h, tile_w):
            padded = weight.new_zeros(tile_h, tile_w)
            padded[: row_end - row_start, : col_end - col_start] = weight_slice
            weight_slice = padded

        parent = analog_module
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        parent._parameters[parts[-1]] = weight_slice


class AIHWKITRotatedLinear(nn.Module):
    """Runtime-rotated LLaMA linear backed by an AIHWKIT AnalogLinear tile."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rpu_config,
        get_r1: Callable[[], torch.Tensor],
        apply_r1: Optional[str] = None,
        get_r2: Optional[Callable[[], torch.Tensor]] = None,
        apply_r2: Optional[str] = None,
        head_dim: Optional[int] = None,
        online_hadamard: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        from aihwkit.nn import AnalogLinear

        self.base_linear = base_linear
        self.get_r1 = get_r1
        self.apply_r1 = apply_r1
        self.get_r2 = get_r2
        self.apply_r2 = apply_r2
        self.head_dim = head_dim
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.analog_linear = AnalogLinear(
            self.in_features,
            self.out_features,
            bias=False,
            rpu_config=copy.deepcopy(rpu_config),
        )
        self.analog_linear.set_weights(base_linear.weight.detach().float().cpu())
        self.analog_linear.to(device=base_linear.weight.device, dtype=torch.float32)
        self.analog_linear.eval()

        if online_hadamard is not None:
            self.register_buffer("online_hadamard", online_hadamard.detach().clone())
        else:
            self.online_hadamard = None

    @property
    def weight(self) -> torch.Tensor:
        return self.base_linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_linear.bias

    def named_analog_layers(self, *args, **kwargs):
        return self.analog_linear.named_analog_layers(*args, **kwargs)

    def _effective_weight_and_bias(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return build_runtime_linear_weight_and_bias(
            self.base_linear.weight,
            self.base_linear.bias,
            r1=self.get_r1(),
            apply_r1=self.apply_r1,
            r2=self.get_r2() if self.get_r2 is not None else None,
            apply_r2=self.apply_r2,
            head_dim=self.head_dim,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight, bias = self._effective_weight_and_bias()
        weight = weight.to(device=inputs.device, dtype=torch.float32)
        analog_inputs = inputs.to(dtype=torch.float32)

        if self.online_hadamard is not None:
            weight = _apply_block_hadamard(weight, self.online_hadamard)
            analog_inputs = _apply_block_hadamard(analog_inputs, self.online_hadamard)

        _load_weight_into_tile(self.analog_linear, weight)

        leading_shape = analog_inputs.shape[:-1]
        flat_inputs = analog_inputs.reshape(-1, analog_inputs.shape[-1])
        flat_outputs = self.analog_linear(flat_inputs)
        outputs = flat_outputs.reshape(*leading_shape, flat_outputs.shape[-1])

        if bias is not None:
            outputs = outputs + bias.to(device=outputs.device, dtype=outputs.dtype)
        return outputs.to(dtype=inputs.dtype)


@dataclass
class TrainFullAIHWKITConfig:
    model_name: str = DEFAULT_MODEL_NAME
    torch_dtype: Optional[torch.dtype] = torch.float32
    init_mode: str = "identity"
    seed: int = 0

    train_r2: bool = True
    r2_mode: Optional[str] = None
    r2_seed_offset: int = 1

    lr: float = 1.5
    momentum: float = 0.9
    num_steps: int = 200
    max_length: int = 128
    batch_size: int = 4
    dataset: str = "wikitext-2"

    hardware_preset: str = "advanced_ir_drop_8bit"
    rpu_overrides: Optional[dict] = None
    online_hadamards: bool = True

    log_every: int = 10
    use_wandb: bool = True
    checkpoint_path: Optional[str] = None

    profile: bool = False
    profile_dir: str = "profiles/train_full_aihwkit"
    profile_wait: int = 1
    profile_warmup: int = 1
    profile_active: int = 3
    profile_record_shapes: bool = True
    profile_with_stack: bool = False


def enable_aihwkit_analog_rotations(
    model: nn.Module,
    rotation_parameters: RotationParameters,
    rpu_config,
    *,
    online_hadamards: bool = True,
) -> None:
    """Replace LLaMA projections with AIHWKIT-backed runtime-rotated linears."""
    model.runtime_rotation_parameters = rotation_parameters
    model.model.embed_tokens = RuntimeRotatedEmbedding(
        model.model.embed_tokens,
        get_r1=lambda params=rotation_parameters: params.R1,
    )

    head_dim = rotation_parameters.metadata["head_dim"]
    intermediate_size = model.config.intermediate_size
    device = model.model.embed_tokens.weight.device

    r3_hadamard = None
    r4_hadamard = None
    if online_hadamards:
        r3_hadamard = hadamard_matrix(head_dim, device=device, dtype=ROTATION_COMPUTE_DTYPE)
        r4_block = largest_power_of_two_divisor(intermediate_size)
        if r4_block < 2:
            raise ValueError(
                f"intermediate_size {intermediate_size} has no power-of-two factor for R4."
            )
        r4_hadamard = hadamard_matrix(r4_block, device=device, dtype=ROTATION_COMPUTE_DTYPE)
        print(
            f"Online Hadamards enabled: R3 (head_dim={head_dim}), "
            f"R4 (block={r4_block}, intermediate_size={intermediate_size})"
        )

    def make(
        base: nn.Linear,
        apply_r1: str,
        apply_r2: Optional[str] = None,
        get_r2: Optional[Callable[[], torch.Tensor]] = None,
        online_hadamard: Optional[torch.Tensor] = None,
    ) -> AIHWKITRotatedLinear:
        return AIHWKITRotatedLinear(
            base,
            rpu_config,
            get_r1=lambda params=rotation_parameters: params.R1,
            apply_r1=apply_r1,
            get_r2=get_r2,
            apply_r2=apply_r2,
            head_dim=head_dim if apply_r2 else None,
            online_hadamard=online_hadamard,
        )

    for layer_idx, layer in enumerate(model.model.layers):
        get_r2 = lambda idx=layer_idx, params=rotation_parameters: params.get_layer_r2(idx)

        layer.self_attn.q_proj = make(layer.self_attn.q_proj, "input")
        layer.self_attn.k_proj = make(layer.self_attn.k_proj, "input")
        layer.self_attn.v_proj = make(layer.self_attn.v_proj, "input", "output", get_r2)
        layer.self_attn.o_proj = make(
            layer.self_attn.o_proj,
            "output",
            "input",
            get_r2,
            online_hadamard=r3_hadamard,
        )
        layer.mlp.up_proj = make(layer.mlp.up_proj, "input")
        layer.mlp.gate_proj = make(layer.mlp.gate_proj, "input")
        layer.mlp.down_proj = make(
            layer.mlp.down_proj,
            "output",
            online_hadamard=r4_hadamard,
        )

    model.lm_head = make(model.lm_head, "input")


def _freeze_non_rotation_params(
    model: nn.Module,
    runtime_params: RotationParameters,
    train_r2: bool,
) -> None:
    trainable_ids = {id(runtime_params.R1)}
    if train_r2:
        trainable_ids.update(id(param) for param in runtime_params.layer_R2.values())
    for param in model.parameters():
        param.requires_grad = id(param) in trainable_ids


def _print_trainable_rotation_parameter_counts(
    runtime_params: RotationParameters,
    train_r2: bool,
) -> None:
    total = runtime_params.R1.numel()
    print("Trainable rotation parameters:")
    print(f"  R1 {tuple(runtime_params.R1.shape)}: {runtime_params.R1.numel():,}")
    if train_r2:
        for name, param in runtime_params.layer_R2.items():
            total += param.numel()
            print(f"  {name} {tuple(param.shape)}: {param.numel():,}")
    else:
        print("  R2 matrices: not trainable")
    print(f"  total trainable rotation parameters: {total:,}")


def _load_dataset_text(dataset: str) -> str:
    from datasets import load_dataset

    if dataset == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif dataset == "wikitext-103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return "\n\n".join(text for text in (example["text"] for example in ds) if text.strip())


def _build_packed_batches(tokenizer, dataset, max_length, batch_size, device, seed):
    text = _load_dataset_text(dataset)
    saved_max = tokenizer.model_max_length
    tokenizer.model_max_length = 10**9
    try:
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    finally:
        tokenizer.model_max_length = saved_max
    tokens = torch.tensor(ids, dtype=torch.long)
    n_chunks = tokens.shape[0] // max_length
    if n_chunks < batch_size:
        raise ValueError(
            f"Dataset {dataset} has only {n_chunks} chunks of length {max_length}; "
            f"need >= batch_size={batch_size}."
        )
    chunks = tokens[: n_chunks * max_length].view(n_chunks, max_length)
    print(f"Loaded {dataset}: {tokens.shape[0]} tokens -> {n_chunks} chunks of length {max_length}")

    rng = random.Random(seed)
    while True:
        idx = [rng.randrange(n_chunks) for _ in range(batch_size)]
        batch = chunks[idx].to(device)
        labels = batch.clone()
        attention_mask = torch.ones_like(batch)
        yield batch, labels, attention_mask


def _build_profiler(config: TrainFullAIHWKITConfig):
    if not config.profile:
        return nullcontext(None)

    Path(config.profile_dir).mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=config.profile_wait,
            warmup=config.profile_warmup,
            active=config.profile_active,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(config.profile_dir),
        record_shapes=config.profile_record_shapes,
        profile_memory=True,
        with_stack=config.profile_with_stack,
    )


def _load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        raw_config = json.load(file)
    if not isinstance(raw_config, dict):
        raise TypeError(f"Config file {path} must contain a JSON object.")

    aliases = {"checkpoint": "checkpoint_path"}
    inverted_aliases = {
        "no_train_r2": "train_r2",
        "no_online_hadamards": "online_hadamards",
        "no_wandb": "use_wandb",
    }
    field_names = {field.name for field in fields(TrainFullAIHWKITConfig)}
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
    return config


def train_full_aihwkit(config: TrainFullAIHWKITConfig) -> dict:
    if config.use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            mode=WANDB_MODE,
            name=(
                f"aihwkit_full_{config.init_mode}"
                f"_hw={config.hardware_preset}"
                f"_bs={config.batch_size}"
                f"_had={int(config.online_hadamards)}"
            ),
            config={
                "model_name": config.model_name,
                "torch_dtype": str(config.torch_dtype),
                "init_mode": config.init_mode,
                "train_r2": config.train_r2,
                "r2_mode": config.r2_mode or config.init_mode,
                "r2_seed_offset": config.r2_seed_offset,
                "lr": config.lr,
                "momentum": config.momentum,
                "num_steps": config.num_steps,
                "batch_size": config.batch_size,
                "max_length": config.max_length,
                "dataset": config.dataset,
                "hardware_preset": config.hardware_preset,
                "rpu_overrides": config.rpu_overrides,
                "online_hadamards": config.online_hadamards,
                "checkpoint_path": config.checkpoint_path,
            },
        )

    device = get_default_device()
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model_name,
        device=device,
        torch_dtype=config.torch_dtype,
    )
    if not is_llama_like_model(model):
        raise ValueError(f"{config.model_name} does not expose a LLaMA-style layout.")

    prepare_model_for_rotation(model)
    rotation_params = RotationParameters.for_model(
        model,
        rotate_mode=config.init_mode,
        r2_mode=config.r2_mode if config.train_r2 else "identity",
        seed=config.seed,
        r2_seed_offset=config.r2_seed_offset,
    )
    rotation_params.to(device=device, dtype=torch.float32)

    rpu_config = build_rpu_config(config.hardware_preset, overrides=config.rpu_overrides)
    enable_aihwkit_analog_rotations(
        model,
        rotation_parameters=rotation_params,
        rpu_config=rpu_config,
        online_hadamards=config.online_hadamards,
    )

    _freeze_non_rotation_params(model, rotation_params, config.train_r2)
    model.train()

    params_to_train = [rotation_params.R1]
    if config.train_r2:
        params_to_train += list(rotation_params.layer_R2.values())
    _print_trainable_rotation_parameter_counts(rotation_params, config.train_r2)
    optimizer = SGDG(params_to_train, lr=config.lr, momentum=config.momentum, stiefel=True)

    batches = _build_packed_batches(
        tokenizer,
        dataset=config.dataset,
        max_length=config.max_length,
        batch_size=config.batch_size,
        device=device,
        seed=config.seed,
    )

    history = []
    best_loss = float("inf")
    best_step = None
    best_state = None
    profiler_obj = None
    with _build_profiler(config) as profiler:
        profiler_obj = profiler
        for step in range(config.num_steps):
            input_ids, labels, attention_mask = next(batches)

            with torch.profiler.record_function("forward"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                analog_loss = outputs.loss

            with torch.profiler.record_function("backward"):
                optimizer.zero_grad(set_to_none=True)
                analog_loss.backward()

            loss_value = float(analog_loss.detach())
            if loss_value < best_loss:
                best_loss = loss_value
                best_step = step
                best_state = {
                    "R1": rotation_params.R1.detach().cpu().clone(),
                    "R2": {
                        key: value.detach().cpu().clone()
                        for key, value in rotation_params.layer_R2.items()
                    }
                    if config.train_r2
                    else {},
                }

            with torch.profiler.record_function("optimizer_step"):
                optimizer.step()

            r1_grad_norm = (
                float(rotation_params.R1.grad.norm())
                if rotation_params.R1.grad is not None
                else float("nan")
            )
            eye = torch.eye(
                rotation_params.R1.shape[0],
                device=rotation_params.R1.device,
                dtype=rotation_params.R1.dtype,
            )
            r1_dist = float((rotation_params.R1.detach() - eye).norm())

            record = {
                "step": step,
                "analog_lm_loss": loss_value,
                "best_analog_lm_loss": best_loss,
                "best_step": best_step,
                "r1_grad_norm": r1_grad_norm,
                "r1_dist": r1_dist,
            }
            if config.train_r2:
                r2_params = list(rotation_params.layer_R2.values())
                r2_grads = [float(param.grad.norm()) for param in r2_params if param.grad is not None]
                eye_r2 = torch.eye(
                    r2_params[0].shape[0],
                    device=r2_params[0].device,
                    dtype=r2_params[0].dtype,
                )
                r2_dists = [float((param.detach() - eye_r2).norm()) for param in r2_params]
                record["r2_grad_norm_mean"] = (
                    float(torch.tensor(r2_grads).mean()) if r2_grads else float("nan")
                )
                record["r2_dist_mean"] = float(torch.tensor(r2_dists).mean())

            history.append(record)
            if config.use_wandb:
                wandb.log(record)

            if step % config.log_every == 0:
                r2_info = (
                    f"  r2_grad={record['r2_grad_norm_mean']:.3e}  "
                    f"|R2-I|={record['r2_dist_mean']:.3e}"
                    if config.train_r2
                    else ""
                )
                print(
                    f"step {step:4d}  analog_lm={record['analog_lm_loss']:.4f}  "
                    f"r1_grad={r1_grad_norm:.3e}  |R1-I|={r1_dist:.3e}{r2_info}"
                )

            if profiler is not None:
                profiler.step()

    if config.profile and profiler_obj is not None:
        print(
            profiler_obj.key_averages().table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                row_limit=30,
            )
        )

    if best_state is None:
        best_state = {
            "R1": rotation_params.R1.detach().cpu().clone(),
            "R2": {
                key: value.detach().cpu().clone()
                for key, value in rotation_params.layer_R2.items()
            }
            if config.train_r2
            else {},
        }
    result = {
        "R1": best_state["R1"],
        "R2": best_state["R2"],
        "history": history,
        "final_analog_lm_loss": history[-1]["analog_lm_loss"] if history else None,
        "best_analog_lm_loss": best_loss if history else None,
        "best_step": best_step,
    }

    if config.checkpoint_path:
        os.makedirs(os.path.dirname(os.path.abspath(config.checkpoint_path)), exist_ok=True)
        torch.save({"R1": result["R1"], "R2": result["R2"]}, config.checkpoint_path)
        print(
            f"Saved best checkpoint from step {result['best_step']} "
            f"(analog_lm={result['best_analog_lm_loss']:.4f}) to {config.checkpoint_path}"
        )
        if config.use_wandb:
            wandb.save(config.checkpoint_path)

    if config.use_wandb:
        wandb.finish()
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train rotations against full-model LM loss using AIHWKIT AnalogLinear."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON config file whose keys match TrainFullAIHWKITConfig fields.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["auto", *TORCH_DTYPE_CHOICES.keys()],
    )
    parser.add_argument(
        "--init-mode",
        default="identity",
        choices=["identity", "sign_flip", "random", "hadamard", "block_hadamard", "hadamard_D"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-train-r2", action="store_true")
    parser.add_argument(
        "--r2-mode",
        default=None,
        choices=["identity", "sign_flip", "random", "hadamard", "block_hadamard", "hadamard_D"],
    )
    parser.add_argument("--r2-seed-offset", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    parser.add_argument(
        "--hardware-preset",
        default="advanced_ir_drop_8bit",
        choices=supported_hardware_presets(),
    )
    parser.add_argument(
        "--rpu-override",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override an RPU config field, e.g. --rpu-override forward.ir_drop=0.5.",
    )
    parser.add_argument(
        "--no-online-hadamards",
        action="store_true",
        help="Disable fixed R3/R4 Hadamards on o_proj/down_proj inputs.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", default="profiles/train_full_aihwkit")
    parser.add_argument("--profile-wait", type=int, default=1)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-active", type=int, default=3)
    parser.add_argument("--profile-no-record-shapes", action="store_true")
    parser.add_argument("--profile-with-stack", action="store_true")
    return parser


def _parse_override_value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_rpu_overrides(raw_overrides: list[str]) -> Optional[dict]:
    if not raw_overrides:
        return None
    overrides = {}
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError(f"RPU override must be PATH=VALUE, got {raw!r}.")
        key, value = raw.split("=", 1)
        overrides[key] = _parse_override_value(value)
    return overrides


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.config:
        config = TrainFullAIHWKITConfig(**_load_json_config(args.config))
    else:
        config = TrainFullAIHWKITConfig(
            model_name=args.model_name,
            torch_dtype=resolve_torch_dtype(args.torch_dtype),
            init_mode=args.init_mode,
            seed=args.seed,
            train_r2=not args.no_train_r2,
            r2_mode=args.r2_mode,
            r2_seed_offset=args.r2_seed_offset,
            lr=args.lr,
            momentum=args.momentum,
            num_steps=args.num_steps,
            max_length=args.max_length,
            batch_size=args.batch_size,
            dataset=args.dataset,
            hardware_preset=args.hardware_preset,
            rpu_overrides=_parse_rpu_overrides(args.rpu_override),
            online_hadamards=not args.no_online_hadamards,
            log_every=args.log_every,
            use_wandb=not args.no_wandb,
            checkpoint_path=args.checkpoint,
            profile=args.profile,
            profile_dir=args.profile_dir,
            profile_wait=args.profile_wait,
            profile_warmup=args.profile_warmup,
            profile_active=args.profile_active,
            profile_record_shapes=not args.profile_no_record_shapes,
            profile_with_stack=args.profile_with_stack,
        )
    result = train_full_aihwkit(config)
    print(f"final analog_lm_loss={result['final_analog_lm_loss']:.4f}")


if __name__ == "__main__":
    main()
