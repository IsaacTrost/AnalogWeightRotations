"""Train rotation matrices (R1, R2) against the actual analog LM loss.

Unlike train_r1.py, the training signal here is the real cross-entropy loss
of the model running through a differentiable analog forward pass:
  - 8-bit symmetric per-tensor weight quantization via STE
  - 8-bit symmetric per-token input (DAC) quantization via STE
  - Simplified IR drop: column-position attenuation V_eff(j) = V * (1 - alpha * j/n_cols)
  - Optional fixed online Hadamards R3 (per-head head_dim, before o_proj) and
    R4 (block-Hadamard intermediate, before down_proj). Both are SpinQuant's
    "online" rotations — fixed, untrained, but applied to both x and W so that
    activation outliers are flattened before quantization while keeping the
    float matmul invariant.

The float LM loss is invariant to rotation, but the analog LM loss is NOT —
quantization rounding and IR drop both depend on the actual values in W_eff,
which change with R. This gives a real nonzero gradient through R.
"""
import argparse
import json
import os
import random
from dataclasses import dataclass, fields
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import wandb

from pathlib import Path
from contextlib import nullcontext

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
from src.rotation_utils import (
    hadamard_matrix,
    largest_power_of_two_divisor,
)
from src.runtime_rotation import (
    RotationParameters,
    build_runtime_linear_weight_and_bias,
)
from src.wandb_config import WANDB_ENTITY, WANDB_PROJECT, WANDB_MODE


def _apply_block_hadamard(x: torch.Tensor, hadamard_block: torch.Tensor) -> torch.Tensor:
    """Apply a block-diagonal Hadamard transform to the last dim of x via reshape.

    x: tensor whose last dim is `num_blocks * block_size`.
    hadamard_block: orthonormal Hadamard of shape [block_size, block_size].

    Math invariance: applying the same H to both an activation and the input side
    of a weight preserves the matmul, because H @ H.T = I.
    """
    block_size = hadamard_block.shape[0]
    dim = x.shape[-1]
    if dim % block_size != 0:
        raise ValueError(
            f"Last dim {dim} not divisible by Hadamard block size {block_size}."
        )
    leading = x.shape[:-1]
    num_blocks = dim // block_size
    x_blocks = x.reshape(*leading, num_blocks, block_size)
    h = hadamard_block.to(x.dtype)
    rotated = x_blocks @ h
    return rotated.reshape(*leading, dim)


@dataclass
class TrainFullAnalogConfig:
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

    # Hardware effects
    num_bits: int = 8
    ir_drop_coeff: float = 0.1
    online_hadamards: bool = True

    log_every: int = 10
    use_wandb: bool = True
    checkpoint_path: Optional[str] = None

    # Profiling
    profile: bool = False
    profile_dir: str = "profiles/train_full_analog"
    profile_wait: int = 1
    profile_warmup: int = 1
    profile_active: int = 3
    profile_record_shapes: bool = True
    profile_with_stack: bool = False

def _load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    if not isinstance(raw_config, dict):
        raise TypeError(f"Config file {path} must contain a JSON object.")

    aliases = {
        "checkpoint": "checkpoint_path",
    }
    inverted_aliases = {
        "no_train_r2": "train_r2",
        "no_online_hadamards": "online_hadamards",
        "no_wandb": "use_wandb",
    }
    field_names = {field.name for field in fields(TrainFullAnalogConfig)}
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


class AnalogRotatedLinear(torch.nn.Module):
    """Rotation wrapper that applies differentiable 8-bit quantization + IR drop in forward.

    Quantization: symmetric per-tensor STE (straight-through estimator).
      - Forward: round to nearest 8-bit level.
      - Backward: gradient passes through as if no quantization.

    IR drop: simplified column-position attenuation.
      - V_eff(column j) = V_input * (1 - ir_drop_coeff * j / n_cols)
      - Implemented by scaling weight columns, which is equivalent for a linear layer.
      - Fully differentiable w.r.t. W_eff and therefore w.r.t. R.

    Why not use AIHWKit AnalogLinear directly: AIHWKit tiles take weights at
    construction time. Re-programming every step to track changing R would break
    the autograd graph. Our in-graph simulation keeps gradient flow intact.
    """

    def __init__(
        self,
        base_linear: torch.nn.Linear,
        get_r1: Callable[[], torch.Tensor],
        apply_r1: Optional[str] = None,
        get_r2: Optional[Callable[[], torch.Tensor]] = None,
        apply_r2: Optional[str] = None,
        head_dim: Optional[int] = None,
        num_bits: int = 8,
        ir_drop_coeff: float = 0.1,
        online_hadamard: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.get_r1 = get_r1
        self.apply_r1 = apply_r1
        self.get_r2 = get_r2
        self.apply_r2 = apply_r2
        self.head_dim = head_dim
        self.num_bits = num_bits
        self.ir_drop_coeff = ir_drop_coeff
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        if online_hadamard is not None:
            self.register_buffer("online_hadamard", online_hadamard.detach())
        else:
            self.online_hadamard = None

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base_linear.weight

    @property
    def bias(self) -> Optional[torch.nn.Parameter]:
        return self.base_linear.bias

    def _effective_weight_and_bias(self):
        return build_runtime_linear_weight_and_bias(
            self.base_linear.weight,
            self.base_linear.bias,
            r1=self.get_r1(),
            apply_r1=self.apply_r1,
            r2=self.get_r2() if self.get_r2 is not None else None,
            apply_r2=self.apply_r2,
            head_dim=self.head_dim,
        )

    def _quantize_ste(self, w: torch.Tensor) -> torch.Tensor:
        """Symmetric per-tensor 8-bit quantization with straight-through gradient."""
        n_levels = 2 ** (self.num_bits - 1) - 1
        scale = w.abs().max() / n_levels
        if scale == 0:
            return w
        w_q = (w / scale).clamp(-n_levels, n_levels).round() * scale
        return w + (w_q - w).detach()

    def _apply_ir_drop(self, w: torch.Tensor) -> torch.Tensor:
        """Scale each column by 1 - ir_drop_coeff * j/n_cols.

        Models the voltage attenuation seen by devices further from the driver.
        Fully differentiable — gradient flows through the column scaling to W and R.
        """
        if self.ir_drop_coeff == 0.0:
            return w
        n_cols = w.shape[1]
        j = torch.arange(n_cols, device=w.device, dtype=w.dtype)
        attenuation = 1.0 - self.ir_drop_coeff * j / (n_cols - 1)
        return w * attenuation.unsqueeze(0)

    def _quantize_input_ste(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric per-token DAC quantization with straight-through gradient.

        Models the input DAC with per-vector bound management: each token's max
        sets the DAC range for that token. Rotations flatten per-token outliers,
        which directly improves the SNR seen by this quantizer.
        """
        n_levels = 2 ** (self.num_bits - 1) - 1
        scale = x.abs().amax(dim=-1, keepdim=True) / n_levels
        scale = scale.clamp(min=1e-8)
        x_q = (x / scale).clamp(-n_levels, n_levels).round() * scale
        return x + (x_q - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, bias = self._effective_weight_and_bias()
        if self.online_hadamard is not None:
            w = _apply_block_hadamard(w, self.online_hadamard)
            x = _apply_block_hadamard(x, self.online_hadamard)
        w = self._quantize_ste(w)
        w = self._apply_ir_drop(w)
        x = self._quantize_input_ste(x)
        return F.linear(x, w, bias)


def enable_analog_rotations(
    model: torch.nn.Module,
    rotation_parameters: RotationParameters,
    num_bits: int = 8,
    ir_drop_coeff: float = 0.1,
    online_hadamards: bool = True,
) -> None:
    """Replace all attention/MLP linear layers with AnalogRotatedLinear modules.

    When `online_hadamards` is True, attach SpinQuant-style fixed Hadamards:
      - R3: per-head head_dim Hadamard, applied online to o_proj input.
      - R4: block-Hadamard with block_size = largest_pow2_divisor(intermediate_size),
        applied online to down_proj input.
    These flatten activation outliers at the two points R1/R2 don't reach.
    """
    from src.runtime_rotation import RuntimeRotatedEmbedding

    model.runtime_rotation_parameters = rotation_parameters
    model.model.embed_tokens = RuntimeRotatedEmbedding(
        model.model.embed_tokens,
        get_r1=lambda p=rotation_parameters: p.R1,
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

    def make(base, apply_r1, apply_r2=None, get_r2=None, online_hadamard=None):
        return AnalogRotatedLinear(
            base,
            get_r1=lambda p=rotation_parameters: p.R1,
            apply_r1=apply_r1,
            get_r2=get_r2,
            apply_r2=apply_r2,
            head_dim=head_dim if apply_r2 else None,
            num_bits=num_bits,
            ir_drop_coeff=ir_drop_coeff,
            online_hadamard=online_hadamard,
        )

    for layer_idx, layer in enumerate(model.model.layers):
        get_r2 = lambda idx=layer_idx, p=rotation_parameters: p.get_layer_r2(idx)

        layer.self_attn.q_proj = make(layer.self_attn.q_proj, "input")
        layer.self_attn.k_proj = make(layer.self_attn.k_proj, "input")
        layer.self_attn.v_proj = make(layer.self_attn.v_proj, "input", "output", get_r2)
        layer.self_attn.o_proj = make(
            layer.self_attn.o_proj, "output", "input", get_r2,
            online_hadamard=r3_hadamard,
        )
        layer.mlp.up_proj   = make(layer.mlp.up_proj,   "input")
        layer.mlp.gate_proj = make(layer.mlp.gate_proj, "input")
        layer.mlp.down_proj = make(
            layer.mlp.down_proj, "output",
            online_hadamard=r4_hadamard,
        )

    model.lm_head = make(model.lm_head, "input")


def _freeze_non_rotation_params(
    model: torch.nn.Module,
    runtime_params: RotationParameters,
    train_r2: bool,
) -> None:
    trainable_ids = {id(runtime_params.R1)}
    if train_r2:
        trainable_ids.update(id(p) for p in runtime_params.layer_R2.values())
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
    return "\n\n".join(t for t in (ex["text"] for ex in ds) if t.strip())


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
        raise ValueError(f"Dataset {dataset} has only {n_chunks} chunks of length {max_length}; need >= batch_size={batch_size}.")
    chunks = tokens[: n_chunks * max_length].view(n_chunks, max_length)
    print(f"Loaded {dataset}: {tokens.shape[0]} tokens -> {n_chunks} chunks of length {max_length}")

    rng = random.Random(seed)
    while True:
        idx = [rng.randrange(n_chunks) for _ in range(batch_size)]
        batch = chunks[idx].to(device)
        labels = batch.clone()
        attn_mask = torch.ones_like(batch)
        yield batch, labels, attn_mask


def _build_profiler(config: TrainFullAnalogConfig):
    """Create a PyTorch profiler for the train loop, or a no-op context."""
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


def train_full_analog(config: TrainFullAnalogConfig) -> dict:
    if config.use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            mode=WANDB_MODE,
            name=f"analog_{config.init_mode}_bits={config.num_bits}_ir={config.ir_drop_coeff}_bs={config.batch_size}_had={int(config.online_hadamards)}",
            config={
                "model_name": config.model_name,
                "init_mode": config.init_mode,
                "train_r2": config.train_r2,
                "lr": config.lr,
                "momentum": config.momentum,
                "num_steps": config.num_steps,
                "num_bits": config.num_bits,
                "ir_drop_coeff": config.ir_drop_coeff,
                "seed": config.seed,
                "batch_size": config.batch_size,
                "max_length": config.max_length,
                "dataset": config.dataset,
                "online_hadamards": config.online_hadamards,
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

    from src.rotation_utils import get_rotation_matrix
    from src.rotation_precision import ROTATION_COMPUTE_DTYPE
    rotation_params = RotationParameters.for_model(
        model,
        rotate_mode=config.init_mode,
        r2_mode=config.r2_mode,
        seed=config.seed,
        r2_seed_offset=config.r2_seed_offset,
    )

    enable_analog_rotations(
        model,
        rotation_parameters=rotation_params,
        num_bits=config.num_bits,
        ir_drop_coeff=config.ir_drop_coeff,
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
    prof_obj = None
    with _build_profiler(config) as prof:
        prof_obj = prof
        for step in range(config.num_steps):
            input_ids, labels, attn_mask = next(batches)

            with torch.profiler.record_function("forward"):
                outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
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

            r1_grad_norm = float(rotation_params.R1.grad.norm()) if rotation_params.R1.grad is not None else float("nan")
            eye = torch.eye(rotation_params.R1.shape[0], device=rotation_params.R1.device, dtype=rotation_params.R1.dtype)
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
                r2_grads = [float(p.grad.norm()) for p in r2_params if p.grad is not None]
                eye_r2 = torch.eye(r2_params[0].shape[0], device=r2_params[0].device, dtype=r2_params[0].dtype)
                r2_dists = [float((p.detach() - eye_r2).norm()) for p in r2_params]
                record["r2_grad_norm_mean"] = float(torch.tensor(r2_grads).mean()) if r2_grads else float("nan")
                record["r2_dist_mean"] = float(torch.tensor(r2_dists).mean())

            history.append(record)
            if config.use_wandb:
                wandb.log(record)

            if step % config.log_every == 0:
                r2_info = (
                    f"  r2_grad={record['r2_grad_norm_mean']:.3e}  |R2-I|={record['r2_dist_mean']:.3e}"
                    if config.train_r2 else ""
                )
                print(
                    f"step {step:4d}  analog_lm={record['analog_lm_loss']:.4f}  "
                    f"r1_grad={r1_grad_norm:.3e}  |R1-I|={r1_dist:.3e}{r2_info}"
                )

            if prof is not None:
                prof.step()

    if config.profile and prof_obj is not None:
        print(
            prof_obj.key_averages().table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                row_limit=30,
            )
        )

    if config.use_wandb:
        wandb.finish()

    if best_state is None:
        best_state = {
            "R1": rotation_params.R1.detach().cpu().clone(),
            "R2": {
                k: v.detach().cpu().clone()
                for k, v in rotation_params.layer_R2.items()
            }
            if config.train_r2
            else {},
        }
    result = {
        "R1": best_state["R1"],
        "R2": best_state["R2"],
        "history": history,
        "final_analog_lm_loss": history[-1]["analog_lm_loss"] if history else None,
        "final_analog_degradation": history[-1]["analog_lm_loss"] if history else None,
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

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train rotations against the analog LM loss.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON config file whose keys match TrainFullAnalogConfig fields.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", *TORCH_DTYPE_CHOICES.keys()])
    parser.add_argument("--init-mode", default="identity",
                        choices=["identity", "sign_flip", "random", "hadamard", "block_hadamard", "hadamard_D"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-train-r2", action="store_true")
    parser.add_argument("--r2-mode", default=None)
    parser.add_argument("--r2-seed-offset", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.5,
                        help="SGDG step-size cap; the actual step is min(lr, 1/||grad||_1) so "
                             "1.5 hands control to the adaptive cap. SpinQuant default.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    parser.add_argument("--num-bits", type=int, default=8)
    parser.add_argument("--ir-drop-coeff", type=float, default=0.1)
    parser.add_argument("--no-online-hadamards", action="store_true",
                        help="Disable the fixed R3/R4 Hadamards on o_proj/down_proj inputs.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", default="profiles/train_full_analog")
    parser.add_argument("--profile-wait", type=int, default=1)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-active", type=int, default=3)
    parser.add_argument("--profile-no-record-shapes", action="store_true")
    parser.add_argument("--profile-with-stack", action="store_true")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.config:
        config = TrainFullAnalogConfig(**_load_json_config(args.config))
    else:
        config = TrainFullAnalogConfig(
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
            num_bits=args.num_bits,
            ir_drop_coeff=args.ir_drop_coeff,
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
    result = train_full_analog(config)
    print(f"final analog_lm_loss={result['final_analog_lm_loss']:.4f}")


if __name__ == "__main__":
    main()

"""Train rotation matrices (R1, R2) against the actual analog LM loss.

Unlike train_r1.py, the training signal here is the real cross-entropy loss
of the model running through a differentiable analog forward pass:
  - 8-bit symmetric per-tensor weight quantization via STE
  - 8-bit symmetric per-token input (DAC) quantization via STE
  - Simplified IR drop: column-position attenuation V_eff(j) = V * (1 - alpha * j/n_cols)
  - Optional fixed online Hadamards R3 (per-head head_dim, before o_proj) and
    R4 (block-Hadamard intermediate, before down_proj). Both are SpinQuant's
    "online" rotations — fixed, untrained, but applied to both x and W so that
    activation outliers are flattened before quantization while keeping the
    float matmul invariant.

The float LM loss is invariant to rotation, but the analog LM loss is NOT —
quantization rounding and IR drop both depend on the actual values in W_eff,
which change with R. This gives a real nonzero gradient through R.
"""
import argparse
import json
import os
import random
from dataclasses import dataclass, fields
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import wandb

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
from src.rotation_utils import (
    hadamard_matrix,
    largest_power_of_two_divisor,
)
from src.runtime_rotation import (
    RotationParameters,
    build_runtime_linear_weight_and_bias,
)
from src.wandb_config import WANDB_ENTITY, WANDB_PROJECT, WANDB_MODE


def _apply_block_hadamard(x: torch.Tensor, hadamard_block: torch.Tensor) -> torch.Tensor:
    """Apply a block-diagonal Hadamard transform to the last dim of x via reshape.

    x: tensor whose last dim is `num_blocks * block_size`.
    hadamard_block: orthonormal Hadamard of shape [block_size, block_size].

    Math invariance: applying the same H to both an activation and the input side
    of a weight preserves the matmul, because H @ H.T = I.
    """
    block_size = hadamard_block.shape[0]
    dim = x.shape[-1]
    if dim % block_size != 0:
        raise ValueError(
            f"Last dim {dim} not divisible by Hadamard block size {block_size}."
        )
    leading = x.shape[:-1]
    num_blocks = dim // block_size
    x_blocks = x.reshape(*leading, num_blocks, block_size)
    h = hadamard_block.to(x.dtype)
    rotated = x_blocks @ h
    return rotated.reshape(*leading, dim)


@dataclass
class TrainFullAnalogConfig:
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

    # Hardware effects
    num_bits: int = 8
    ir_drop_coeff: float = 0.1
    online_hadamards: bool = True

    log_every: int = 10
    use_wandb: bool = True
    checkpoint_path: Optional[str] = None


def _load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    if not isinstance(raw_config, dict):
        raise TypeError(f"Config file {path} must contain a JSON object.")

    aliases = {
        "checkpoint": "checkpoint_path",
    }
    inverted_aliases = {
        "no_train_r2": "train_r2",
        "no_online_hadamards": "online_hadamards",
        "no_wandb": "use_wandb",
    }
    field_names = {field.name for field in fields(TrainFullAnalogConfig)}
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


class AnalogRotatedLinear(torch.nn.Module):
    """Rotation wrapper that applies differentiable 8-bit quantization + IR drop in forward.

    Quantization: symmetric per-tensor STE (straight-through estimator).
      - Forward: round to nearest 8-bit level.
      - Backward: gradient passes through as if no quantization.

    IR drop: simplified column-position attenuation.
      - V_eff(column j) = V_input * (1 - ir_drop_coeff * j / n_cols)
      - Implemented by scaling weight columns, which is equivalent for a linear layer.
      - Fully differentiable w.r.t. W_eff and therefore w.r.t. R.

    Why not use AIHWKit AnalogLinear directly: AIHWKit tiles take weights at
    construction time. Re-programming every step to track changing R would break
    the autograd graph. Our in-graph simulation keeps gradient flow intact.
    """

    def __init__(
        self,
        base_linear: torch.nn.Linear,
        get_r1: Callable[[], torch.Tensor],
        apply_r1: Optional[str] = None,
        get_r2: Optional[Callable[[], torch.Tensor]] = None,
        apply_r2: Optional[str] = None,
        head_dim: Optional[int] = None,
        num_bits: int = 8,
        ir_drop_coeff: float = 0.1,
        online_hadamard: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.get_r1 = get_r1
        self.apply_r1 = apply_r1
        self.get_r2 = get_r2
        self.apply_r2 = apply_r2
        self.head_dim = head_dim
        self.num_bits = num_bits
        self.ir_drop_coeff = ir_drop_coeff
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        if online_hadamard is not None:
            self.register_buffer("online_hadamard", online_hadamard.detach())
        else:
            self.online_hadamard = None

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base_linear.weight

    @property
    def bias(self) -> Optional[torch.nn.Parameter]:
        return self.base_linear.bias

    def _effective_weight_and_bias(self):
        return build_runtime_linear_weight_and_bias(
            self.base_linear.weight,
            self.base_linear.bias,
            r1=self.get_r1(),
            apply_r1=self.apply_r1,
            r2=self.get_r2() if self.get_r2 is not None else None,
            apply_r2=self.apply_r2,
            head_dim=self.head_dim,
        )

    def _quantize_ste(self, w: torch.Tensor) -> torch.Tensor:
        """Symmetric per-tensor 8-bit quantization with straight-through gradient."""
        n_levels = 2 ** (self.num_bits - 1) - 1
        scale = w.abs().max() / n_levels
        if scale == 0:
            return w
        w_q = (w / scale).clamp(-n_levels, n_levels).round() * scale
        return w + (w_q - w).detach()

    def _apply_ir_drop(self, w: torch.Tensor) -> torch.Tensor:
        """Scale each column by 1 - ir_drop_coeff * j/n_cols.

        Models the voltage attenuation seen by devices further from the driver.
        Fully differentiable — gradient flows through the column scaling to W and R.
        """
        if self.ir_drop_coeff == 0.0:
            return w
        n_cols = w.shape[1]
        j = torch.arange(n_cols, device=w.device, dtype=w.dtype)
        attenuation = 1.0 - self.ir_drop_coeff * j / (n_cols - 1)
        return w * attenuation.unsqueeze(0)

    def _quantize_input_ste(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric per-token DAC quantization with straight-through gradient.

        Models the input DAC with per-vector bound management: each token's max
        sets the DAC range for that token. Rotations flatten per-token outliers,
        which directly improves the SNR seen by this quantizer.
        """
        n_levels = 2 ** (self.num_bits - 1) - 1
        scale = x.abs().amax(dim=-1, keepdim=True) / n_levels
        scale = scale.clamp(min=1e-8)
        x_q = (x / scale).clamp(-n_levels, n_levels).round() * scale
        return x + (x_q - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, bias = self._effective_weight_and_bias()
        if self.online_hadamard is not None:
            w = _apply_block_hadamard(w, self.online_hadamard)
            x = _apply_block_hadamard(x, self.online_hadamard)
        w = self._quantize_ste(w)
        w = self._apply_ir_drop(w)
        x = self._quantize_input_ste(x)
        return F.linear(x, w, bias)


def enable_analog_rotations(
    model: torch.nn.Module,
    rotation_parameters: RotationParameters,
    num_bits: int = 8,
    ir_drop_coeff: float = 0.1,
    online_hadamards: bool = True,
) -> None:
    """Replace all attention/MLP linear layers with AnalogRotatedLinear modules.

    When `online_hadamards` is True, attach SpinQuant-style fixed Hadamards:
      - R3: per-head head_dim Hadamard, applied online to o_proj input.
      - R4: block-Hadamard with block_size = largest_pow2_divisor(intermediate_size),
        applied online to down_proj input.
    These flatten activation outliers at the two points R1/R2 don't reach.
    """
    from src.runtime_rotation import RuntimeRotatedEmbedding

    model.runtime_rotation_parameters = rotation_parameters
    model.model.embed_tokens = RuntimeRotatedEmbedding(
        model.model.embed_tokens,
        get_r1=lambda p=rotation_parameters: p.R1,
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

    def make(base, apply_r1, apply_r2=None, get_r2=None, online_hadamard=None):
        return AnalogRotatedLinear(
            base,
            get_r1=lambda p=rotation_parameters: p.R1,
            apply_r1=apply_r1,
            get_r2=get_r2,
            apply_r2=apply_r2,
            head_dim=head_dim if apply_r2 else None,
            num_bits=num_bits,
            ir_drop_coeff=ir_drop_coeff,
            online_hadamard=online_hadamard,
        )

    for layer_idx, layer in enumerate(model.model.layers):
        get_r2 = lambda idx=layer_idx, p=rotation_parameters: p.get_layer_r2(idx)

        layer.self_attn.q_proj = make(layer.self_attn.q_proj, "input")
        layer.self_attn.k_proj = make(layer.self_attn.k_proj, "input")
        layer.self_attn.v_proj = make(layer.self_attn.v_proj, "input", "output", get_r2)
        layer.self_attn.o_proj = make(
            layer.self_attn.o_proj, "output", "input", get_r2,
            online_hadamard=r3_hadamard,
        )
        layer.mlp.up_proj   = make(layer.mlp.up_proj,   "input")
        layer.mlp.gate_proj = make(layer.mlp.gate_proj, "input")
        layer.mlp.down_proj = make(
            layer.mlp.down_proj, "output",
            online_hadamard=r4_hadamard,
        )

    model.lm_head = make(model.lm_head, "input")


def _freeze_non_rotation_params(
    model: torch.nn.Module,
    runtime_params: RotationParameters,
    train_r2: bool,
) -> None:
    trainable_ids = {id(runtime_params.R1)}
    if train_r2:
        trainable_ids.update(id(p) for p in runtime_params.layer_R2.values())
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
    return "\n\n".join(t for t in (ex["text"] for ex in ds) if t.strip())


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
        raise ValueError(f"Dataset {dataset} has only {n_chunks} chunks of length {max_length}; need >= batch_size={batch_size}.")
    chunks = tokens[: n_chunks * max_length].view(n_chunks, max_length)
    print(f"Loaded {dataset}: {tokens.shape[0]} tokens -> {n_chunks} chunks of length {max_length}")

    rng = random.Random(seed)
    while True:
        idx = [rng.randrange(n_chunks) for _ in range(batch_size)]
        batch = chunks[idx].to(device)
        labels = batch.clone()
        attn_mask = torch.ones_like(batch)
        yield batch, labels, attn_mask


def train_full_analog(config: TrainFullAnalogConfig) -> dict:
    if config.use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            mode=WANDB_MODE,
            name=f"analog_{config.init_mode}_bits={config.num_bits}_ir={config.ir_drop_coeff}_bs={config.batch_size}_had={int(config.online_hadamards)}",
            config={
                "model_name": config.model_name,
                "init_mode": config.init_mode,
                "train_r2": config.train_r2,
                "lr": config.lr,
                "momentum": config.momentum,
                "num_steps": config.num_steps,
                "num_bits": config.num_bits,
                "ir_drop_coeff": config.ir_drop_coeff,
                "seed": config.seed,
                "batch_size": config.batch_size,
                "max_length": config.max_length,
                "dataset": config.dataset,
                "online_hadamards": config.online_hadamards,
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

    from src.rotation_utils import get_rotation_matrix
    from src.rotation_precision import ROTATION_COMPUTE_DTYPE
    rotation_params = RotationParameters.for_model(
        model,
        rotate_mode=config.init_mode,
        r2_mode=config.r2_mode,
        seed=config.seed,
        r2_seed_offset=config.r2_seed_offset,
    )

    enable_analog_rotations(
        model,
        rotation_parameters=rotation_params,
        num_bits=config.num_bits,
        ir_drop_coeff=config.ir_drop_coeff,
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
    for step in range(config.num_steps):
        input_ids, labels, attn_mask = next(batches)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        analog_loss = outputs.loss

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
        optimizer.step()

        r1_grad_norm = float(rotation_params.R1.grad.norm()) if rotation_params.R1.grad is not None else float("nan")
        eye = torch.eye(rotation_params.R1.shape[0], device=rotation_params.R1.device, dtype=rotation_params.R1.dtype)
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
            r2_grads = [float(p.grad.norm()) for p in r2_params if p.grad is not None]
            eye_r2 = torch.eye(r2_params[0].shape[0], device=r2_params[0].device, dtype=r2_params[0].dtype)
            r2_dists = [float((p.detach() - eye_r2).norm()) for p in r2_params]
            record["r2_grad_norm_mean"] = float(torch.tensor(r2_grads).mean()) if r2_grads else float("nan")
            record["r2_dist_mean"] = float(torch.tensor(r2_dists).mean())

        history.append(record)
        if config.use_wandb:
            wandb.log(record)

        if step % config.log_every == 0:
            r2_info = (
                f"  r2_grad={record['r2_grad_norm_mean']:.3e}  |R2-I|={record['r2_dist_mean']:.3e}"
                if config.train_r2 else ""
            )
            print(
                f"step {step:4d}  analog_lm={record['analog_lm_loss']:.4f}  "
                f"r1_grad={r1_grad_norm:.3e}  |R1-I|={r1_dist:.3e}{r2_info}"
            )

    if config.use_wandb:
        wandb.finish()

    if best_state is None:
        best_state = {
            "R1": rotation_params.R1.detach().cpu().clone(),
            "R2": {
                k: v.detach().cpu().clone()
                for k, v in rotation_params.layer_R2.items()
            }
            if config.train_r2
            else {},
        }
    result = {
        "R1": best_state["R1"],
        "R2": best_state["R2"],
        "history": history,
        "final_analog_lm_loss": history[-1]["analog_lm_loss"] if history else None,
        "final_analog_degradation": history[-1]["analog_lm_loss"] if history else None,
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

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train rotations against the analog LM loss.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON config file whose keys match TrainFullAnalogConfig fields.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--torch-dtype", default="float32", choices=["auto", *TORCH_DTYPE_CHOICES.keys()])
    parser.add_argument("--init-mode", default="identity",
                        choices=["identity", "sign_flip", "random", "hadamard", "block_hadamard", "hadamard_D"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-train-r2", action="store_true")
    parser.add_argument("--r2-mode", default=None)
    parser.add_argument("--r2-seed-offset", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.5,
                        help="SGDG step-size cap; the actual step is min(lr, 1/||grad||_1) so "
                             "1.5 hands control to the adaptive cap. SpinQuant default.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset", default="wikitext-2", choices=["wikitext-2", "wikitext-103"])
    parser.add_argument("--num-bits", type=int, default=8)
    parser.add_argument("--ir-drop-coeff", type=float, default=0.1)
    parser.add_argument("--no-online-hadamards", action="store_true",
                        help="Disable the fixed R3/R4 Hadamards on o_proj/down_proj inputs.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.config:
        config = TrainFullAnalogConfig(**_load_json_config(args.config))
    else:
        config = TrainFullAnalogConfig(
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
            num_bits=args.num_bits,
            ir_drop_coeff=args.ir_drop_coeff,
            online_hadamards=not args.no_online_hadamards,
            log_every=args.log_every,
            use_wandb=not args.no_wandb,
            checkpoint_path=args.checkpoint,
        )
    result = train_full_analog(config)
    print(f"final analog_lm_loss={result['final_analog_lm_loss']:.4f}")


if __name__ == "__main__":
    main()
