"""
Train R1/R2 rotations against aihwkit IR drop + quantization hardware simulation.

Scope mirrors train_r1.py: ALL rotated linear layers in every transformer block
contribute to the loss.

  R1 input projections  (q, k, v, gate, up): W_eff = W @ R1
  R1 output projections (o_proj, down_proj): W_eff = R1.T @ W
  R2 OV path            (v, o_proj): blockwise per-layer head rotation

Calibration activations are captured once from the float model. This is exact
for R1 and a layerwise proxy for R2; a fully exact R2 objective would evaluate
the paired V/O path or the full forward pass.

Memory strategy: we call .backward() once per layer (gradient accumulation)
so the large Thevenin 4-D tensors [batch, 512, 512, time_steps] (~1 GB each)
from TorchInferenceRPUConfigIRDropT are freed immediately rather than kept
alive for a single backward through all 150+ layers simultaneously.

The gradient flows through both the analog tile output and the float ideal:
  loss_l = ||y_analog_l(W_eff) - y_ideal_l(W_eff)||^2 / ||y_ideal_l||^2
Both terms depend on R through W_eff, so the gradient captures how R
shifts the hardware error, not just the float output.

Optimizer: SGDG with Cayley retraction (keeps R on the Stiefel manifold).
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb

from src.hardware_configs import build_rpu_config as build_hardware_rpu_config
from src.llama_model import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXTS,
    TORCH_DTYPE_CHOICES,
    build_inputs,
    get_default_device,
    is_llama_like_model,
    load_model_and_tokenizer,
    resolve_torch_dtype,
)
from src.llama_prepare import prepare_model_for_rotation
from src.optimizer import SGDG
from src.runtime_rotation import (
    RotationParameters,
    build_runtime_linear_weight_and_bias,
)
from src.wandb_config import WANDB_ENTITY, WANDB_MODE


INPUT_PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
OUTPUT_PROJ_SUFFIXES = ("o_proj", "down_proj")
ALL_ROT_SUFFIXES = INPUT_PROJ_SUFFIXES + OUTPUT_PROJ_SUFFIXES
AIMC_WANDB_PROJECT = "aimc-rotations"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainAnalogConfig:
    model_name: str = DEFAULT_MODEL_NAME
    torch_dtype: Optional[torch.dtype] = torch.float32
    init_mode: str = "hadamard"
    seed: int = 0
    train_r2: bool = True
    r2_mode: Optional[str] = None
    r2_seed_offset: int = 1

    lr: float = 1.5
    momentum: float = 0.9
    num_steps: int = 100
    max_length: int = 128
    texts: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_TEXTS))

    hardware_preset: str = "advanced_ir_drop_8bit"
    ir_drop: float = 1.0
    ir_drop_segments: int = 4
    ir_drop_v_read: float = 0.4
    inp_bits: int = 8
    out_bits: int = 8

    # TorchInferenceRPUConfigIRDropT builds [batch, 512, 512, time_steps] tensors
    # on CUDA per tile. Subsample captured tokens so each tile stays ~1 GB.
    max_tokens: int = 8

    log_every: int = 1
    use_wandb: bool = True
    wandb_project: str = AIMC_WANDB_PROJECT
    wandb_run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None


@dataclass
class AnalogTileSpec:
    name: str
    analog_layer: nn.Module
    weight: torch.Tensor
    apply_r1: str
    layer_idx: Optional[int] = None
    apply_r2: Optional[str] = None


# ---------------------------------------------------------------------------
# Hardware config
# ---------------------------------------------------------------------------

def build_rpu_config(cfg: TrainAnalogConfig):
    from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType
    rpu = build_hardware_rpu_config(cfg.hardware_preset)
    rpu.forward.ir_drop          = cfg.ir_drop
    rpu.forward.ir_drop_segments = cfg.ir_drop_segments
    rpu.forward.ir_drop_v_read   = cfg.ir_drop_v_read
    rpu.forward.inp_res          = 2 ** cfg.inp_bits - 2
    rpu.forward.out_res          = 2 ** cfg.out_bits - 2
    rpu.forward.bound_management = BoundManagementType.NONE
    rpu.forward.noise_management = NoiseManagementType.NONE
    return rpu


# ---------------------------------------------------------------------------
# Capture calibration inputs for all rotated layers
# ---------------------------------------------------------------------------

def capture_layer_inputs(
    model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    device: str,
) -> dict:
    """
    Run one float forward pass and record the input to every rotated Linear.
    Returns {layer_name: Tensor[tokens, in_dim]}.
    Inputs are invariant to R (rotation equivariance), so one capture suffices.
    """
    captured: dict = {}
    handles = []

    for name, module in model.named_modules():
        if not any(name.endswith(s) for s in ALL_ROT_SUFFIXES):
            continue
        if not isinstance(module, nn.Linear):
            continue

        def _make_hook(n):
            def _hook(mod, inp, out):
                captured[n] = inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu().float()
            return _hook

        handles.append(module.register_forward_hook(_make_hook(name)))

    encoded = build_inputs(tokenizer, texts=list(texts), device=device, max_length=max_length)
    with torch.no_grad():
        model(**encoded)

    for h in handles:
        h.remove()

    return captured


# ---------------------------------------------------------------------------
# Analog tile construction
# ---------------------------------------------------------------------------

def _layer_idx_from_name(name: str) -> Optional[int]:
    if ".layers." not in name:
        return None
    return int(name.split(".layers.", 1)[1].split(".", 1)[0])


def _rotation_sides(name: str) -> Tuple[str, Optional[str], Optional[int]]:
    """Return (R1 side, R2 side, layer_idx) for one LLaMA projection."""
    if name.endswith("v_proj"):
        return "input", "output", _layer_idx_from_name(name)
    if name.endswith("o_proj"):
        return "output", "input", _layer_idx_from_name(name)
    if name.endswith(OUTPUT_PROJ_SUFFIXES):
        return "output", None, None
    return "input", None, None


def build_analog_tiles(
    frozen_weights: List[Tuple[str, torch.Tensor]],
    rpu_config,
    device: str,
) -> List[AnalogTileSpec]:
    """
    One AnalogLinear per rotated layer.
    Returns one tile spec per rotated layer.
    """
    from aihwkit.nn import AnalogLinear
    import copy

    tiles = []
    for name, W in frozen_weights:
        out_f, in_f = W.shape
        layer = AnalogLinear(in_f, out_f, bias=False, rpu_config=copy.deepcopy(rpu_config))
        for _, tile in layer.named_analog_layers():
            tile.set_weights(W.cpu().float())
        layer.to(device=device, dtype=torch.float32)
        layer.eval()
        apply_r1, apply_r2, layer_idx = _rotation_sides(name)
        tiles.append(
            AnalogTileSpec(
                name=name,
                analog_layer=layer,
                weight=W,
                apply_r1=apply_r1,
                apply_r2=apply_r2,
                layer_idx=layer_idx,
            )
        )
    return tiles


# ---------------------------------------------------------------------------
# Weight injection (single tile or TileModuleArray)
# ---------------------------------------------------------------------------

def _load_weight_into_tile(analog_layer, W_eff: torch.Tensor) -> None:
    """
    Replace tile weights with slices of W_eff while keeping the autograd chain.
    W_eff: [out_f, in_f] — same convention as nn.Linear.weight.

    Small layers use a single tile at analog_module.tile.weight.
    Large layers (> 512 in either dim) use a TileModuleArray with sub-tiles at
    analog_module.array.{row}.{col}.tile.weight, each covering a 512×512 block.
    """
    am = analog_layer.analog_module

    if hasattr(am, "tile"):
        am.tile._parameters["weight"] = W_eff
        return

    tile_h = tile_w = None
    for pname, param in list(am.named_parameters(recurse=True)):
        if not pname.endswith("tile.weight"):
            continue
        parts = pname.split(".")
        try:
            row_idx, col_idx = int(parts[1]), int(parts[2])
        except (IndexError, ValueError):
            continue

        if tile_h is None:
            tile_h, tile_w = param.shape

        r0, c0 = row_idx * tile_h, col_idx * tile_w
        r1 = min(r0 + tile_h, W_eff.shape[0])
        c1 = min(c0 + tile_w, W_eff.shape[1])
        W_slice = W_eff[r0:r1, c0:c1]

        if W_slice.shape != (tile_h, tile_w):
            pad = W_eff.new_zeros(tile_h, tile_w)
            pad[: r1 - r0, : c1 - c0] = W_slice
            W_slice = pad

        parent = am
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        parent._parameters["weight"] = W_slice


# ---------------------------------------------------------------------------
# Training step (per-layer gradient accumulation)
# ---------------------------------------------------------------------------

def analog_train_step(
    rotation_params: RotationParameters,
    tiles: List[AnalogTileSpec],
    inputs: dict,
    max_tokens: int,
    device: str,
) -> float:
    """
    Accumulate gradients into R1/R2 by calling .backward() once per layer.

    This frees each layer's 4-D Thevenin tensors immediately, keeping peak
    VRAM to ~1 GB per tile rather than holding all layers simultaneously.

    Returns the mean loss value over all processed layers (for logging only;
    the accumulated gradient is already in R1/R2 before this returns).
    """
    n_layers = sum(1 for spec in tiles if spec.name in inputs)
    if n_layers == 0:
        return 0.0

    total_loss = 0.0

    head_dim = rotation_params.metadata["head_dim"]
    r1 = rotation_params.R1

    for spec in tiles:
        x = inputs.get(spec.name)
        if x is None:
            continue

        x = x.to(device=device, dtype=torch.float32)
        if x.shape[0] > max_tokens:
            idx = torch.randperm(x.shape[0], device=x.device)[:max_tokens]
            x = x[idx]

        W = spec.weight.to(device=device, dtype=torch.float32)
        r2 = None
        if spec.apply_r2 is not None:
            if spec.layer_idx is None:
                raise ValueError(f"{spec.name} needs an R2 layer index.")
            r2 = rotation_params.get_layer_r2(spec.layer_idx)
        W_eff, _ = build_runtime_linear_weight_and_bias(
            W,
            None,
            r1=r1,
            apply_r1=spec.apply_r1,
            r2=r2,
            apply_r2=spec.apply_r2,
            head_dim=head_dim if spec.apply_r2 is not None else None,
        )
        W_eff = W_eff.to(dtype=torch.float32)

        _load_weight_into_tile(spec.analog_layer, W_eff)

        y_analog = spec.analog_layer(x)
        y_ideal  = F.linear(x, W_eff)

        denom      = y_ideal.detach().pow(2).mean() + 1e-8
        layer_loss = (y_analog - y_ideal).pow(2).mean() / denom

        # Divide by n_layers so the accumulated gradient equals the mean gradient
        (layer_loss / n_layers).backward()

        total_loss += float(layer_loss.detach())

    return total_loss / n_layers


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_analog(config: TrainAnalogConfig) -> dict:
    if config.use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            entity=WANDB_ENTITY,
            project=config.wandb_project,
            mode=WANDB_MODE,
            name=config.wandb_run_name or (
                f"aihwkit_r1_r2_{config.init_mode}"
                f"_r2={config.r2_mode or config.init_mode}"
                f"_lr={config.lr:g}"
                f"_tokens={config.max_tokens}"
                f"_ir={config.ir_drop:g}"
            ),
            config={
                "model_name": config.model_name,
                "torch_dtype": str(config.torch_dtype),
                "init_mode": config.init_mode,
                "seed": config.seed,
                "train_r2": config.train_r2,
                "r2_mode": config.r2_mode or config.init_mode,
                "r2_seed_offset": config.r2_seed_offset,
                "lr": config.lr,
                "momentum": config.momentum,
                "num_steps": config.num_steps,
                "max_length": config.max_length,
                "hardware_preset": config.hardware_preset,
                "ir_drop": config.ir_drop,
                "ir_drop_segments": config.ir_drop_segments,
                "ir_drop_v_read": config.ir_drop_v_read,
                "inp_bits": config.inp_bits,
                "out_bits": config.out_bits,
                "max_tokens": config.max_tokens,
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
        raise ValueError(f"{config.model_name} does not look like a LLaMA model.")

    prepare_model_for_rotation(model)
    model.eval()

    rotation_params = RotationParameters.for_model(
        model,
        rotate_mode=config.init_mode,
        r2_mode=config.r2_mode if config.train_r2 else "identity",
        seed=config.seed,
        r2_seed_offset=config.r2_seed_offset,
    )
    rotation_params.to(device=device, dtype=torch.float32)

    frozen_weights: List[Tuple[str, torch.Tensor]] = []
    for name, module in model.named_modules():
        if any(name.endswith(s) for s in ALL_ROT_SUFFIXES) and isinstance(module, nn.Linear):
            frozen_weights.append((name, module.weight.data.detach().clone().float()))

    print(f"Target layers: {len(frozen_weights)} rotated linear layers across all transformer blocks")

    print("Capturing calibration inputs ...")
    inputs = capture_layer_inputs(
        model, tokenizer,
        texts=list(config.texts),
        max_length=config.max_length,
        device=device,
    )
    print(f"  Captured {len(inputs)} layers")

    # Release the float model — inputs and frozen weights are all we need from here on.
    del model
    torch.cuda.empty_cache()

    rpu_config = build_rpu_config(config)
    print("Building analog tiles ...")
    tiles = build_analog_tiles(frozen_weights, rpu_config, device=device)
    print(f"  Built {len(tiles)} analog tiles")

    params_to_train = [rotation_params.R1]
    if config.train_r2:
        params_to_train += list(rotation_params.layer_R2.values())

    optimizer = SGDG(params_to_train, lr=config.lr, momentum=config.momentum, stiefel=True)

    history = []
    for step in range(config.num_steps):
        optimizer.zero_grad(set_to_none=True)

        loss = analog_train_step(rotation_params, tiles, inputs, config.max_tokens, device)

        r1_grad_norm = (
            float(rotation_params.R1.grad.norm())
            if rotation_params.R1.grad is not None else float("nan")
        )
        r1_riem_grad = (
            float((rotation_params.R1.grad - rotation_params.R1.grad.T).norm() / 2)
            if rotation_params.R1.grad is not None else float("nan")
        )

        optimizer.step()

        record = {
            "step": step,
            "loss": loss,
            "r1_grad_norm": r1_grad_norm,
            "r1_riem_grad": r1_riem_grad,
        }
        if config.train_r2:
            r2_params = list(rotation_params.layer_R2.values())
            r2_grads = [float(p.grad.norm()) for p in r2_params if p.grad is not None]
            record["r2_grad_norm_mean"] = (
                float(torch.tensor(r2_grads).mean()) if r2_grads else float("nan")
            )
        history.append(record)
        if config.use_wandb:
            wandb.log(record)
        if step % config.log_every == 0:
            r2_info = (
                f"  r2_grad={record['r2_grad_norm_mean']:.3e}"
                if config.train_r2 else ""
            )
            print(
                f"step {step:4d}  loss={loss:.6f}  "
                f"r1_grad={r1_grad_norm:.3e}  r1_riem={r1_riem_grad:.3e}{r2_info}"
            )

    result = {
        "R": rotation_params.R1.detach().cpu(),
        "R1": rotation_params.R1.detach().cpu(),
        "R2": {k: v.detach().cpu() for k, v in rotation_params.layer_R2.items()} if config.train_r2 else {},
        "history": history,
        "final_loss": history[-1]["loss"] if history else None,
    }
    if config.checkpoint_path:
        os.makedirs(os.path.dirname(os.path.abspath(config.checkpoint_path)), exist_ok=True)
        torch.save({"R1": result["R1"], "R2": result["R2"]}, config.checkpoint_path)
        print(f"Saved checkpoint to {config.checkpoint_path}")
        if config.use_wandb:
            wandb.save(config.checkpoint_path)
    if config.use_wandb:
        wandb.finish()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train R1/R2 against aihwkit IR drop + quantization (all rotated layers)."
    )
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--torch-dtype", default="float32",
                   choices=["auto", *TORCH_DTYPE_CHOICES.keys()])
    p.add_argument("--init-mode", default="hadamard",
                   choices=["identity", "sign_flip", "random", "hadamard",
                            "block_hadamard", "hadamard_D"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-train-r2", action="store_true", help="Train R1 only, skip R2.")
    p.add_argument("--r2-mode", default=None,
                   choices=["identity", "sign_flip", "random", "hadamard",
                            "block_hadamard", "hadamard_D"],
                   help="R2 init mode. Defaults to --init-mode.")
    p.add_argument("--r2-seed-offset", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--hardware-preset", default="advanced_ir_drop_8bit",
                   choices=["advanced_ir_drop_8bit", "advanced_ir_drop"])
    p.add_argument("--ir-drop", type=float, default=1.0)
    p.add_argument("--ir-drop-segments", type=int, default=4)
    p.add_argument("--ir-drop-v-read", type=float, default=0.4)
    p.add_argument("--inp-bits", type=int, default=8)
    p.add_argument("--out-bits", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    p.add_argument("--wandb-project", default=AIMC_WANDB_PROJECT)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--checkpoint", default=None, help="Save final R1/R2 matrices to this .pt path.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    config = TrainAnalogConfig(
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
        hardware_preset=args.hardware_preset,
        ir_drop=args.ir_drop,
        ir_drop_segments=args.ir_drop_segments,
        ir_drop_v_read=args.ir_drop_v_read,
        inp_bits=args.inp_bits,
        out_bits=args.out_bits,
        max_tokens=args.max_tokens,
        log_every=args.log_every,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        checkpoint_path=args.checkpoint,
    )
    result = train_analog(config)
    print(f"final loss={result['final_loss']:.6f}")


if __name__ == "__main__":
    main()
