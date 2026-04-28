"""
Train rotation R1 against aihwkit IR drop + quantization hardware simulation.

Scope mirrors train_r1.py: ALL rotated linear layers in every transformer block
contribute to the loss.

  Input projections  (q, k, v, gate, up): W_eff = W @ R
  Output projections (o_proj, down_proj): W_eff = R.T @ W

Calibration activations are captured once from the float model. They are
invariant to R (rotation equivariance cancels R through every MLP/attention
block), so capturing once from the float model is exact for any R.

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
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

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
from src.rotation_utils import get_rotation_matrix


INPUT_PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
OUTPUT_PROJ_SUFFIXES = ("o_proj", "down_proj")
ALL_ROT_SUFFIXES = INPUT_PROJ_SUFFIXES + OUTPUT_PROJ_SUFFIXES


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainAnalogConfig:
    model_name: str = DEFAULT_MODEL_NAME
    torch_dtype: Optional[torch.dtype] = torch.float32
    init_mode: str = "hadamard"
    seed: int = 0

    lr: float = 1e-2
    momentum: float = 0.9
    num_steps: int = 100
    max_length: int = 128
    texts: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_TEXTS))

    ir_drop: float = 1.0
    ir_drop_segments: int = 4
    ir_drop_v_read: float = 0.4
    inp_bits: int = 8
    out_bits: int = 8

    # TorchInferenceRPUConfigIRDropT builds [batch, 512, 512, time_steps] tensors
    # on CUDA per tile. Subsample captured tokens so each tile stays ~1 GB.
    max_tokens: int = 8

    log_every: int = 1


# ---------------------------------------------------------------------------
# Hardware config
# ---------------------------------------------------------------------------

def build_rpu_config(cfg: TrainAnalogConfig):
    from aihwkit.simulator.configs import TorchInferenceRPUConfigIRDropT
    from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType
    rpu = TorchInferenceRPUConfigIRDropT()
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

def _rot_type(name: str) -> str:
    """'output' → W_eff = R.T @ W;  'input' → W_eff = W @ R."""
    return "output" if any(name.endswith(s) for s in OUTPUT_PROJ_SUFFIXES) else "input"


def build_analog_tiles(
    frozen_weights: List[Tuple[str, torch.Tensor]],
    rpu_config,
    device: str,
) -> List[Tuple[str, "AnalogLinear", torch.Tensor, str]]:
    """
    One AnalogLinear per rotated layer.
    Returns list of (layer_name, analog_layer, W_frozen, rot_type).
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
        tiles.append((name, layer, W, _rot_type(name)))
    return tiles


# ---------------------------------------------------------------------------
# Weight injection (single tile or TileModuleArray)
# ---------------------------------------------------------------------------

def _load_weight_into_tile(analog_layer, W_eff: torch.Tensor) -> None:
    """
    Replace tile weight Parameters with slices of W_eff, keeping the autograd chain.
    W_eff: [out_f, in_f] — same convention as nn.Linear.weight.

    Small layers use a single tile at analog_module.tile.weight.
    Large layers (> 512 in either dim) use a TileModuleArray with sub-tiles at
    analog_module.array.{row}.{col}.tile.weight, each covering a 512×512 block.
    """
    am = analog_layer.analog_module

    if hasattr(am, "tile"):
        am.tile.weight = nn.Parameter(W_eff)
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
        setattr(parent, "weight", nn.Parameter(W_slice))


# ---------------------------------------------------------------------------
# Training step (per-layer gradient accumulation)
# ---------------------------------------------------------------------------

def analog_train_step(
    R: torch.Tensor,
    tiles: List[Tuple[str, "AnalogLinear", torch.Tensor, str]],
    inputs: dict,
    max_tokens: int,
    device: str,
) -> float:
    """
    Accumulate gradients into R.grad by calling .backward() once per layer.

    This frees each layer's 4-D Thevenin tensors immediately, keeping peak
    VRAM to ~1 GB per tile rather than holding all layers simultaneously.

    Returns the mean loss value over all processed layers (for logging only;
    the accumulated gradient is already in R.grad before this returns).
    """
    n_layers = sum(1 for name, _, _, _ in tiles if name in inputs)
    if n_layers == 0:
        return 0.0

    total_loss = 0.0

    for name, analog_layer, W_frozen, rot_type in tiles:
        x = inputs.get(name)
        if x is None:
            continue

        x = x.to(device=device, dtype=R.dtype)
        if x.shape[0] > max_tokens:
            idx = torch.randperm(x.shape[0], device=x.device)[:max_tokens]
            x = x[idx]

        W = W_frozen.to(device=device, dtype=R.dtype)
        W_eff = (W @ R) if rot_type == "input" else (R.T @ W)

        _load_weight_into_tile(analog_layer, W_eff)

        y_analog = analog_layer(x)
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

    hidden_size = model.config.hidden_size

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

    R_init = get_rotation_matrix(
        hidden_size,
        mode=config.init_mode,
        device=device,
        dtype=torch.float32,
        seed=config.seed,
    )
    R = nn.Parameter(R_init.clone())

    optimizer = SGDG([R], lr=config.lr, momentum=config.momentum, stiefel=True)

    history = []
    for step in range(config.num_steps):
        optimizer.zero_grad(set_to_none=True)

        loss = analog_train_step(R, tiles, inputs, config.max_tokens, device)

        grad_norm  = float(R.grad.norm())       if R.grad is not None else float("nan")
        riem_grad  = float((R.grad - R.grad.T).norm() / 2) if R.grad is not None else float("nan")

        optimizer.step()

        record = {
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "riem_grad": riem_grad,
        }
        history.append(record)
        if step % config.log_every == 0:
            print(
                f"step {step:4d}  loss={loss:.6f}  "
                f"grad={grad_norm:.3e}  riem={riem_grad:.3e}"
            )

    return {
        "R": R.detach().cpu(),
        "history": history,
        "final_loss": history[-1]["loss"] if history else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train R1 against aihwkit IR drop + quantization (all rotated layers)."
    )
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--torch-dtype", default="float32",
                   choices=["auto", *TORCH_DTYPE_CHOICES.keys()])
    p.add_argument("--init-mode", default="hadamard",
                   choices=["identity", "sign_flip", "random", "hadamard",
                            "block_hadamard", "hadamard_D"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--ir-drop", type=float, default=1.0)
    p.add_argument("--ir-drop-segments", type=int, default=4)
    p.add_argument("--ir-drop-v-read", type=float, default=0.4)
    p.add_argument("--inp-bits", type=int, default=8)
    p.add_argument("--out-bits", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=8)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    config = TrainAnalogConfig(
        model_name=args.model_name,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        init_mode=args.init_mode,
        seed=args.seed,
        lr=args.lr,
        momentum=args.momentum,
        num_steps=args.num_steps,
        max_length=args.max_length,
        ir_drop=args.ir_drop,
        ir_drop_segments=args.ir_drop_segments,
        ir_drop_v_read=args.ir_drop_v_read,
        inp_bits=args.inp_bits,
        out_bits=args.out_bits,
        max_tokens=args.max_tokens,
        log_every=args.log_every,
    )
    result = train_analog(config)
    print(f"final loss={result['final_loss']:.6f}")


if __name__ == "__main__":
    main()
