from typing import Dict, Optional

import torch

from src.rotation_precision import ROTATION_COMPUTE_DTYPE
from src.rotation_utils import get_rotation_matrix


def rotate_input_weight_tensor(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate a linear map that consumes the hidden dimension on its input side."""
    rotated = weight.data.to(dtype=ROTATION_COMPUTE_DTYPE) @ rotation.to(dtype=ROTATION_COMPUTE_DTYPE)
    return rotated.to(weight.dtype)


def rotate_output_weight_tensor(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate a linear map that writes back into the hidden dimension."""
    rotated = rotation.to(dtype=ROTATION_COMPUTE_DTYPE).T @ weight.data.to(dtype=ROTATION_COMPUTE_DTYPE)
    return rotated.to(weight.dtype)


def rotate_output_bias_tensor(bias: Optional[torch.Tensor], rotation: torch.Tensor) -> Optional[torch.Tensor]:
    """Rotate a bias vector that lives in the residual hidden space."""
    if bias is None:
        return None
    rotated = rotation.to(dtype=ROTATION_COMPUTE_DTYPE).T @ bias.data.to(dtype=ROTATION_COMPUTE_DTYPE)
    return rotated.to(bias.dtype)


def _reshape_weight_blocks(weight: torch.Tensor, block_size: int, axis: str) -> torch.Tensor:
    """Group a linear weight into contiguous head-sized blocks on the requested axis."""
    if axis == "rows":
        if weight.shape[0] % block_size != 0:
            raise ValueError(f"Cannot split {weight.shape[0]} output rows into blocks of size {block_size}.")
        return weight.reshape(weight.shape[0] // block_size, block_size, weight.shape[1])
    if axis == "cols":
        if weight.shape[1] % block_size != 0:
            raise ValueError(f"Cannot split {weight.shape[1]} input cols into blocks of size {block_size}.")
        return weight.reshape(weight.shape[0], weight.shape[1] // block_size, block_size)
    raise ValueError(f"Unknown block axis: {axis}")


def rotate_blockwise_output_weight_tensor(
    weight: torch.Tensor,
    rotation: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Rotate each output block when a projection emits per-head features such as `v_proj`."""
    rotated = weight.data.to(dtype=ROTATION_COMPUTE_DTYPE)
    block_rotation = rotation.to(dtype=ROTATION_COMPUTE_DTYPE)
    blocks = _reshape_weight_blocks(rotated, block_size, axis="rows")
    rotated_blocks = torch.matmul(block_rotation.T.unsqueeze(0), blocks)
    return rotated_blocks.reshape_as(rotated).to(weight.dtype)


def rotate_blockwise_input_weight_tensor(
    weight: torch.Tensor,
    rotation: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Rotate each input block when a projection consumes concatenated head features such as `o_proj`."""
    rotated = weight.data.to(dtype=ROTATION_COMPUTE_DTYPE)
    block_rotation = rotation.to(dtype=ROTATION_COMPUTE_DTYPE)
    blocks = _reshape_weight_blocks(rotated, block_size, axis="cols")
    rotated_blocks = torch.matmul(blocks, block_rotation)
    return rotated_blocks.reshape_as(rotated).to(weight.dtype)


def _rotate_input_weight(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Keep the static rewrite helpers readable while sharing the pure tensor math."""
    return rotate_input_weight_tensor(weight, rotation)


def _rotate_output_weight(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Keep the static rewrite helpers readable while sharing the pure tensor math."""
    return rotate_output_weight_tensor(weight, rotation)


def _rotate_output_bias(bias: Optional[torch.Tensor], rotation: torch.Tensor) -> Optional[torch.Tensor]:
    """Keep the static rewrite helpers readable while sharing the pure tensor math."""
    return rotate_output_bias_tensor(bias, rotation)


def _rotate_blockwise_output_weight(weight: torch.Tensor, rotation: torch.Tensor, block_size: int) -> torch.Tensor:
    """Keep the static rewrite helpers readable while sharing the pure tensor math."""
    return rotate_blockwise_output_weight_tensor(weight, rotation, block_size)


def _rotate_blockwise_input_weight(weight: torch.Tensor, rotation: torch.Tensor, block_size: int) -> torch.Tensor:
    """Keep the static rewrite helpers readable while sharing the pure tensor math."""
    return rotate_blockwise_input_weight_tensor(weight, rotation, block_size)


def rotate_embeddings(model: torch.nn.Module, rotation: torch.Tensor) -> None:
    """Rotate token embeddings so the residual stream starts in the new basis."""
    embed = model.model.embed_tokens
    embed.weight.data = _rotate_input_weight(embed.weight, rotation)


def rotate_attention_inputs(layer: torch.nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the attention projections that read the residual stream."""
    for linear in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        linear.weight.data = _rotate_input_weight(linear.weight, rotation)


def rotate_attention_output(layer: torch.nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the attention output projection back into the rotated residual basis."""
    linear = layer.self_attn.o_proj
    linear.weight.data = _rotate_output_weight(linear.weight, rotation)
    if linear.bias is not None:
        linear.bias.data = _rotate_output_bias(linear.bias, rotation)


def rotate_mlp_input(layer: torch.nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the MLP projections that consume the residual stream."""
    for linear in [layer.mlp.up_proj, layer.mlp.gate_proj]:
        linear.weight.data = _rotate_input_weight(linear.weight, rotation)


def rotate_mlp_output(layer: torch.nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the MLP output projection back into the rotated residual basis."""
    linear = layer.mlp.down_proj
    linear.weight.data = _rotate_output_weight(linear.weight, rotation)
    if linear.bias is not None:
        linear.bias.data = _rotate_output_bias(linear.bias, rotation)


def rotate_head(model: torch.nn.Module, rotation: torch.Tensor) -> None:
    """Rotate the LM head only when it is not tied to the embedding matrix."""
    head = model.lm_head
    embed = model.model.embed_tokens
    if head.weight.data_ptr() == embed.weight.data_ptr():
        return
    head.weight.data = _rotate_input_weight(head.weight, rotation)


def rotate_ov_proj(layer: torch.nn.Module, rotation: torch.Tensor, head_dim: int) -> None:
    """Apply the paired OV rotation that can later become a trainable per-layer `R2`."""
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    v_proj.weight.data = _rotate_blockwise_output_weight(v_proj.weight, rotation, head_dim)
    o_proj.weight.data = _rotate_blockwise_input_weight(o_proj.weight, rotation, head_dim)


@torch.inference_mode()
def rotate_model(
    model: torch.nn.Module,
    rotation: Optional[torch.Tensor] = None,
    rotate_mode: str = "random",
    seed: int = 0,
    r2_mode: Optional[str] = None,
    r2_seed_offset: int = 1,
) -> Dict[str, object]:
    """Apply the residual-stream `R1` and per-layer OV `R2` rotations across a LLaMA model."""
    if rotation is None:
        rotation = get_rotation_matrix(
            model.config.hidden_size,
            mode=rotate_mode,
            device=model.model.embed_tokens.weight.device.type,
            dtype=ROTATION_COMPUTE_DTYPE,
            seed=seed,
        )
    else:
        rotation = rotation.to(device=model.model.embed_tokens.weight.device, dtype=ROTATION_COMPUTE_DTYPE)

    r2_mode = r2_mode or rotate_mode
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    rotation_state: Dict[str, object] = {
        "R1": rotation,
        "layers": {},
        "metadata": {
            "rotate_mode": rotate_mode,
            "r2_mode": r2_mode,
            "seed": seed,
            "r2_seed_offset": r2_seed_offset,
            "hidden_size": model.config.hidden_size,
            "num_attention_heads": num_heads,
            "head_dim": head_dim,
        },
    }

    rotate_embeddings(model, rotation)
    rotate_head(model, rotation)

    for layer_idx, layer in enumerate(model.model.layers):
        layer_r2 = get_rotation_matrix(
            head_dim,
            mode=r2_mode,
            device=model.model.embed_tokens.weight.device.type,
            dtype=ROTATION_COMPUTE_DTYPE,
            seed=seed + r2_seed_offset + layer_idx,
        )
        rotate_attention_inputs(layer, rotation)
        rotate_attention_output(layer, rotation)
        rotate_mlp_input(layer, rotation)
        rotate_mlp_output(layer, rotation)
        rotate_ov_proj(layer, layer_r2, head_dim)
        rotation_state["layers"][f"model.layers.{layer_idx}.self_attn.R2"] = layer_r2

    return rotation_state
