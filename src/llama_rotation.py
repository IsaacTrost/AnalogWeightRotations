from typing import Dict, Optional

import torch

from src.rotation_precision import ROTATION_COMPUTE_DTYPE
from src.rotation_utils import get_rotation_matrix


def _rotate_input_weight(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate a linear map that consumes the hidden dimension on its input side."""
    rotated = weight.data.to(dtype=ROTATION_COMPUTE_DTYPE) @ rotation.to(dtype=ROTATION_COMPUTE_DTYPE)
    return rotated.to(weight.dtype)


def _rotate_output_weight(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate a linear map that writes back into the hidden dimension."""
    rotated = rotation.to(dtype=ROTATION_COMPUTE_DTYPE).T @ weight.data.to(dtype=ROTATION_COMPUTE_DTYPE)
    return rotated.to(weight.dtype)


def _rotate_output_bias(bias: Optional[torch.Tensor], rotation: torch.Tensor) -> Optional[torch.Tensor]:
    """Rotate a bias vector that lives in the residual hidden space."""
    if bias is None:
        return None
    rotated = rotation.to(dtype=ROTATION_COMPUTE_DTYPE).T @ bias.data.to(dtype=ROTATION_COMPUTE_DTYPE)
    return rotated.to(bias.dtype)


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


@torch.inference_mode()
def rotate_model(
    model: torch.nn.Module,
    rotation: Optional[torch.Tensor] = None,
    rotate_mode: str = "random",
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Apply a hidden-state basis change across the full LLaMA residual path."""
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

    rotate_embeddings(model, rotation)
    rotate_head(model, rotation)

    for layer in model.model.layers:
        rotate_attention_inputs(layer, rotation)
        rotate_attention_output(layer, rotation)
        rotate_mlp_input(layer, rotation)
        rotate_mlp_output(layer, rotation)

    return {"R1": rotation}
