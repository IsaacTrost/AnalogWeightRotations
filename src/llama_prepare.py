from typing import Iterable

import torch

from src.rotation_precision import ROTATION_COMPUTE_DTYPE


def fuse_rmsnorm_linear(
    norm: torch.nn.Module,
    linear_layers: Iterable[torch.nn.Linear],
) -> None:
    """Fold the learned RMSNorm scale into adjacent linear weights."""
    norm_weight = norm.weight.data.to(dtype=ROTATION_COMPUTE_DTYPE)
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        fused = linear.weight.data.to(dtype=ROTATION_COMPUTE_DTYPE) * norm_weight
        linear.weight.data = fused.to(linear_dtype)


def center_embedding_weights(model: torch.nn.Module) -> None:
    """Remove the mean from each embedding row like SpinQuant's LLaMA prep stage."""
    embed = model.model.embed_tokens
    embed_dtype = embed.weight.dtype
    centered = embed.weight.data.to(dtype=ROTATION_COMPUTE_DTYPE)
    centered = centered - centered.mean(dim=-1, keepdim=True)
    embed.weight.data = centered.to(embed_dtype)


def prepare_model_for_rotation(model: torch.nn.Module) -> None:
    """Fuse RMSNorm scales so the hidden-space rotation acts on plain linear blocks."""
    center_embedding_weights(model)

    for layer in model.model.layers:
        fuse_rmsnorm_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )
        fuse_rmsnorm_linear(
            layer.post_attention_layernorm,
            [
                layer.mlp.up_proj,
                layer.mlp.gate_proj,
            ],
        )

        # After folding the learned scales forward, the norms keep only their RMS behavior.
        layer.input_layernorm.weight.data = torch.ones_like(layer.input_layernorm.weight.data)
        layer.post_attention_layernorm.weight.data = torch.ones_like(
            layer.post_attention_layernorm.weight.data
        )

    fuse_rmsnorm_linear(model.model.norm, [model.lm_head])
    model.model.norm.weight.data = torch.ones_like(model.model.norm.weight.data)
