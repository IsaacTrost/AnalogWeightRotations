from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from src.llama_rotation import (
    rotate_blockwise_input_weight_tensor,
    rotate_blockwise_output_weight_tensor,
    rotate_input_weight_tensor,
    rotate_output_bias_tensor,
    rotate_output_weight_tensor,
)
from src.rotation_precision import ROTATION_COMPUTE_DTYPE
from src.rotation_utils import get_rotation_matrix


class RotationParameters(torch.nn.Module):
    """Store the trainable global `R1` and per-layer `R2` matrices used at runtime."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        rotate_mode: str = "random",
        r2_mode: Optional[str] = None,
        seed: int = 0,
        r2_seed_offset: int = 1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        head_dim = hidden_size // num_attention_heads
        resolved_r2_mode = r2_mode or rotate_mode

        self.R1 = torch.nn.Parameter(
            get_rotation_matrix(
                hidden_size,
                mode=rotate_mode,
                device=device,
                dtype=ROTATION_COMPUTE_DTYPE,
                seed=seed,
            )
        )
        self.layer_R2 = torch.nn.ParameterDict(
            {
                f"layer_{layer_idx}": torch.nn.Parameter(
                    get_rotation_matrix(
                        head_dim,
                        mode=resolved_r2_mode,
                        device=device,
                        dtype=ROTATION_COMPUTE_DTYPE,
                        seed=seed + r2_seed_offset + layer_idx,
                    )
                )
                for layer_idx in range(num_hidden_layers)
            }
        )
        self.metadata = {
            "rotate_mode": rotate_mode,
            "r2_mode": resolved_r2_mode,
            "seed": seed,
            "r2_seed_offset": r2_seed_offset,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "head_dim": head_dim,
        }

    @classmethod
    def for_model(
        cls,
        model: torch.nn.Module,
        rotate_mode: str = "random",
        r2_mode: Optional[str] = None,
        seed: int = 0,
        r2_seed_offset: int = 1,
    ) -> "RotationParameters":
        """Initialize runtime rotation parameters from a LLaMA-shaped model config."""
        return cls(
            hidden_size=model.config.hidden_size,
            num_attention_heads=model.config.num_attention_heads,
            num_hidden_layers=len(model.model.layers),
            rotate_mode=rotate_mode,
            r2_mode=r2_mode,
            seed=seed,
            r2_seed_offset=r2_seed_offset,
            device=model.model.embed_tokens.weight.device.type,
        )

    def get_layer_r2(self, layer_idx: int) -> torch.Tensor:
        """Return the trainable per-layer OV rotation for one decoder block."""
        return self.layer_R2[f"layer_{layer_idx}"]

    def export_state(self) -> Dict[str, object]:
        """Export a checkpoint-shaped view so the runtime path matches the static summary code."""
        return {
            "R1": self.R1.detach().clone(),
            "layers": {
                f"model.layers.{layer_idx}.self_attn.R2": self.get_layer_r2(layer_idx).detach().clone()
                for layer_idx in range(len(self.layer_R2))
            },
            "metadata": dict(self.metadata),
        }


def build_runtime_linear_weight_and_bias(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    r1: torch.Tensor,
    apply_r1: Optional[str],
    r2: Optional[torch.Tensor] = None,
    apply_r2: Optional[str] = None,
    head_dim: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compose the runtime weight transform so tests and wrappers share one rotation order."""
    rotated_weight = weight
    rotated_bias = bias

    if apply_r1 == "input":
        rotated_weight = rotate_input_weight_tensor(rotated_weight, r1)
    elif apply_r1 == "output":
        rotated_weight = rotate_output_weight_tensor(rotated_weight, r1)
        rotated_bias = rotate_output_bias_tensor(rotated_bias, r1)

    if apply_r2 is not None:
        if r2 is None or head_dim is None:
            raise ValueError("Blockwise runtime rotations require both r2 and head_dim.")
        if apply_r2 == "output":
            rotated_weight = rotate_blockwise_output_weight_tensor(rotated_weight, r2, head_dim)
        elif apply_r2 == "input":
            rotated_weight = rotate_blockwise_input_weight_tensor(rotated_weight, r2, head_dim)
        else:
            raise ValueError(f"Unknown blockwise runtime rotation: {apply_r2}")

    return rotated_weight, rotated_bias


class RuntimeRotatedLinear(torch.nn.Module):
    """Apply fixed or trainable rotations at forward time while keeping the base weight unchanged."""

    def __init__(
        self,
        base_linear: torch.nn.Linear,
        get_r1: Callable[[], torch.Tensor],
        apply_r1: Optional[str] = None,
        get_r2: Optional[Callable[[], torch.Tensor]] = None,
        apply_r2: Optional[str] = None,
        head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.get_r1 = get_r1
        self.apply_r1 = apply_r1
        self.get_r2 = get_r2
        self.apply_r2 = apply_r2
        self.head_dim = head_dim
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

    @property
    def weight(self) -> torch.nn.Parameter:
        """Expose the wrapped weight so downstream utilities can still inspect the module."""
        return self.base_linear.weight

    @property
    def bias(self) -> Optional[torch.nn.Parameter]:
        """Expose the wrapped bias so downstream utilities can still inspect the module."""
        return self.base_linear.bias

    def _rotated_weight_and_bias(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build the effective weight used by the current forward pass."""
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
        """Run the wrapped linear layer with the effective rotated weight."""
        weight, bias = self._rotated_weight_and_bias()
        return F.linear(inputs, weight, bias)


class RuntimeRotatedEmbedding(torch.nn.Module):
    """Apply the runtime residual rotation after lookup so embedding outputs match the static path."""

    def __init__(
        self,
        base_embedding: torch.nn.Embedding,
        get_r1: Callable[[], torch.Tensor],
    ) -> None:
        super().__init__()
        self.base_embedding = base_embedding
        self.get_r1 = get_r1
        self.num_embeddings = base_embedding.num_embeddings
        self.embedding_dim = base_embedding.embedding_dim
        self.padding_idx = base_embedding.padding_idx
        self.max_norm = base_embedding.max_norm
        self.norm_type = base_embedding.norm_type
        self.scale_grad_by_freq = base_embedding.scale_grad_by_freq
        self.sparse = base_embedding.sparse

    @property
    def weight(self) -> torch.nn.Parameter:
        """Expose the wrapped table so tied-weight logic and inspection still work."""
        return self.base_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Rotate the looked-up hidden states into the runtime residual basis."""
        embedded = self.base_embedding(input_ids)
        rotated = embedded.to(dtype=ROTATION_COMPUTE_DTYPE) @ self.get_r1().to(
            dtype=ROTATION_COMPUTE_DTYPE
        )
        return rotated.to(embedded.dtype)


def enable_runtime_attention_rotations(
    model: torch.nn.Module,
    rotation_parameters: Optional[RotationParameters] = None,
    rotate_mode: str = "random",
    r2_mode: Optional[str] = None,
    seed: int = 0,
    r2_seed_offset: int = 1,
) -> Dict[str, object]:
    """Wrap the attention projections so runtime-applied `R1` and `R2` are used in forward()."""
    runtime_parameters = rotation_parameters or RotationParameters.for_model(
        model,
        rotate_mode=rotate_mode,
        r2_mode=r2_mode,
        seed=seed,
        r2_seed_offset=r2_seed_offset,
    )
    model.runtime_rotation_parameters = runtime_parameters
    model.model.embed_tokens = RuntimeRotatedEmbedding(
        model.model.embed_tokens,
        get_r1=lambda params=runtime_parameters: params.R1,
    )

    head_dim = runtime_parameters.metadata["head_dim"]
    for layer_idx, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj = RuntimeRotatedLinear(
            layer.self_attn.q_proj,
            get_r1=lambda params=runtime_parameters: params.R1,
            apply_r1="input",
        )
        layer.self_attn.k_proj = RuntimeRotatedLinear(
            layer.self_attn.k_proj,
            get_r1=lambda params=runtime_parameters: params.R1,
            apply_r1="input",
        )
        layer.self_attn.v_proj = RuntimeRotatedLinear(
            layer.self_attn.v_proj,
            get_r1=lambda params=runtime_parameters: params.R1,
            apply_r1="input",
            get_r2=lambda idx=layer_idx, params=runtime_parameters: params.get_layer_r2(idx),
            apply_r2="output",
            head_dim=head_dim,
        )
        layer.self_attn.o_proj = RuntimeRotatedLinear(
            layer.self_attn.o_proj,
            get_r1=lambda params=runtime_parameters: params.R1,
            apply_r1="output",
            get_r2=lambda idx=layer_idx, params=runtime_parameters: params.get_layer_r2(idx),
            apply_r2="input",
            head_dim=head_dim,
        )
        layer.mlp.up_proj = RuntimeRotatedLinear(
            layer.mlp.up_proj,
            get_r1=lambda params=runtime_parameters: params.R1,
            apply_r1="input",
        )
        layer.mlp.gate_proj = RuntimeRotatedLinear(
            layer.mlp.gate_proj,
            get_r1=lambda params=runtime_parameters: params.R1,
            apply_r1="input",
        )
        layer.mlp.down_proj = RuntimeRotatedLinear(
            layer.mlp.down_proj,
            get_r1=lambda params=runtime_parameters: params.R1,
            apply_r1="output",
        )

    model.lm_head = RuntimeRotatedLinear(
        model.lm_head,
        get_r1=lambda params=runtime_parameters: params.R1,
        apply_r1="input",
    )

    return runtime_parameters.export_state()
