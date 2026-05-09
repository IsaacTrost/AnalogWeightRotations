from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from src.rotation_precision import ROTATION_COMPUTE_DTYPE
from src.rotation_utils import get_rotation_matrix


class TrainableRotation(nn.Module):
    """Hold the residual-stream rotation R1 as a learnable Stiefel parameter.

    The matrix is stored in fp32 for Cayley-transform stability; callers cast
    to the model's working dtype at the point of use.
    """

    def __init__(
        self,
        dim: int,
        init_mode: str = "identity",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        init = get_rotation_matrix(
            dim,
            mode=init_mode,
            device=device.type if isinstance(device, torch.device) else (device or "cpu"),
            dtype=ROTATION_COMPUTE_DTYPE,
            seed=seed,
        )
        self.R = nn.Parameter(init.to(dtype=dtype))

    @property
    def dim(self) -> int:
        return self.R.shape[0]

    def forward(self) -> torch.Tensor:
        return self.R


def _apply_input_rotation(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Compute W @ R so the rotated input basis cancels on the way in."""
    return weight @ rotation


def _apply_output_rotation(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Compute R^T @ W so the output writes back into the rotated residual basis."""
    return rotation.T @ weight


class RotatedLinear(nn.Module):
    """Functional rotation of a frozen nn.Linear.

    The original weight and bias are held as buffers (no grad). On every forward
    the rotated weight is recomputed inside the autograd graph, so gradients
    propagate only to the shared rotation parameter.
    """

    SIDES = ("input", "output")

    def __init__(
        self,
        original: nn.Linear,
        rotation: TrainableRotation,
        side: str,
    ) -> None:
        super().__init__()
        if side not in self.SIDES:
            raise ValueError(f"side must be one of {self.SIDES}, got {side}.")
        self.side = side
        self.rotation = rotation
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.register_buffer("weight", original.weight.data.detach().clone())
        if original.bias is not None:
            self.register_buffer("bias", original.bias.data.detach().clone())
        else:
            self.bias = None

    def _rotated_matrices(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (W_eff, bias_eff) with R in the autograd graph."""
        r_full = self.rotation()
        r = r_full.to(dtype=self.weight.dtype)
        if self.side == "input":
            return _apply_input_rotation(self.weight, r), self.bias
        # output side also rotates any bias that lives in the residual space
        w_eff = _apply_output_rotation(self.weight, r)
        if self.bias is None:
            return w_eff, None
        return w_eff, r.T @ self.bias

    def effective_weight(self) -> torch.Tensor:
        """Expose the rotated weight tensor for auxiliary losses."""
        return self._rotated_matrices()[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff, b_eff = self._rotated_matrices()
        return F.linear(x, w_eff, b_eff)


class RotatedEmbedding(nn.Module):
    """Functional rotation of a frozen nn.Embedding (rotates the embedding rows)."""

    def __init__(self, original: nn.Embedding, rotation: TrainableRotation) -> None:
        super().__init__()
        self.rotation = rotation
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        self.padding_idx = original.padding_idx
        self.max_norm = original.max_norm
        self.norm_type = original.norm_type
        self.scale_grad_by_freq = original.scale_grad_by_freq
        self.sparse = original.sparse
        self.register_buffer("weight", original.weight.data.detach().clone())

    def effective_weight(self) -> torch.Tensor:
        r = self.rotation().to(dtype=self.weight.dtype)
        return self.weight @ r

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        w_eff = self.effective_weight()
        return F.embedding(
            input_ids,
            w_eff,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


def _install_rotated_linear(
    parent: nn.Module,
    attr: str,
    rotation: TrainableRotation,
    side: str,
) -> RotatedLinear:
    """Replace an attribute on `parent` with a RotatedLinear wrapper around it."""
    original = getattr(parent, attr)
    if not isinstance(original, nn.Linear):
        raise TypeError(f"{attr} on {type(parent).__name__} is not nn.Linear.")
    wrapped = RotatedLinear(original, rotation, side)
    wrapped.to(device=original.weight.device)
    setattr(parent, attr, wrapped)
    return wrapped


def install_trainable_rotation(
    model: nn.Module,
    init_mode: str = "identity",
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> TrainableRotation:
    """Swap every residual-connected projection in a LLaMA model for a rotated variant.

    All wrappers share a single TrainableRotation parameter. After this call,
    R is the only trainable tensor that matters; call `freeze_non_rotation_params`
    to make that explicit on the optimizer side.
    """
    embed = model.model.embed_tokens
    device = embed.weight.device
    rotation = TrainableRotation(
        dim=model.config.hidden_size,
        init_mode=init_mode,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    rotation.to(device=device)

    rotated_embed = RotatedEmbedding(embed, rotation)
    rotated_embed.to(device=device)
    model.model.embed_tokens = rotated_embed

    head = model.lm_head
    tied = head.weight.data_ptr() == embed.weight.data_ptr()
    if not tied:
        _install_rotated_linear(model, "lm_head", rotation, side="input")

    for layer in model.model.layers:
        _install_rotated_linear(layer.self_attn, "q_proj", rotation, side="input")
        _install_rotated_linear(layer.self_attn, "k_proj", rotation, side="input")
        _install_rotated_linear(layer.self_attn, "v_proj", rotation, side="input")
        _install_rotated_linear(layer.self_attn, "o_proj", rotation, side="output")
        _install_rotated_linear(layer.mlp, "up_proj", rotation, side="input")
        _install_rotated_linear(layer.mlp, "gate_proj", rotation, side="input")
        _install_rotated_linear(layer.mlp, "down_proj", rotation, side="output")

    return rotation


def freeze_non_rotation_params(model: nn.Module, rotation: TrainableRotation) -> None:
    """Disable grad on every parameter that is not the shared rotation R."""
    rotation_ids = {id(p) for p in rotation.parameters()}
    for param in model.parameters():
        param.requires_grad = id(param) in rotation_ids


def iter_rotated_modules(
    model: nn.Module,
    target_suffixes: Optional[Sequence[str]] = None,
) -> Iterable[Tuple[str, nn.Module]]:
    """Yield (name, module) for each installed rotated wrapper, optionally filtered."""
    suffix_tuple = tuple(target_suffixes) if target_suffixes else None
    for name, module in model.named_modules():
        if not isinstance(module, (RotatedLinear, RotatedEmbedding)):
            continue
        if suffix_tuple is not None and not name.endswith(suffix_tuple):
            continue
        yield name, module


def collect_effective_weights(
    model: nn.Module,
    target_suffixes: Optional[Sequence[str]] = None,
) -> List[Tuple[str, torch.Tensor]]:
    """Materialize the rotated weights once for auxiliary loss computation."""
    return [(name, mod.effective_weight()) for name, mod in iter_rotated_modules(model, target_suffixes)]
