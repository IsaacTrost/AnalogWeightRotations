from typing import Iterable, Sequence, Union

import torch


SetPoints = Union[float, Sequence[float]]


def _normalize_set_points(set_points: SetPoints, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Pack user-supplied set points into a 1D tensor on the weight's device/dtype."""
    if isinstance(set_points, (int, float)):
        set_points = [float(set_points)]
    return torch.tensor(list(set_points), device=device, dtype=dtype)


def _gaussian_penalty(values: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """Return mean of exp(-(W - s)^2 / (2 sigma^2)) summed over all set points.

    High where values sit on a set point, zero far from it. Gradient is smooth
    and pushes each value away from the nearest center.
    """
    diffs = values.unsqueeze(-1) - centers
    return torch.exp(-(diffs ** 2) / (2.0 * sigma ** 2)).sum(dim=-1).mean()


def _hinge_penalty(values: torch.Tensor, centers: torch.Tensor, margin: float) -> torch.Tensor:
    """Return mean of max(0, margin - min_s |W - s|).

    Flat penalty outside the forbidden zone, linear ramp inside it.
    """
    distances = torch.abs(values.unsqueeze(-1) - centers).min(dim=-1).values
    return torch.clamp(margin - distances, min=0.0).mean()


def setpoint_repulsion_loss(
    weights: Iterable[torch.Tensor],
    set_points: SetPoints,
    sigma: float = 0.05,
    kind: str = "gaussian",
    margin: float = 0.1,
) -> torch.Tensor:
    """Aux loss that pushes rotated weight values away from unsafe set points.

    Args:
        weights: iterable of rotated weight tensors (see `collect_effective_weights`).
        set_points: scalar or sequence of weight values to repel away from.
        sigma: Gaussian bandwidth (only used when kind='gaussian').
        kind: 'gaussian' or 'hinge'.
        margin: hinge half-width (only used when kind='hinge').

    Gradients flow through the weight tensors (i.e., through R) because the
    effective weights are built inside the autograd graph by RotatedLinear.
    """
    weights = list(weights)
    if not weights:
        raise ValueError("setpoint_repulsion_loss received no weights.")

    device = weights[0].device
    dtype = weights[0].dtype
    centers = _normalize_set_points(set_points, device=device, dtype=dtype)

    total = torch.zeros((), device=device, dtype=dtype)
    for w in weights:
        w_cast = w.to(device=device, dtype=dtype)
        if kind == "gaussian":
            total = total + _gaussian_penalty(w_cast, centers, sigma)
        elif kind == "hinge":
            total = total + _hinge_penalty(w_cast, centers, margin)
        else:
            raise ValueError(f"Unknown repulsion kind: {kind}")
    return total / len(weights)


def simulated_quant_error(
    weights: Iterable[torch.Tensor],
    num_bits: int = 4,
) -> torch.Tensor:
    """Mean squared quantization error under per-tensor symmetric uniform quantization.

    Uses the straight-through estimator so gradients flow back through W_eff to R.
    Minimizing this pushes the rotation to reduce outliers and tighten the scale factor.
    """
    weights = list(weights)
    if not weights:
        raise ValueError("simulated_quant_error received no weights.")

    n_levels = 2 ** (num_bits - 1) - 1
    total = torch.zeros((), device=weights[0].device, dtype=weights[0].dtype)
    for w in weights:
        scale = w.abs().max() / n_levels
        w_q = (w / scale).round() * scale
        w_q_ste = w + (w_q - w).detach()
        total = total + (w - w_q_ste).pow(2).mean()
    return total / len(weights)


def orthogonality_regularizer(rotation: torch.Tensor) -> torch.Tensor:
    """Soft orthogonality prior for ablations that do not use the Stiefel optimizer."""
    identity = torch.eye(rotation.shape[0], device=rotation.device, dtype=rotation.dtype)
    return (rotation.T @ rotation - identity).pow(2).mean()
