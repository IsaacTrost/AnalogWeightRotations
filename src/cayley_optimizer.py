import random

import torch
from torch.optim import Optimizer


def unit(matrix: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a 2D matrix along one dimension for Stiefel-style updates."""
    matrix_norm = matrix.norm(p=2, dim=dim, keepdim=True)
    return matrix / matrix_norm.add(eps), matrix_norm


def matrix_norm_one(matrix: torch.Tensor) -> torch.Tensor:
    """Measure the matrix one-norm used to cap the Cayley step size."""
    return torch.abs(matrix).sum(dim=0).max()


def cayley_loop(
    basis: torch.Tensor,
    skew_symmetric: torch.Tensor,
    tangent_vector: torch.Tensor,
    step_size: float,
    iterations: int = 5,
) -> torch.Tensor:
    """Approximate the Cayley transform with a small fixed-point loop."""
    updated = basis + step_size * tangent_vector
    for _ in range(iterations):
        updated = basis + step_size * torch.matmul(skew_symmetric, 0.5 * (basis + updated))
    return updated.t()


def qr_retraction(matrix: torch.Tensor) -> torch.Tensor:
    """Project a 2D matrix back onto the orthogonal manifold with a stable QR sign fix."""
    transposed = matrix.t()
    q_factor, r_factor = torch.linalg.qr(transposed)
    diagonal = torch.diag(r_factor, 0)
    phase = diagonal.sign()
    phase[phase == 0] = 1
    q_factor = q_factor * phase.unsqueeze(0)
    return q_factor.t()


class CayleySGDG(Optimizer):
    """Update rotation matrices with the SpinQuant-style Cayley/Stiefel rule."""

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        stiefel: bool = True,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        grad_clip: float | None = None,
        qr_retraction_frequency: int = 1,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}.")
        if qr_retraction_frequency <= 0:
            raise ValueError("QR retraction frequency must be positive.")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires positive momentum and zero dampening.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            stiefel=stiefel,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            grad_clip=grad_clip,
            qr_retraction_frequency=qr_retraction_frequency,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Advance each rotation matrix with the direct manifold update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                reshaped = parameter.data.view(parameter.size(0), -1)
                normalized, _ = unit(reshaped)
                gradient = parameter.grad.data.view(parameter.size(0), -1)

                if group["grad_clip"] is not None:
                    gradient = gradient.clamp(min=-group["grad_clip"], max=group["grad_clip"])

                if group["stiefel"] and normalized.size(0) <= normalized.size(1):
                    state = self.state[parameter]
                    if "step" not in state:
                        state["step"] = 0
                    state["step"] += 1

                    if state["step"] % group["qr_retraction_frequency"] == 0:
                        normalized = qr_retraction(normalized)

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(gradient.t())

                    momentum_buffer = state["momentum_buffer"]
                    momentum_buffer.mul_(group["momentum"]).add_(-gradient.t())
                    mixed = torch.mm(momentum_buffer, normalized)
                    projected = torch.mm(normalized, mixed)
                    gram = torch.mm(normalized.t(), projected)
                    skew_candidate = mixed - 0.5 * gram
                    skew_symmetric = skew_candidate - skew_candidate.t()
                    stability_bound = torch.tensor(
                        1.0,
                        device=skew_symmetric.device,
                        dtype=skew_symmetric.dtype,
                    )
                    step_cap = stability_bound / (matrix_norm_one(skew_symmetric) + 1e-8)
                    step_size = min(step_cap.item(), group["lr"])

                    updated = cayley_loop(normalized.t(), skew_symmetric, momentum_buffer, step_size)
                    if state["step"] % group["qr_retraction_frequency"] == 0:
                        updated = qr_retraction(updated.view(parameter.size(0), -1)).view_as(parameter.data)
                    momentum_buffer.copy_(torch.mm(skew_symmetric, normalized.t()))
                    parameter.data.copy_(updated.view_as(parameter.data))
                    continue

                update = gradient
                if group["weight_decay"] != 0:
                    update = update.add(parameter.data, alpha=group["weight_decay"])

                if group["momentum"] != 0:
                    state = self.state[parameter]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = update.clone()
                    else:
                        state["momentum_buffer"].mul_(group["momentum"]).add_(
                            update,
                            alpha=1 - group["dampening"],
                        )
                    if group["nesterov"]:
                        update = update.add(state["momentum_buffer"], alpha=group["momentum"])
                    else:
                        update = state["momentum_buffer"]

                parameter.data.add_(update, alpha=-group["lr"])

        return loss
