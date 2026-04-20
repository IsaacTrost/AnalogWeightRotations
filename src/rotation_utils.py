import math
from typing import Optional

import torch

from src.rotation_precision import ROTATION_COMPUTE_DTYPE


def is_power_of_two(n: int) -> bool:
    """Return whether `n` is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def largest_power_of_two_divisor(n: int) -> int:
    """Return the largest power-of-two factor that evenly divides `n`."""
    divisor = 1
    while divisor * 2 <= n and n % (divisor * 2) == 0:
        divisor *= 2
    return divisor


def _make_generator(device: Optional[str], seed: Optional[int]) -> Optional[torch.Generator]:
    """Build a per-call RNG so seeded rotations are reproducible."""
    if seed is None:
        return None
    generator = torch.Generator(device=device if device != "mps" else "cpu")
    generator.manual_seed(seed)
    return generator


def identity_matrix(dim: int, device: str = "cpu", dtype: torch.dtype = ROTATION_COMPUTE_DTYPE) -> torch.Tensor:
    """Return the identity rotation for the hidden dimension."""
    return torch.eye(dim, device=device, dtype=dtype)


def sign_flip_matrix(
    dim: int,
    device: str = "cpu",
    dtype: torch.dtype = ROTATION_COMPUTE_DTYPE,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Return a diagonal +/- 1 matrix for lightweight sign scrambling."""
    generator = _make_generator(device, seed)
    signs = torch.randint(0, 2, (dim,), generator=generator, device=device, dtype=torch.int64)
    signed = signs.to(dtype=dtype) * 2 - 1
    return torch.diag(signed)


def random_orthogonal_matrix(
    dim: int,
    device: str = "cpu",
    dtype: torch.dtype = ROTATION_COMPUTE_DTYPE,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate a proper orthogonal matrix from a QR factorization."""
    generator = _make_generator(device, seed)
    base = torch.randn(dim, dim, generator=generator, device=device, dtype=ROTATION_COMPUTE_DTYPE)
    q, r = torch.linalg.qr(base)

    # The diagonal sign correction makes the QR output deterministic up to rotation.
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    q = q * signs.unsqueeze(0)
    return q.to(dtype=dtype)


def hadamard_matrix(dim: int, device: str = "cpu", dtype: torch.dtype = ROTATION_COMPUTE_DTYPE) -> torch.Tensor:
    """Generate a normalized Hadamard matrix for power-of-two dimensions."""
    if not is_power_of_two(dim):
        raise ValueError(f"Hadamard rotation requires a power-of-two dimension, got {dim}.")

    matrix = torch.tensor([[1.0]], device=device, dtype=dtype)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix / math.sqrt(dim)


def block_hadamard_matrix(
    dim: int,
    device: str = "cpu",
    dtype: torch.dtype = ROTATION_COMPUTE_DTYPE,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Build a block-diagonal Hadamard for dimensions like 768 or 5632."""
    if block_size is None:
        block_size = largest_power_of_two_divisor(dim)
    if not is_power_of_two(block_size) or dim % block_size != 0:
        raise ValueError(f"Invalid Hadamard block size {block_size} for dimension {dim}.")

    blocks = dim // block_size
    had_block = hadamard_matrix(block_size, device=device, dtype=dtype)
    return torch.block_diag(*([had_block] * blocks))


def signed_hadamard_matrix(
    dim: int,
    device: str = "cpu",
    dtype: torch.dtype = ROTATION_COMPUTE_DTYPE,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Apply a random sign flip after a full Hadamard transform."""
    return hadamard_matrix(dim, device=device, dtype=dtype) @ sign_flip_matrix(
        dim,
        device=device,
        dtype=dtype,
        seed=seed,
    )


def signed_block_hadamard_matrix(
    dim: int,
    device: str = "cpu",
    dtype: torch.dtype = ROTATION_COMPUTE_DTYPE,
    seed: Optional[int] = None,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Apply sign randomization after the block-Hadamard used for non-power-of-two widths."""
    return block_hadamard_matrix(
        dim,
        device=device,
        dtype=dtype,
        block_size=block_size,
    ) @ sign_flip_matrix(dim, device=device, dtype=dtype, seed=seed)


def get_rotation_matrix(
    dim: int,
    mode: str,
    device: str = "cpu",
    dtype: torch.dtype = ROTATION_COMPUTE_DTYPE,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Dispatch to the configured rotation builder."""
    if mode == "identity":
        return identity_matrix(dim, device=device, dtype=dtype)
    if mode == "sign_flip":
        return sign_flip_matrix(dim, device=device, dtype=dtype, seed=seed)
    if mode == "random":
        return random_orthogonal_matrix(dim, device=device, dtype=dtype, seed=seed)
    if mode == "hadamard":
        return hadamard_matrix(dim, device=device, dtype=dtype)
    if mode == "block_hadamard":
        return block_hadamard_matrix(dim, device=device, dtype=dtype)
    if mode == "hadamard_D":
        if is_power_of_two(dim):
            return signed_hadamard_matrix(dim, device=device, dtype=dtype, seed=seed)
        return signed_block_hadamard_matrix(dim, device=device, dtype=dtype, seed=seed)
    raise ValueError(f"Unknown rotation mode: {mode}")


def orthogonality_error(rotation: torch.Tensor) -> float:
    """Measure how close a rotation is to an orthonormal matrix."""
    identity = torch.eye(rotation.shape[0], device=rotation.device, dtype=rotation.dtype)
    return torch.linalg.norm(rotation @ rotation.T - identity).item()


def is_orthonormal(rotation: torch.Tensor, tol: float = 1e-5) -> bool:
    """Check `R^T R = I` numerically."""
    identity = torch.eye(rotation.shape[0], device=rotation.device, dtype=rotation.dtype)
    return torch.allclose(rotation.T @ rotation, identity, atol=tol)


def apply_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply a right-multiplied basis change to the last tensor dimension."""
    return x @ rotation


def inverse_rotation(rotation: torch.Tensor) -> torch.Tensor:
    """Return the inverse of an orthogonal rotation."""
    return rotation.T