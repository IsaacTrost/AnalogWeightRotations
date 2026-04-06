import torch
import math


def random_orthogonal_matrix(dim, device="cpu"):
    """
    Generate a random orthonormal matrix using QR decomposition.
    """
    A = torch.randn(dim, dim, device=device)
    Q, R = torch.linalg.qr(A)

    # ensure determinant = 1 (proper rotation)
    d = torch.diag(R)
    ph = d.sign()
    Q *= ph

    return Q


def hadamard_matrix(n, device="cpu"):
    """
    Generate Hadamard matrix (n must be power of 2).
    """
    assert (n & (n - 1) == 0), "n must be power of 2"

    H = torch.tensor([[1.0]], device=device)

    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)

    return H / math.sqrt(n)


def signed_hadamard_matrix(n, device="cpu"):
    """
    Random sign-flipped Hadamard.
    """
    H = hadamard_matrix(n, device=device)
    signs = torch.randint(0, 2, (n,), device=device) * 2 - 1
    D = torch.diag(signs.float())
    return D @ H


def is_orthonormal(R, tol=1e-5):
    """
    Check R^T R = I
    """
    I = torch.eye(R.shape[0], device=R.device)
    return torch.allclose(R.T @ R, I, atol=tol)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def apply_rotation(x, R):
    """
    x: (..., dim)
    R: (dim, dim)
    """
    return x @ R


def inverse_rotation(R):
    return R.T  # orthonormal