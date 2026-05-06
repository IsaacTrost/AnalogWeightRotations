#!/usr/bin/env python3
"""
train_rotation_matrix.py

Learns an orthogonal rotation R that minimizes analog hardware error by training
directly against the ACTUAL aihwkit hardware simulation (no surrogate noise model).

Gradient strategy — SPSA (Simultaneous Perturbation Stochastic Approximation):
    Zeroth-order gradient estimate on the orthogonal group (Stiefel manifold).
    No autograd through the analog layer — avoids memory issues with PCM/IR-drop.

    Each step:
      1. Sample a random skew-symmetric tangent direction A  (||A||_F = 1)
      2. Perturb:  R± = QR(R ± ε·A@R)       (retract back to O(n))
      3. Evaluate hardware loss at R+ and R−  (actual analog forward passes)
      4. Gradient estimate:  ĝ = (L(R+) − L(R−)) / (2ε) · A@R
      5. Adam step with R.grad = ĝ,  then QR retraction

This is unbiased, memory-safe, and works for any hardware config including full_pcm.

Change TRAIN_FOR to target a different hardware config.
Evaluated against: identity, block Hadamard, random orthogonal, and trained R.
"""

import ctypes
import gc
import os
import sys
import warnings
import numpy as np
import torch

try:
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
except OSError:
    def _malloc_trim():
        pass

sys.path.insert(0, os.path.dirname(__file__))
from explore_rotations import (
    load_gpt2_layer_and_inputs,
    make_analog_layer, eval_analog,
    cfg_irdrop_only, cfg_w_noise_only, cfg_inp_quant, cfg_full_pcm,
    make_identity, make_block_hadamard, make_rand_orth,
    IN_DIM, N_TRIALS as _N_TRIALS,
)

N_TRIALS = 10   # override: 10 is plenty for eval; saves memory after heavy training

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_FOR   = "full_pcm"    # hardware config to optimise against; change as needed
TRAIN_STEPS = 200
TRAIN_LR    = 5e-3
SPSA_EPS    = 0.01          # perturbation magnitude on the Stiefel manifold
N_EVAL      = 1             # hardware trials averaged per SPSA function evaluation
TRAIN_BATCH = 32

CONFIGS = {
    "irdrop_only":  (cfg_irdrop_only,  None),
    "w_noise_only": (cfg_w_noise_only, None),
    "inp_quant":    (cfg_inp_quant,    None),
    "full_pcm":     (cfg_full_pcm,     lambda l: l.program_analog_weights()),
}


# ---------------------------------------------------------------------------
# Memory-safe helpers
# ---------------------------------------------------------------------------

def retract(M):
    """Project M back onto the orthogonal group via QR decomposition."""
    Q, _ = torch.linalg.qr(M)
    return Q


@torch.no_grad()
def eval_loss(R_eval, W, b, x_batch, cfg_fn, post_init, n_trials):
    """Average hardware loss for rotation R_eval.  No gradients tracked."""
    x_rot   = x_batch @ R_eval.T
    W_rot   = W @ R_eval.T
    y_ideal = x_batch @ W.T + b.unsqueeze(0)

    total = 0.0
    for _ in range(n_trials):
        cfg   = cfg_fn()
        layer = make_analog_layer(W_rot, b, cfg)
        if post_init is not None:
            post_init(layer)
        y_a    = layer(x_rot)
        total += (y_a - y_ideal).norm().item() / (y_ideal.norm().item() + 1e-12)
        del layer, cfg, y_a
        gc.collect()
        _malloc_trim()

    return total / n_trials


def run_trials(W, b, x, R, rpu_config_fn, n_trials, post_init=None):
    """Memory-safe evaluation (explicit cleanup per trial)."""
    MAX_BATCH = 32
    x_eval  = x[:MAX_BATCH]
    x_rot   = x_eval @ R.T
    W_rot   = W @ R.T
    y_ideal = x_eval @ W.T + b.unsqueeze(0)

    metrics = {"rel_error": [], "snr_db": [], "cos_sim": []}
    for _ in range(n_trials):
        cfg   = rpu_config_fn()
        layer = make_analog_layer(W_rot, b, cfg)
        if post_init is not None:
            post_init(layer)
        m = eval_analog(layer, x_rot, y_ideal)
        for k in metrics:
            metrics[k].append(m[k])
        del layer, cfg
        gc.collect()

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# SPSA training step
# ---------------------------------------------------------------------------

def spsa_step(R, opt, W, b, x_batch, cfg_fn, post_init):
    """
    One SPSA gradient step on the Stiefel manifold.

    Returns (loss_estimate, gradient_norm).
    """
    n = R.shape[0]

    # Random skew-symmetric tangent direction (uniform on the Lie algebra)
    A = torch.randn(n, n)
    A = (A - A.T) / 2
    A = A / (A.norm() + 1e-12)

    # Two perturbed rotations (retracted back onto O(n))
    R_plus  = retract(R.detach() + SPSA_EPS * A @ R.detach())
    R_minus = retract(R.detach() - SPSA_EPS * A @ R.detach())

    # Evaluate hardware loss at both perturbations
    L_plus  = eval_loss(R_plus,  W, b, x_batch, cfg_fn, post_init, N_EVAL)
    L_minus = eval_loss(R_minus, W, b, x_batch, cfg_fn, post_init, N_EVAL)

    # SPSA gradient estimate: projected onto tangent space at R
    grad = (L_plus - L_minus) / (2 * SPSA_EPS) * (A @ R.detach())

    # Adam step (manually set R.grad, then call opt.step)
    opt.zero_grad()
    if R.grad is None:
        R.grad = grad
    else:
        R.grad.copy_(grad)
    opt.step()

    # QR retraction to keep R on the orthogonal group
    with torch.no_grad():
        Q, _ = torch.linalg.qr(R.data)
        R.data.copy_(Q)

    return (L_plus + L_minus) / 2, grad.norm().item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_rotation_matrix(W, b, x, init="identity"):
    cfg_fn, post_init = CONFIGS[TRAIN_FOR]
    x_train = x[:TRAIN_BATCH]

    if init == "identity":
        R = torch.eye(IN_DIM)
    elif init == "hadamard":
        R = make_block_hadamard(IN_DIM)
    elif init == "random":
        R = make_rand_orth(IN_DIM)
    else:
        raise ValueError(f"Unknown init: {init!r}")

    R   = R.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([R], lr=TRAIN_LR)

    print(f"\nTraining rotation R ({init} init) against '{TRAIN_FOR}' "
          f"for {TRAIN_STEPS} steps (SPSA, ε={SPSA_EPS}) ...")

    for step in range(1, TRAIN_STEPS + 1):
        loss_val, grad_norm = spsa_step(R, opt, W, b, x_train, cfg_fn, post_init)
        gc.collect()
        _malloc_trim()

        if step % 40 == 0 or step == 1:
            with torch.no_grad():
                orth_err = (R.data @ R.data.T - torch.eye(IN_DIM)).norm().item()
            print(f"  step {step:4d}/{TRAIN_STEPS}  loss≈{loss_val:.6f}"
                  f"  |grad|={grad_norm:.4f}  orth_err={orth_err:.2e}")

    R_trained = R.detach()
    orth_final = (R_trained @ R_trained.T - torch.eye(IN_DIM)).norm().item()
    print(f"\n  Final orthogonality error ||R R^T - I||_F = {orth_final:.2e}")
    return R_trained


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SAVE_PATH = f"results/trained_R_{TRAIN_FOR}.pt"


def main():
    W, b, x = load_gpt2_layer_and_inputs()
    gc.collect()

    if os.path.exists(SAVE_PATH):
        # Training already done in a previous run; skip directly to evaluation.
        # This avoids the C++ tile memory leak accumulating across 800+ PCM
        # programming calls before the evaluation phase.
        print(f"\nLoading saved rotation from {SAVE_PATH}  (delete to retrain)")
        R_trained = torch.load(SAVE_PATH)
    else:
        R_trained = train_rotation_matrix(W, b, x, init="identity")
        torch.save(R_trained, SAVE_PATH)
        print(f"\n  Saved trained_R to {SAVE_PATH}")
        print("  Re-run this script to evaluate (memory will be clean).")
        return   # exit now so C++ allocations die with the process


    rotations = {
        "identity":  make_identity(IN_DIM),
        "hadamard":  make_block_hadamard(IN_DIM),
        "rand_orth": make_rand_orth(IN_DIM, seed=7),
        "trained_R": R_trained,
    }

    print("\nOrthogonality check  ||R R^T - I||_F:")
    for name, R in rotations.items():
        err = (R @ R.T - torch.eye(IN_DIM)).norm().item()
        print(f"  {name:12s}  {err:.2e}")

    print(f"\nEvaluating {len(rotations)} rotations × {len(CONFIGS)} "
          f"hw configs ({N_TRIALS} trials each) ...")

    results = {}
    for rot_name, R in rotations.items():
        results[rot_name] = {}
        for cfg_name, (cfg_fn, post_init) in CONFIGS.items():
            print(f"  {rot_name:12s} × {cfg_name} ...", end=" ", flush=True)
            res = run_trials(W, b, x, R, cfg_fn, N_TRIALS, post_init=post_init)
            results[rot_name][cfg_name] = res
            print(f"rel_err={res['rel_error'][0]:.4f} ± {res['rel_error'][1]:.4f}")

    # --- Summary ---
    col_w = 19
    width = 14 + col_w * len(CONFIGS)
    print("\n" + "=" * width)
    print(f"SUMMARY  —  Relative L2 error  (mean ± std)   [trained on: {TRAIN_FOR}]")
    print("=" * width)
    print(f"{'Rotation':13s}" + "".join(f" {c:>{col_w-1}s}" for c in CONFIGS))
    print("-" * width)
    for rot_name in rotations:
        row = f"{rot_name:13s}"
        for cfg_name in CONFIGS:
            m, s = results[rot_name][cfg_name]["rel_error"]
            row += f"  {m:.4f}±{s:.4f}  "
        print(row)
    print("=" * width)

    print("\nImprovement vs. identity baseline (positive = better than no rotation):")
    for rot_name in ["hadamard", "rand_orth", "trained_R"]:
        print(f"\n  {rot_name}:")
        for cfg_name in CONFIGS:
            base = results["identity"][cfg_name]["rel_error"][0]
            val  = results[rot_name][cfg_name]["rel_error"][0]
            pct  = 100.0 * (base - val) / (base + 1e-12)
            marker = " ← trained on this" if cfg_name == TRAIN_FOR else ""
            print(f"    {cfg_name:15s}: {pct:+.2f}%  ({base:.4f} → {val:.4f}){marker}")

    print("\nSNR (dB) — higher is better:")
    print(f"{'Rotation':13s}" + "".join(f" {c:>{col_w-1}s}" for c in CONFIGS))
    print("-" * width)
    for rot_name in rotations:
        row = f"{rot_name:13s}"
        for cfg_name in CONFIGS:
            m, s = results[rot_name][cfg_name]["snr_db"]
            row += f"  {m:+.2f}±{s:.2f} dB  "
        print(row)


if __name__ == "__main__":
    main()
