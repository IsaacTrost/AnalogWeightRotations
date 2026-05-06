#!/usr/bin/env python3
"""
train_weight_matrix.py

Directly optimizes the weight matrix W_opt to minimize analog hardware error,
training on the ACTUAL aihwkit hardware simulation (no surrogate noise model).

Gradient strategy — straight-through via programming noise extraction:
    1. Program W_opt into hardware: layer = make_analog_layer(W_opt.detach(), ...)
    2. Call program_analog_weights() to bake in PCM programming noise
    3. Extract the programmed (noisy) weight: W_noisy = layer.get_weights()
    4. Compute differentiable proxy:  y = x @ (W_opt + noise_const).T + b
       where  noise_const = (W_noisy - W_opt.detach()).detach()  is treated as constant
    5. loss = ||y - y_ideal||_F / ||y_ideal||_F  → backprop to W_opt

The gradient says: move W_opt to pre-compensate for the systematic programming error.
For PCM, this converges to  W_opt ≈ W_original + |programming_undershoot|.
For configs without baked-in noise (w_noise_only, irdrop_only), noise_const ≈ 0
and gradient → 0, correctly reflecting that weight re-training doesn't help there.

y_ideal = W_original @ x.T + b  is the fixed digital baseline throughout.

Change TRAIN_FOR to target a different hardware config.
"""

import gc
import os
import sys
import warnings
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from explore_rotations import (
    load_gpt2_layer_and_inputs,
    make_analog_layer, eval_analog,
    cfg_irdrop_only, cfg_w_noise_only, cfg_inp_quant, cfg_full_pcm,
    IN_DIM, OUT_DIM, N_TRIALS,
)

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_FOR   = "full_pcm"    # hardware config to optimise against; change as needed
TRAIN_STEPS = 150
TRAIN_LR    = 1e-3
N_NOISE     = 3             # independent noise realisations averaged per gradient step
TRAIN_BATCH = 32

CONFIGS = {
    "irdrop_only":  (cfg_irdrop_only,  None),
    "w_noise_only": (cfg_w_noise_only, None),
    "inp_quant":    (cfg_inp_quant,    None),
    "full_pcm":     (cfg_full_pcm,     lambda l: l.program_analog_weights()),
}


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def hardware_step(W_opt, b, x_batch, W_ideal, cfg_fn, post_init, n_noise=N_NOISE):
    """
    Compute loss for one gradient step using the actual aihwkit hardware config.

    For configs with baked-in programming noise (e.g. full_pcm):
        - Programs W_opt into hardware, extracts the noisy weight W_noisy
        - Builds a differentiable proxy:  y = x @ (W_opt + noise_const).T + b
          where noise_const is treated as a constant for this step
        - Gradient flows to W_opt through the proxy

    For configs where noise is per-forward-pass (w_noise_only, irdrop_only):
        - get_weights() returns the original weight (noise_const ≈ 0)
        - The proxy collapses to y ≈ y_ideal → gradient ≈ 0
        - This correctly signals that weight optimisation does not help here

    Returns: (loss_tensor, max_noise_magnitude)
    """
    y_ideal = (x_batch @ W_ideal.T + b.unsqueeze(0)).detach()

    total_loss = torch.zeros(1)
    total_noise_mag = 0.0

    for _ in range(n_noise):
        cfg   = cfg_fn()
        layer = make_analog_layer(W_opt.detach(), b, cfg)
        if post_init is not None:
            post_init(layer)

        # Extract the weight as seen by hardware after programming
        W_noisy, _ = layer.get_weights()   # (out, in), CPU
        W_noisy = W_noisy.detach()

        # Constant noise term for this realisation
        noise_const = (W_noisy - W_opt.detach()).detach()
        total_noise_mag += noise_const.abs().max().item()

        # Differentiable proxy: gradient flows through W_opt
        y_a = x_batch @ (W_opt + noise_const).T + b.unsqueeze(0)
        total_loss = total_loss + (y_a - y_ideal).norm() / (y_ideal.norm() + 1e-12)

    return total_loss / n_noise, total_noise_mag / n_noise


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_weight_matrix(W, b, x):
    cfg_fn, post_init = CONFIGS[TRAIN_FOR]
    x_train  = x[:TRAIN_BATCH]
    y_ideal  = (x_train @ W.T + b.unsqueeze(0)).detach()

    W_opt = W.clone().detach().requires_grad_(True)
    opt   = torch.optim.Adam([W_opt], lr=TRAIN_LR)

    print(f"\nTraining W_opt against '{TRAIN_FOR}' for {TRAIN_STEPS} steps ...")

    for step in range(1, TRAIN_STEPS + 1):
        opt.zero_grad()
        loss, noise_mag = hardware_step(W_opt, b, x_train, W, cfg_fn, post_init)
        loss.backward()
        opt.step()

        if step % 30 == 0 or step == 1:
            with torch.no_grad():
                delta = (W_opt - W).norm().item()
            print(f"  step {step:4d}/{TRAIN_STEPS}  loss={loss.item():.6f}"
                  f"  ||ΔW||={delta:.4f}  max_prog_noise={noise_mag:.4f}")

    W_trained = W_opt.detach()
    delta = (W_trained - W).norm().item()
    print(f"\n  Final ||W_trained - W||_F = {delta:.4f}  "
          f"(relative: {delta / (W.norm().item() + 1e-12):.4f})")

    if noise_mag < 1e-4:
        print(f"\n  NOTE: '{TRAIN_FOR}' has near-zero baked-in programming noise.")
        print(f"  Weight optimisation has no gradient signal for this config.")
        print(f"  Try TRAIN_FOR = 'full_pcm' for a config where this helps.")

    return W_trained


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_trials_W(W_test, b, x, W_ideal, rpu_config_fn, n_trials, post_init=None):
    """Evaluate W_test on hardware; y_ideal is computed from W_ideal."""
    MAX_BATCH = 32
    x       = x[:MAX_BATCH]
    y_ideal = x @ W_ideal.T + b.unsqueeze(0)

    metrics = {"rel_error": [], "snr_db": [], "cos_sim": []}
    for _ in range(n_trials):
        cfg   = rpu_config_fn()
        layer = make_analog_layer(W_test, b, cfg)
        if post_init is not None:
            post_init(layer)
        m = eval_analog(layer, x, y_ideal)
        for k in metrics:
            metrics[k].append(m[k])
        del layer, cfg
        gc.collect()

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    W, b, x = load_gpt2_layer_and_inputs()

    W_trained = train_weight_matrix(W, b, x)

    variants = {"original_W": W, "trained_W": W_trained}

    print(f"\nEvaluating {len(variants)} weight variants × {len(CONFIGS)} "
          f"hw configs ({N_TRIALS} trials each) ...")

    results = {}
    for var_name, W_var in variants.items():
        results[var_name] = {}
        for cfg_name, (cfg_fn, post_init) in CONFIGS.items():
            print(f"  {var_name:12s} × {cfg_name} ...", end=" ", flush=True)
            res = run_trials_W(W_var, b, x, W, cfg_fn, N_TRIALS, post_init=post_init)
            results[var_name][cfg_name] = res
            print(f"rel_err={res['rel_error'][0]:.4f} ± {res['rel_error'][1]:.4f}")

    # --- Summary ---
    col_w = 19
    width = 14 + col_w * len(CONFIGS)
    print("\n" + "=" * width)
    print(f"SUMMARY  —  Relative L2 error  (mean ± std)   [trained on: {TRAIN_FOR}]")
    print("=" * width)
    print(f"{'Variant':13s}" + "".join(f" {c:>{col_w-1}s}" for c in CONFIGS))
    print("-" * width)
    for var_name in variants:
        row = f"{var_name:13s}"
        for cfg_name in CONFIGS:
            m, s = results[var_name][cfg_name]["rel_error"]
            row += f"  {m:.4f}±{s:.4f}  "
        print(row)
    print("=" * width)

    print("\nImprovement of trained_W over original_W (positive = better):")
    for cfg_name in CONFIGS:
        orig    = results["original_W"][cfg_name]["rel_error"][0]
        trained = results["trained_W"][cfg_name]["rel_error"][0]
        pct     = 100.0 * (orig - trained) / (orig + 1e-12)
        marker  = " ← trained on this" if cfg_name == TRAIN_FOR else ""
        print(f"  {cfg_name:15s}: {pct:+.2f}%  ({orig:.4f} → {trained:.4f}){marker}")

    print("\nSNR (dB) — higher is better:")
    print(f"{'Variant':13s}" + "".join(f" {c:>{col_w-1}s}" for c in CONFIGS))
    print("-" * width)
    for var_name in variants:
        row = f"{var_name:13s}"
        for cfg_name in CONFIGS:
            m, s = results[var_name][cfg_name]["snr_db"]
            row += f"  {m:+.2f}±{s:.2f} dB  "
        print(row)


if __name__ == "__main__":
    main()
