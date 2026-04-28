#!/usr/bin/env python3
"""
stuck_at_faults.py

Tests the effect of orthogonal input rotations on analog error under
stuck-at device faults — where a fraction of crossbar cells are permanently
stuck at their maximum (stuck-high) or minimum (stuck-low/zero) conductance.

This models a real failure mode in PCM, RRAM, and other NVM crossbars where
individual devices stop responding to programming pulses and lock at an extreme
conductance state. Unlike weight noise (which is random and zero-mean),
stuck-at faults introduce a *systematic, persistent* bias in specific weights.

The key question: does rotating the weight matrix before programming reduce
the damage caused by stuck-at faults?

Intuition
---------
With no rotation, a stuck-at fault on a high-magnitude weight has an
outsized effect — that one device was doing a lot of work. After an
incoherent rotation (Hadamard, rand_orth), each original weight is spread
across many physical devices. A single stuck device now only corrupts a
small fraction of every logical weight, and the errors partially cancel.

Fault models
------------
  stuck_high  — faulty device locked at +W_max (contributes max positive signal)
  stuck_low   — faulty device locked at  0     (contributes nothing; open circuit)
  stuck_mixed — equal split of stuck_high and stuck_low faults

Fault rates tested: 0%, 0.5%, 1%, 2%, 5%, 10%

Rotations tested (same as explore_rotations.py)
-----------------------------------------------
  identity, sign_flip, rand_orth, hadamard, hadamard_D, sorted_perm

Base hardware config: w_noise_only (σ=0.02) so there is a realistic noise
floor on top of which stuck-at faults are added.
"""

import os
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from transformers import GPT2Model, GPT2Tokenizer

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.parameters.enums import (
    WeightNoiseType, BoundManagementType, NoiseManagementType
)

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
N_TRIALS   = 20
IN_DIM     = 768
OUT_DIM    = 3072
MAX_BATCH  = 32
FAULT_RATES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]   # fraction of devices
DEVICE     = "cpu"

# ---------------------------------------------------------------------------
# Rotation matrices  (identical to explore_rotations.py)
# ---------------------------------------------------------------------------

def make_identity(n):
    return torch.eye(n)

def make_sign_flip(n, seed=0):
    rng = torch.Generator(); rng.manual_seed(seed)
    d = torch.randint(0, 2, (n,), generator=rng).float() * 2 - 1
    return torch.diag(d)

def make_rand_orth(n, seed=0):
    rng = torch.Generator(); rng.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=rng))
    return Q

def make_block_hadamard(n):
    b = 1
    while b * 2 <= n and n % (b * 2) == 0:
        b *= 2
    k = n // b
    H_b = torch.tensor(hadamard(b), dtype=torch.float32) / (b ** 0.5)
    return torch.block_diag(*([H_b] * k))

def make_hadamard_D(n, seed=0):
    return make_block_hadamard(n) @ make_sign_flip(n, seed=seed)

def make_sorted_perm(n, ref_activations):
    order = torch.argsort(ref_activations.abs().mean(dim=0), descending=True)
    P = torch.zeros(n, n)
    for new_idx, old_idx in enumerate(order):
        P[new_idx, old_idx] = 1.0
    return P

# ---------------------------------------------------------------------------
# Base RPU config: mild weight noise, no IR drop
# (stuck-at faults are injected on top of this)
# ---------------------------------------------------------------------------

def base_rpu_config():
    cfg = InferenceRPUConfig()
    cfg.forward.ir_drop      = 0.0
    cfg.forward.w_noise      = 0.02
    cfg.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
    cfg.forward.inp_noise    = 0.0
    cfg.forward.out_noise    = 0.0
    cfg.forward.inp_res      = -1.0
    cfg.forward.out_res      = -1.0
    cfg.forward.out_bound    = -1.0
    cfg.forward.bound_management = BoundManagementType.NONE
    cfg.forward.noise_management = NoiseManagementType.NONE
    return cfg

# ---------------------------------------------------------------------------
# Load GPT-2 weights and real activations
# ---------------------------------------------------------------------------

def load_gpt2_layer_and_inputs(n_texts=20):
    print("Loading GPT-2 small ...")
    model = GPT2Model.from_pretrained("gpt2")
    tok   = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    mlp = model.h[0].mlp
    W = mlp.c_fc.weight.detach().T.float()   # (3072, 768)
    b = mlp.c_fc.bias.detach().float()

    captured = []
    def hook(module, inp, out):
        captured.append(inp[0].detach().reshape(-1, IN_DIM))
    handle = mlp.c_fc.register_forward_hook(hook)

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models are transforming natural language processing.",
        "In the beginning was the word, and the word was with God.",
        "To be or not to be, that is the question.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael. Some years ago I thought I would sail about a little.",
        "The only way to do great work is to love what you do.",
        "In mathematics you don't understand things, you just get used to them.",
        "Science is not only a disciple of reason but also one of romance.",
        "The universe is stranger than we imagine, and stranger than we can imagine.",
        "Two roads diverged in a yellow wood, and I took the one less traveled by.",
        "Ask not what your country can do for you; ask what you can do for your country.",
        "Elementary, my dear Watson. The game is afoot.",
        "I think therefore I am. Cogito ergo sum.",
        "The medium is the message in the age of electronic communication.",
        "Language is the house of being. In its home man dwells.",
        "The unexamined life is not worth living, said Socrates.",
        "Neurons that fire together wire together during learning.",
        "Attention is all you need for sequence to sequence modelling.",
    ]

    with torch.no_grad():
        for text in texts[:n_texts]:
            tokens = tok(text, return_tensors="pt", truncation=True, max_length=64)
            model(**tokens)

    handle.remove()
    x = torch.cat(captured, dim=0).float()
    print(f"  W shape: {W.shape} | {x.shape[0]} activation samples, "
          f"mean={x.mean():.3f}, std={x.std():.3f}")
    return W, b, x

# ---------------------------------------------------------------------------
# Stuck-at fault injection
# ---------------------------------------------------------------------------

def inject_stuck_at(W: torch.Tensor, fault_rate: float,
                    mode: str = "mixed", seed: int = 0) -> torch.Tensor:
    """
    Return a copy of W with `fault_rate` fraction of elements replaced by
    stuck-at values.

    mode:
      'high'  — all faulty devices stuck at +W_max
      'low'   — all faulty devices stuck at 0  (open-circuit / min conductance)
      'mixed' — 50% stuck_high, 50% stuck_low
    """
    if fault_rate == 0.0:
        return W.clone()

    rng = np.random.RandomState(seed)
    W_out = W.clone()
    W_max = W.abs().max().item()

    flat = W_out.view(-1)
    n_faults = int(round(fault_rate * flat.numel()))
    fault_idx = rng.choice(flat.numel(), size=n_faults, replace=False)

    if mode == "high":
        flat[fault_idx] = W_max
    elif mode == "low":
        flat[fault_idx] = 0.0
    else:  # mixed
        split = n_faults // 2
        flat[fault_idx[:split]]  = W_max
        flat[fault_idx[split:]]  = 0.0

    return W_out

# ---------------------------------------------------------------------------
# Layer creation and evaluation
# ---------------------------------------------------------------------------

def make_analog_layer(W_rot, b, rpu_config):
    out_f, in_f = W_rot.shape
    layer = AnalogLinear(in_f, out_f, bias=True, rpu_config=rpu_config)
    for _, tile in layer.named_analog_layers():
        tile.set_weights(W_rot, b)
    layer.eval()
    return layer


@torch.no_grad()
def eval_analog(layer, x_rot, y_ideal):
    y_a = layer(x_rot)
    err = y_a - y_ideal
    rel = err.norm() / y_ideal.norm()
    snr = 10 * torch.log10(y_ideal.norm()**2 / (err.norm()**2 + 1e-12))
    cos = torch.nn.functional.cosine_similarity(y_a, y_ideal, dim=1).mean()
    return {"rel_error": rel.item(), "snr_db": snr.item(), "cos_sim": cos.item()}


def run_trials(W, b, x, R, fault_rate, fault_mode, n_trials):
    """
    For each trial: apply rotation, inject a fresh set of stuck-at faults
    (different random positions each trial — models unit-to-unit variation),
    load into an analog layer with base noise, evaluate.
    """
    x  = x[:MAX_BATCH]
    x_rot   = x @ R.T
    W_rot   = W @ R.T
    y_ideal = x @ W.T + b.unsqueeze(0)

    metrics = {"rel_error": [], "snr_db": [], "cos_sim": []}

    for trial in range(n_trials):
        W_faulty = inject_stuck_at(W_rot, fault_rate,
                                   mode=fault_mode, seed=trial)
        cfg   = base_rpu_config()
        layer = make_analog_layer(W_faulty, b, cfg)
        m     = eval_analog(layer, x_rot, y_ideal)
        for k in metrics:
            metrics[k].append(m[k])

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

ROT_LABELS = {
    "identity":    "Identity",
    "sign_flip":   "Sign Flip (D)",
    "rand_orth":   "Rand. Orth.",
    "hadamard":    "Block Hadamard",
    "hadamard_D":  "Hadamard+D\n(QuaRot)",
    "sorted_perm": "Sorted Perm.",
}

ROT_COLORS = {
    "identity":    "#abb2bf",
    "sign_flip":   "#e06c75",
    "rand_orth":   "#61afef",
    "hadamard":    "#98c379",
    "hadamard_D":  "#c678dd",
    "sorted_perm": "#e5c07b",
}


def plot_vs_fault_rate(results, rotations, fault_mode, metric="rel_error"):
    """Line plot: x = fault rate, y = metric, one line per rotation."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for rot_name in rotations:
        means = [results[fault_mode][fr][rot_name][metric][0] for fr in FAULT_RATES]
        stds  = [results[fault_mode][fr][rot_name][metric][1] for fr in FAULT_RATES]
        xs    = [fr * 100 for fr in FAULT_RATES]  # percent
        ax.errorbar(xs, means, yerr=stds, label=ROT_LABELS[rot_name],
                    color=ROT_COLORS[rot_name], marker="o", linewidth=1.8,
                    capsize=3, markersize=5)

    ax.set_xlabel("Stuck-at fault rate (%)")
    ax.set_ylabel("Relative L2 error  ||y_a − y_f|| / ||y_f||")
    ax.set_title(f"Stuck-at faults ({fault_mode}) — effect on analog layer error\n"
                 f"(GPT-2 small, layer 0 MLP c_fc, 768→3072)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    fname = f"results/stuck_{fault_mode}_{metric}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_heatmap_at_rate(results, rotations, fault_mode, fault_rate):
    """
    Heatmap of relative improvement vs. identity at a fixed fault rate.
    Rows = fault modes, cols = rotations (excluding identity).
    """
    rot_names    = list(rotations.keys())
    rot_no_id    = [r for r in rot_names if r != "identity"]
    fault_modes  = list(results.keys())

    data = np.zeros((len(fault_modes), len(rot_no_id)))
    for i, fm in enumerate(fault_modes):
        baseline = results[fm][fault_rate]["identity"]["rel_error"][0]
        for j, rot in enumerate(rot_no_id):
            val = results[fm][fault_rate][rot]["rel_error"][0]
            data[i, j] = 100 * (baseline - val) / (baseline + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    vmax = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(rot_no_id)))
    ax.set_xticklabels([ROT_LABELS[r] for r in rot_no_id], fontsize=9)
    ax.set_yticks(range(len(fault_modes)))
    ax.set_yticklabels(fault_modes)
    for i in range(len(fault_modes)):
        for j in range(len(rot_no_id)):
            ax.text(j, i, f"{data[i,j]:+.1f}%", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="black" if abs(data[i, j]) < 30 else "white")
    plt.colorbar(im, ax=ax, label="Relative improvement (%) vs. identity")
    ax.set_title(f"Rotation improvement at {fault_rate*100:.1f}% fault rate\n"
                 "(green = rotation reduces error, red = worsens)")
    fig.tight_layout()
    fname = f"results/stuck_heatmap_{int(fault_rate*100):03d}pct.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_snr_panel(results, rotations):
    """
    3-panel SNR plot (one panel per fault mode), x=fault rate, y=SNR(dB).
    """
    fault_modes = list(results.keys())
    fig, axes = plt.subplots(1, len(fault_modes),
                             figsize=(5 * len(fault_modes), 4.5), sharey=True)

    for ax, fm in zip(axes, fault_modes):
        for rot_name in rotations:
            means = [results[fm][fr][rot_name]["snr_db"][0] for fr in FAULT_RATES]
            xs    = [fr * 100 for fr in FAULT_RATES]
            ax.plot(xs, means, label=ROT_LABELS[rot_name],
                    color=ROT_COLORS[rot_name], marker="o", linewidth=1.8, markersize=4)
        ax.set_title(fm)
        ax.set_xlabel("Fault rate (%)")
        ax.grid(alpha=0.35)

    axes[0].set_ylabel("Output SNR (dB)")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("SNR vs. fault rate by fault mode  (GPT-2 c_fc, 768→3072)",
                 fontsize=11)
    fig.tight_layout()
    fname = "results/stuck_snr_panel.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    W, b, x = load_gpt2_layer_and_inputs()

    print("\nBuilding rotation matrices ...")
    rotations = {
        "identity":    make_identity(IN_DIM),
        "sign_flip":   make_sign_flip(IN_DIM, seed=7),
        "rand_orth":   make_rand_orth(IN_DIM, seed=7),
        "hadamard":    make_block_hadamard(IN_DIM),
        "hadamard_D":  make_hadamard_D(IN_DIM, seed=7),
        "sorted_perm": make_sorted_perm(IN_DIM, x),
    }

    for name, R in rotations.items():
        err = (R @ R.T - torch.eye(IN_DIM)).norm().item()
        print(f"  {name:15s}  R R^T - I = {err:.2e}")

    fault_modes = ["mixed", "high", "low"]

    total = len(fault_modes) * len(FAULT_RATES) * len(rotations)
    print(f"\nRunning {N_TRIALS} trials per combo "
          f"[{len(fault_modes)} modes × {len(FAULT_RATES)} rates × "
          f"{len(rotations)} rotations = {total} combos] ...")

    # results[fault_mode][fault_rate][rot_name] = {metric: (mean, std)}
    results = {}
    for fm in fault_modes:
        results[fm] = {}
        for fr in FAULT_RATES:
            results[fm][fr] = {}
            for rot_name, R in rotations.items():
                print(f"  {fm} | {fr*100:4.1f}% | {rot_name} ...",
                      end=" ", flush=True)
                res = run_trials(W, b, x, R, fr, fm, N_TRIALS)
                results[fm][fr][rot_name] = res
                print(f"rel_err={res['rel_error'][0]:.4f} ± {res['rel_error'][1]:.4f}")

    # --- Summary table at 5% fault rate ---
    fr_summary = 0.05
    print(f"\n{'='*90}")
    print(f"SUMMARY — Relative L2 error at {fr_summary*100:.0f}% fault rate  (mean ± std)")
    print(f"{'='*90}")
    header = f"{'Mode':12s}" + "".join(f" {r:>15s}" for r in rotations)
    print(header)
    print("-" * 90)
    for fm in fault_modes:
        row = f"{fm:12s}"
        for rot_name in rotations:
            m, s = results[fm][fr_summary][rot_name]["rel_error"]
            row += f"  {m:.4f}±{s:.4f}"
        print(row)
    print("=" * 90)

    # --- Plots ---
    for fm in fault_modes:
        plot_vs_fault_rate(results, rotations, fm, metric="rel_error")

    plot_snr_panel(results, rotations)

    # Heatmaps at 1% and 5%
    for fr in [0.01, 0.05]:
        plot_heatmap_at_rate(results, rotations, fault_mode="mixed",
                             fault_rate=fr)

    print("\nDone. Plots saved to ./results/")


if __name__ == "__main__":
    main()
