#!/usr/bin/env python3
"""
plot_combined.py

Cross-model combined plots for the rotation experiment.

Produces three figures saved to results/combined/:
  1. full_pcm_comparison.png  — grouped bar chart, all layers × key rotations, full_pcm
  2. improvement_heatmap.png  — % improvement vs identity, all layers × rotations,
                                one panel per hardware config
  3. config_summary.png       — identity vs best rotation for every layer × config
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

LAYERS = [
    ("GPT-2s h[0]\nc_fc 768→3072",     "results/gpt2s_h0_mlp/results.json"),
    ("GPT-2s h[11]\nc_fc 768→3072",    "results/gpt2s_h11_mlp/results.json"),
    ("SmolLM2-135M\ngate 576→1536",    "results/smollm2_135m_l0_gate/results.json"),
    ("Llama-3.2-1B\no_proj 2048→2048", "results/llama32_1b_l0_oproj/results.json"),
    ("Qwen2-1.5B\nq_proj 1536→1536",   "results/qwen2_15b_l0_qproj/results.json"),
]

CONFIGS  = ["irdrop_only", "w_noise_only", "inp_quant", "full_pcm"]
CFG_LABELS = {
    "irdrop_only":  "IR Drop",
    "w_noise_only": "Weight Noise",
    "inp_quant":    "8-bit Quant",
    "full_pcm":     "Full PCM",
}
ROTATIONS = ["identity", "sign_flip", "rand_orth", "hadamard", "hadamard_D", "sorted_perm"]
ROT_LABELS = {
    "identity":    "Identity",
    "sign_flip":   "Sign Flip",
    "rand_orth":   "Rand Orth",
    "hadamard":    "Hadamard",
    "hadamard_D":  "Hadamard·D",
    "sorted_perm": "Sorted Perm",
}

# Load all results
data = {}
for label, path in LAYERS:
    with open(path) as f:
        data[label] = json.load(f)

layer_labels = [l for l, _ in LAYERS]

os.makedirs("results/combined", exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1: Full PCM grouped bar chart
# Rows = layers, groups of bars = rotations
# ---------------------------------------------------------------------------

ROT_COLORS = {
    "identity":    "#abb2bf",
    "sign_flip":   "#e06c75",
    "rand_orth":   "#61afef",
    "hadamard":    "#98c379",
    "hadamard_D":  "#c678dd",
    "sorted_perm": "#e5c07b",
}

fig, ax = plt.subplots(figsize=(13, 5))

n_layers = len(layer_labels)
n_rots   = len(ROTATIONS)
group_w  = 0.8
bar_w    = group_w / n_rots
x_base   = np.arange(n_layers)

for i, rot in enumerate(ROTATIONS):
    vals = [data[l]["full_pcm"][rot]["rel_error"][0] * 100 for l in layer_labels]
    errs = [data[l]["full_pcm"][rot]["rel_error"][1] * 100 for l in layer_labels]
    offset = (i - n_rots / 2 + 0.5) * bar_w
    ax.bar(x_base + offset, vals, bar_w * 0.9,
           yerr=errs, capsize=2,
           color=ROT_COLORS[rot], label=ROT_LABELS[rot], alpha=0.9)

ax.set_xticks(x_base)
ax.set_xticklabels(layer_labels, fontsize=9)
ax.set_ylabel("Relative L2 Error (%)", fontsize=11)
ax.set_title("Full PCM Config — Relative Error by Layer and Rotation", fontsize=12)
ax.legend(ncol=6, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.0))
ax.set_ylim(bottom=0)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("results/combined/full_pcm_comparison.png", dpi=130)
plt.close(fig)
print("Saved results/combined/full_pcm_comparison.png")

# ---------------------------------------------------------------------------
# Figure 2: Improvement heatmap (% reduction vs identity)
# 4 panels (one per config), rows = layers, cols = non-identity rotations
# ---------------------------------------------------------------------------

NON_ID_ROTS = ["sign_flip", "rand_orth", "hadamard", "hadamard_D", "sorted_perm"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)

for ax, cfg in zip(axes, CONFIGS):
    matrix = np.zeros((n_layers, len(NON_ID_ROTS)))
    for i, lbl in enumerate(layer_labels):
        base = data[lbl][cfg]["identity"]["rel_error"][0]
        for j, rot in enumerate(NON_ID_ROTS):
            val  = data[lbl][cfg][rot]["rel_error"][0]
            matrix[i, j] = 100.0 * (base - val) / (base + 1e-12)  # + = improvement

    vmax = max(abs(matrix).max(), 1.0)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(NON_ID_ROTS)))
    ax.set_xticklabels([ROT_LABELS[r] for r in NON_ID_ROTS],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_labels, fontsize=8)
    ax.set_title(CFG_LABELS[cfg], fontsize=10, fontweight="bold")

    # Annotate cells
    for i in range(n_layers):
        for j in range(len(NON_ID_ROTS)):
            v = matrix[i, j]
            color = "white" if abs(v) > vmax * 0.6 else "black"
            ax.text(j, i, f"{v:+.0f}%", ha="center", va="center",
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% improvement vs identity")

fig.suptitle("Improvement vs Identity Baseline  (green = rotation helps, red = hurts)",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig("results/combined/improvement_heatmap.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("Saved results/combined/improvement_heatmap.png")

# ---------------------------------------------------------------------------
# Figure 3: Identity vs best-rotation for every layer × config
# Line plot: x = layer, two lines per config (identity and best)
# ---------------------------------------------------------------------------

BEST_ROT = "hadamard_D"   # consistently best or tied-best across GPT-2/SmolLM2/Qwen2

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
axes = axes.flatten()

CFG_COLORS = {
    "irdrop_only":  "#e06c75",
    "w_noise_only": "#61afef",
    "inp_quant":    "#98c379",
    "full_pcm":     "#c678dd",
}

x = np.arange(n_layers)

for ax, cfg in zip(axes, CONFIGS):
    id_vals  = [data[l][cfg]["identity"]["rel_error"][0] * 100  for l in layer_labels]
    id_errs  = [data[l][cfg]["identity"]["rel_error"][1] * 100  for l in layer_labels]
    rot_vals = [data[l][cfg][BEST_ROT]["rel_error"][0]  * 100  for l in layer_labels]
    rot_errs = [data[l][cfg][BEST_ROT]["rel_error"][1]  * 100  for l in layer_labels]

    c = CFG_COLORS[cfg]
    ax.errorbar(x, id_vals,  yerr=id_errs,  fmt="o--", color=c,
                alpha=0.5, linewidth=1.5, capsize=3, label="Identity")
    ax.errorbar(x, rot_vals, yerr=rot_errs, fmt="o-",  color=c,
                alpha=1.0, linewidth=2.0, capsize=3, label=f"{ROT_LABELS[BEST_ROT]}")
    ax.fill_between(x, id_vals, rot_vals, alpha=0.12, color=c)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=8)
    ax.set_ylabel("Relative L2 Error (%)", fontsize=9)
    ax.set_title(CFG_LABELS[cfg], fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

fig.suptitle(f"Identity vs {ROT_LABELS[BEST_ROT]} — All Layers × All Configs\n"
             f"(shaded area = rotation benefit)",
             fontsize=12)
fig.tight_layout()
fig.savefig("results/combined/identity_vs_best.png", dpi=130)
plt.close(fig)
print("Saved results/combined/identity_vs_best.png")

print("\nDone. All plots in results/combined/")
