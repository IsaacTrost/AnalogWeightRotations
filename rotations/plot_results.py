#!/usr/bin/env python3
"""
plot_results.py — generate plots from the hardcoded results of the completed run.
Run this instead of the full explore_rotations.py to avoid re-running experiments.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# Results from the completed run (mean, std) for rel_error
# Structure: results[config][rotation] = {metric: (mean, std)}
results = {
    "irdrop_only": {
        "identity":    {"rel_error": (0.0685, 0.0000), "snr_db": (23.29, 0.0),  "cos_sim": (0.9995, 0.0)},
        "sign_flip":   {"rel_error": (0.0685, 0.0000), "snr_db": (23.29, 0.0),  "cos_sim": (0.9995, 0.0)},
        "rand_orth":   {"rel_error": (0.0013, 0.0000), "snr_db": (57.54, 0.0),  "cos_sim": (1.0000, 0.0)},
        "hadamard":    {"rel_error": (0.0013, 0.0000), "snr_db": (57.54, 0.0),  "cos_sim": (1.0000, 0.0)},
        "hadamard_D":  {"rel_error": (0.0013, 0.0000), "snr_db": (57.54, 0.0),  "cos_sim": (1.0000, 0.0)},
        "sorted_perm": {"rel_error": (0.0682, 0.0000), "snr_db": (23.33, 0.0),  "cos_sim": (0.9995, 0.0)},
    },
    "w_noise_only": {
        "identity":    {"rel_error": (0.0907, 0.0002), "snr_db": (20.84, 0.02), "cos_sim": (0.9992, 0.0)},
        "sign_flip":   {"rel_error": (0.0907, 0.0002), "snr_db": (20.84, 0.02), "cos_sim": (0.9992, 0.0)},
        "rand_orth":   {"rel_error": (0.0624, 0.0001), "snr_db": (24.09, 0.02), "cos_sim": (0.9996, 0.0)},
        "hadamard":    {"rel_error": (0.0624, 0.0002), "snr_db": (24.09, 0.02), "cos_sim": (0.9996, 0.0)},
        "hadamard_D":  {"rel_error": (0.0624, 0.0001), "snr_db": (24.09, 0.02), "cos_sim": (0.9996, 0.0)},
        "sorted_perm": {"rel_error": (0.0905, 0.0002), "snr_db": (20.86, 0.02), "cos_sim": (0.9992, 0.0)},
    },
    "inp_quant": {
        "identity":    {"rel_error": (0.0734, 0.0000), "snr_db": (22.68, 0.0),  "cos_sim": (0.9997, 0.0)},
        "sign_flip":   {"rel_error": (0.0734, 0.0000), "snr_db": (22.68, 0.0),  "cos_sim": (0.9997, 0.0)},
        "rand_orth":   {"rel_error": (0.0282, 0.0000), "snr_db": (30.99, 0.0),  "cos_sim": (0.9999, 0.0)},
        "hadamard":    {"rel_error": (0.0283, 0.0000), "snr_db": (30.97, 0.0),  "cos_sim": (0.9999, 0.0)},
        "hadamard_D":  {"rel_error": (0.0283, 0.0000), "snr_db": (30.97, 0.0),  "cos_sim": (0.9999, 0.0)},
        "sorted_perm": {"rel_error": (0.0735, 0.0000), "snr_db": (22.67, 0.0),  "cos_sim": (0.9997, 0.0)},
    },
    "full_pcm": {
        "identity":    {"rel_error": (0.1483, 0.0010), "snr_db": (16.57, 0.06), "cos_sim": (0.9989, 0.0)},
        "sign_flip":   {"rel_error": (0.1484, 0.0011), "snr_db": (16.57, 0.06), "cos_sim": (0.9989, 0.0)},
        "rand_orth":   {"rel_error": (0.0629, 0.0004), "snr_db": (24.02, 0.06), "cos_sim": (0.9998, 0.0)},
        "hadamard":    {"rel_error": (0.0640, 0.0004), "snr_db": (23.87, 0.06), "cos_sim": (0.9998, 0.0)},
        "hadamard_D":  {"rel_error": (0.0610, 0.0004), "snr_db": (24.29, 0.06), "cos_sim": (0.9998, 0.0)},
        "sorted_perm": {"rel_error": (0.1599, 0.0012), "snr_db": (15.92, 0.07), "cos_sim": (0.9987, 0.0)},
    },
}

rotations = ["identity", "sign_flip", "rand_orth", "hadamard", "hadamard_D", "sorted_perm"]
cfg_names  = ["irdrop_only", "w_noise_only", "inp_quant", "full_pcm"]

ROT_LABELS = {
    "identity":    "Identity\n(baseline)",
    "sign_flip":   "Sign Flip\n(D)",
    "rand_orth":   "Rand. Orth.\n(QR)",
    "hadamard":    "Block\nHadamard",
    "hadamard_D":  "Hadamard+D\n(QuaRot)",
    "sorted_perm": "Sorted\nPerm.",
}

CFG_COLORS = {
    "irdrop_only":  "#e06c75",
    "w_noise_only": "#61afef",
    "inp_quant":    "#98c379",
    "full_pcm":     "#c678dd",
}

CFG_LABELS = {
    "irdrop_only":  "IR Drop only",
    "w_noise_only": "Weight noise only",
    "inp_quant":    "8-bit quant. only",
    "full_pcm":     "Full PCM (realistic)",
}


# ---------------------------------------------------------------------------
# 1. Grouped bar chart — relative error
# ---------------------------------------------------------------------------
def plot_bar(metric, ylabel, title, fname, higher_is_better=False):
    n_cfg = len(cfg_names)
    n_rot = len(rotations)
    x_pos = np.arange(n_rot)
    width = 0.8 / n_cfg

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, cfg in enumerate(cfg_names):
        means = [results[cfg][r][metric][0] for r in rotations]
        stds  = [results[cfg][r][metric][1] for r in rotations]
        offset = (i - n_cfg / 2 + 0.5) * width
        ax.bar(x_pos + offset, means, width * 0.9, yerr=stds,
               label=CFG_LABELS[cfg], color=CFG_COLORS[cfg],
               error_kw=dict(elinewidth=1.2, capsize=3), alpha=0.9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ROT_LABELS[r] for r in rotations], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# 2. Improvement heatmap vs identity baseline
# ---------------------------------------------------------------------------
def plot_heatmap(fname):
    rot_no_id = [r for r in rotations if r != "identity"]
    data = np.zeros((len(cfg_names), len(rot_no_id)))
    for i, cfg in enumerate(cfg_names):
        base = results[cfg]["identity"]["rel_error"][0]
        for j, rot in enumerate(rot_no_id):
            val = results[cfg][rot]["rel_error"][0]
            data[i, j] = 100 * (base - val) / base

    fig, ax = plt.subplots(figsize=(9, 4))
    vmax = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(rot_no_id)))
    ax.set_xticklabels([ROT_LABELS[r] for r in rot_no_id], fontsize=9)
    ax.set_yticks(range(len(cfg_names)))
    ax.set_yticklabels([CFG_LABELS[c] for c in cfg_names], fontsize=9)
    for i in range(len(cfg_names)):
        for j in range(len(rot_no_id)):
            ax.text(j, i, f"{data[i,j]:+.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="black" if abs(data[i, j]) < 50 else "white")
    plt.colorbar(im, ax=ax, label="Improvement over identity baseline (%)")
    ax.set_title("Rotation improvement over no-rotation baseline\n"
                 "green = rotation reduces analog error, red = worsens", fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# 3. SNR panel — one subplot per config
# ---------------------------------------------------------------------------
def plot_snr_panel(fname):
    fig, axes = plt.subplots(1, len(cfg_names), figsize=(14, 4), sharey=False)
    for ax, cfg in zip(axes, cfg_names):
        means = [results[cfg][r]["snr_db"][0] for r in rotations]
        colors = [CFG_COLORS[cfg] if r == "identity" else
                  "#aaaaaa" if r in ("sign_flip", "sorted_perm") else
                  "#2ecc71"
                  for r in rotations]
        bars = ax.bar(range(len(rotations)), means, color=colors, alpha=0.9, edgecolor="white")
        ax.set_xticks(range(len(rotations)))
        ax.set_xticklabels([ROT_LABELS[r] for r in rotations], fontsize=7, rotation=15, ha="right")
        ax.set_title(CFG_LABELS[cfg], fontsize=9)
        ax.set_ylabel("SNR (dB)" if cfg == cfg_names[0] else "")
        ax.grid(axis="y", alpha=0.3)
        # Annotate values
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5)
    fig.suptitle("Output SNR (dB) by rotation — higher is better", fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# 4. Summary table as figure
# ---------------------------------------------------------------------------
def plot_table(fname):
    col_labels = [ROT_LABELS[r].replace("\n", " ") for r in rotations]
    row_labels  = [CFG_LABELS[c] for c in cfg_names]
    cell_text   = []
    cell_colors = []

    for cfg in cfg_names:
        row_vals = [results[cfg][r]["rel_error"][0] for r in rotations]
        best = min(row_vals)
        worst = max(row_vals)
        row_str, row_col = [], []
        for v in row_vals:
            row_str.append(f"{v*100:.2f}%")
            if v == best:
                row_col.append("#b5e8b0")
            elif v == worst:
                row_col.append("#f5b5b5")
            else:
                row_col.append("#f8f8f8")
        cell_text.append(row_str)
        cell_colors.append(row_col)

    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels,
                   colLabels=col_labels, cellColours=cell_colors,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax.set_title("Relative L2 error  ||y_analog − y_ideal|| / ||y_ideal||   "
                 "(green = best per row, red = worst)", fontsize=10, pad=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


if __name__ == "__main__":
    print("Generating plots ...")
    plot_bar("rel_error",
             "Relative L2 error  ||y_a − y_f|| / ||y_f||",
             "Effect of input rotation on analog layer output fidelity\n"
             "(GPT-2 small, layer 0 MLP c_fc, 768→3072)",
             "results/rel_error.png")
    plot_bar("snr_db",
             "Output SNR (dB)",
             "Output SNR by rotation and analog config",
             "results/snr_db.png", higher_is_better=True)
    plot_heatmap("results/improvement_heatmap.png")
    plot_snr_panel("results/snr_panel.png")
    plot_table("results/summary_table.png")
    print("Done. All plots in ./results/")
