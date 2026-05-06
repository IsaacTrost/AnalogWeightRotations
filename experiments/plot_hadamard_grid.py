#!/usr/bin/env python3
"""Plot Hadamard-D grid-search results."""
from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import pandas as pd


def load_summary(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No rows in {input_path}")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["improvement_ratio"] = pd.to_numeric(df["improvement_ratio"], errors="coerce")
    df["identity_ppl"] = pd.to_numeric(df["identity_ppl"], errors="coerce")
    df["rotated_ppl"] = pd.to_numeric(df["rotated_ppl"], errors="coerce")
    df["ir_drop"] = pd.to_numeric(df["ir_drop"], errors="coerce")
    df["input_bits"] = pd.to_numeric(df["input_bits"], errors="coerce")
    df["weight_noise"] = pd.to_numeric(df["weight_noise"], errors="coerce")
    return df


def hadamard_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rotated = df[df["run_kind"] == "hadamard_D"].copy()
    rotated = rotated.dropna(subset=[metric, "ir_drop", "input_bits", "weight_noise"])
    if rotated.empty:
        raise ValueError(f"No Hadamard-D rows with metric {metric!r}")

    group_cols = ["ir_drop", "input_bits", "weight_noise"]
    duplicates = rotated.duplicated(group_cols, keep=False)
    if duplicates.any():
        duplicate_cells = (
            rotated.loc[duplicates, group_cols + ["seed"]]
            .sort_values(group_cols + ["seed"])
            .to_string(index=False)
        )
        raise ValueError(
            "Expected one Hadamard-D seed per hardware cell, but found duplicates:\n"
            f"{duplicate_cells}"
        )

    return rotated.sort_values(["weight_noise", "input_bits", "ir_drop"])


def metric_label(metric: str) -> str:
    if metric == "improvement_ratio":
        return "Identity PPL / Hadamard-D PPL"
    if metric == "rotated_ppl":
        return "Hadamard-D PPL"
    return metric


def output_stem(metric: str, weight_noise: float) -> str:
    weight_noise_text = f"{weight_noise:g}".replace("-", "m").replace(".", "p")
    return f"hadamard_{metric}_wnoise{weight_noise_text}"


def input_bits_order(values: pd.Index) -> list[float]:
    finite_bits = sorted((value for value in values if value != -1), reverse=True)
    return [-1, *finite_bits] if -1 in values else finite_bits


def plot_hadamard_heatmaps(hadamard: pd.DataFrame, output_dir: Path, metric: str) -> None:
    label = metric_label(metric)
    for weight_noise, sub in hadamard.groupby("weight_noise"):
        pivot = sub.pivot_table(
            index="input_bits",
            columns="ir_drop",
            values=metric,
            aggfunc="first",
        )
        pivot = pivot.reindex(input_bits_order(pivot.index))

        plt.figure(figsize=(max(6, 1.0 * len(pivot.columns)), max(4, 0.6 * len(pivot.index))))
        norm = None
        if metric == "improvement_ratio" and pivot.min().min() < 1.0 < pivot.max().max():
            norm = TwoSlopeNorm(vcenter=1.0, vmin=pivot.min().min(), vmax=pivot.max().max())
        image = plt.imshow(pivot.values, aspect="auto", origin="upper", cmap="RdYlGn", norm=norm)
        plt.colorbar(image, label=label)
        plt.xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
        plt.yticks(range(len(pivot.index)), [str(int(v)) for v in pivot.index])
        plt.xlabel("IR drop")
        plt.ylabel("Input bits (-1 = off)")
        plt.title(f"Hadamard-D {label}, weight_noise={weight_noise:g}")

        for row_idx, input_bits in enumerate(pivot.index):
            for col_idx, ir_drop in enumerate(pivot.columns):
                value = pivot.loc[input_bits, ir_drop]
                if pd.notna(value):
                    plt.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center")

        plt.tight_layout()
        out = output_dir / f"{output_stem(metric, weight_noise)}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"saved {out}")


def plot_absolute_ppl_bars(hadamard: pd.DataFrame, output_dir: Path) -> None:
    for weight_noise, sub in hadamard.dropna(subset=["identity_ppl", "rotated_ppl"]).groupby("weight_noise"):
        sub = sub.copy()
        bits_order = input_bits_order(pd.Index(sub["input_bits"].unique()))
        sub["input_bits"] = pd.Categorical(sub["input_bits"], categories=bits_order, ordered=True)
        sub = sub.sort_values(["input_bits", "ir_drop"])

        x = list(range(len(sub)))
        width = 0.38
        identity_x = [value - width / 2 for value in x]
        hadamard_x = [value + width / 2 for value in x]
        labels = [
            f"bits={int(row.input_bits)}\nir={row.ir_drop:g}"
            for row in sub.itertuples(index=False)
        ]

        plt.figure(figsize=(max(10, 0.55 * len(sub)), 5))
        plt.bar(identity_x, sub["identity_ppl"], width=width, label="Identity")
        plt.bar(hadamard_x, sub["rotated_ppl"], width=width, label="Hadamard-D")
        plt.xticks(x, labels, rotation=60, ha="right")
        plt.ylabel("Perplexity")
        plt.xlabel("Hardware cell")
        plt.title(f"Absolute Perplexity by Rotation, weight_noise={weight_noise:g}")
        plt.legend()
        plt.tight_layout()
        out = output_dir / f"absolute_ppl_bars_wnoise{f'{weight_noise:g}'.replace('-', 'm').replace('.', 'p')}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Hadamard-D grid-search summaries.")
    parser.add_argument("--input", default="results/hadamard_grid/summary.csv")
    parser.add_argument("--output-dir", default="results/hadamard_grid/plots")
    parser.add_argument(
        "--metric",
        default="improvement_ratio",
        choices=["improvement_ratio", "rotated_ppl"],
        help="Heatmap value. improvement_ratio > 1 means Hadamard-D beats identity.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(input_path)
    hadamard = hadamard_table(df, args.metric)
    hadamard_csv = output_dir / "hadamard_by_cell.csv"
    hadamard.to_csv(hadamard_csv, index=False)
    print(f"saved {hadamard_csv}")

    plot_hadamard_heatmaps(hadamard, output_dir, args.metric)
    plot_absolute_ppl_bars(hadamard, output_dir)


if __name__ == "__main__":
    main()
