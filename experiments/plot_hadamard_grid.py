#!/usr/bin/env python3
"""Plot Hadamard-D grid-search results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No rows in {input_path}")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["improvement_ratio"] = pd.to_numeric(df["improvement_ratio"], errors="coerce")
    return df


def best_seed_table(df: pd.DataFrame) -> pd.DataFrame:
    rotated = df[df["run_kind"] == "hadamard_D"].copy()
    rotated = rotated.dropna(subset=["improvement_ratio"])
    idx = rotated.groupby(["ir_drop", "input_bits", "weight_noise"])["improvement_ratio"].idxmax()
    return rotated.loc[idx].sort_values(["weight_noise", "input_bits", "ir_drop"])


def plot_best_improvement_heatmaps(best: pd.DataFrame, output_dir: Path) -> None:
    for weight_noise, sub in best.groupby("weight_noise"):
        pivot = sub.pivot_table(
            index="input_bits",
            columns="ir_drop",
            values="improvement_ratio",
            aggfunc="first",
        ).sort_index(ascending=False)

        plt.figure(figsize=(max(6, 1.0 * len(pivot.columns)), max(4, 0.6 * len(pivot.index))))
        image = plt.imshow(pivot.values, aspect="auto", origin="upper")
        plt.colorbar(image, label="Identity PPL / best Hadamard-D PPL")
        plt.xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
        plt.yticks(range(len(pivot.index)), [str(int(v)) for v in pivot.index])
        plt.xlabel("IR drop")
        plt.ylabel("Input bits (-1 = off)")
        plt.title(f"Best Hadamard-D Improvement, weight_noise={weight_noise:g}")

        for row_idx, input_bits in enumerate(pivot.index):
            for col_idx, ir_drop in enumerate(pivot.columns):
                value = pivot.loc[input_bits, ir_drop]
                if pd.notna(value):
                    seed = sub[
                        (sub["input_bits"] == input_bits)
                        & (sub["ir_drop"] == ir_drop)
                    ]["seed"].iloc[0]
                    plt.text(col_idx, row_idx, f"{value:.2f}\ns={int(seed)}", ha="center", va="center")

        plt.tight_layout()
        out = output_dir / f"best_improvement_wnoise{str(weight_noise).replace('.', 'p')}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"saved {out}")


def plot_seed_spread(df: pd.DataFrame, output_dir: Path) -> None:
    rotated = df[df["run_kind"] == "hadamard_D"].dropna(subset=["improvement_ratio"]).copy()
    grouped = rotated.groupby(["ir_drop", "input_bits", "weight_noise"])["improvement_ratio"]
    spread = grouped.agg(["min", "mean", "max", "std"]).reset_index()
    spread["label"] = spread.apply(
        lambda row: f"ir={row.ir_drop:g}, bits={int(row.input_bits)}, wn={row.weight_noise:g}",
        axis=1,
    )
    spread = spread.sort_values("max", ascending=False)

    plt.figure(figsize=(max(10, 0.45 * len(spread)), 5))
    x = range(len(spread))
    plt.errorbar(
        x,
        spread["mean"],
        yerr=[
            spread["mean"] - spread["min"],
            spread["max"] - spread["mean"],
        ],
        fmt="o",
        capsize=3,
    )
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xticks(list(x), spread["label"], rotation=65, ha="right")
    plt.ylabel("Identity PPL / Hadamard-D PPL")
    plt.title("Hadamard-D Seed Spread by Hardware Cell")
    plt.tight_layout()
    out = output_dir / "seed_spread.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Hadamard-D grid-search summaries.")
    parser.add_argument("--input", default="results/hadamard_grid/summary.csv")
    parser.add_argument("--output-dir", default="results/hadamard_grid/plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(input_path)
    best = best_seed_table(df)
    best_csv = output_dir / "best_seed_by_cell.csv"
    best.to_csv(best_csv, index=False)
    print(f"saved {best_csv}")

    plot_best_improvement_heatmaps(best, output_dir)
    plot_seed_spread(df, output_dir)


if __name__ == "__main__":
    main()
