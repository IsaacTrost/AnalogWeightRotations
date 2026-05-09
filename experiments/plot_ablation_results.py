#!/usr/bin/env python3
"""
Plot ablation results from eval_analog_perplexity JSON outputs.

Expected filename like:
  results/ablation/smollm2_135m_quant_only_downproj_identity_2048.json

Usage:
  conda run -n aihwkit python scripts/plot_ablation_results.py \
    --input-dir results/ablation \
    --output-dir results/ablation_plots
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FILENAME_RE = re.compile(
    r"smollm2_135m_(?P<hardware>.+?)_(?P<target>downproj|mlp|all_no_lm_head|all)_(?P<mode>identity|hadamardD|learned)_(?P<tokens>\d+)\.json$"
)


def parse_file(path: Path) -> dict:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    groups = match.groupdict()
    mode = groups["mode"]

    # For each JSON, the relevant result is usually:
    # identity file  -> analog_identity
    # hadamardD file -> analog_rotated
    # learned file   -> analog_rotated
    if mode == "identity":
        run_key = "analog_identity"
    else:
        run_key = "analog_rotated"

    runs = data.get("runs", {})
    if run_key not in runs:
        raise KeyError(f"{path.name} missing expected run '{run_key}'")

    float_ppl = None
    if "float_prepared" in runs:
        float_ppl = runs["float_prepared"]["ppl"]

    return {
        "file": str(path),
        "hardware": groups["hardware"],
        "target": groups["target"],
        "mode": mode,
        "tokens": int(groups["tokens"]),
        "run_key": run_key,
        "nll": runs[run_key]["nll"],
        "ppl": runs[run_key]["ppl"],
        "float_ppl": float_ppl,
    }


def load_results(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            rows.append(parse_file(path))
        except Exception as e:
            print(f"[skip] {path.name}: {e}")

    if not rows:
        raise ValueError(f"No valid JSON files found in {input_dir}")

    df = pd.DataFrame(rows)
    return df


def make_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["hardware", "target", "tokens"],
        columns="mode",
        values="ppl",
        aggfunc="first",
    ).reset_index()

    for col in ["identity", "hadamardD", "learned"]:
        if col not in pivot.columns:
            pivot[col] = math.nan

    pivot["identity_over_hadamardD"] = pivot["identity"] / pivot["hadamardD"]
    pivot["identity_over_learned"] = pivot["identity"] / pivot["learned"]
    pivot["hadamardD_over_learned"] = pivot["hadamardD"] / pivot["learned"]

    return pivot


def plot_ppl_bars(df: pd.DataFrame, output_dir: Path) -> None:
    mode_order = ["identity", "hadamardD", "learned"]

    for (hardware, target), sub in df.groupby(["hardware", "target"]):
        sub = sub.copy()
        sub["mode"] = pd.Categorical(sub["mode"], categories=mode_order, ordered=True)
        sub = sub.sort_values("mode")

        plt.figure(figsize=(7, 5))
        plt.bar(sub["mode"].astype(str), sub["ppl"])
        plt.yscale("log")
        plt.ylabel("Perplexity, log scale")
        plt.title(f"PPL: {hardware} / {target}")
        plt.tight_layout()

        out = output_dir / f"ppl_{hardware}_{target}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"saved {out}")


def plot_improvement_bars(pivot: pd.DataFrame, output_dir: Path) -> None:
    rows = pivot.dropna(subset=["identity_over_hadamardD", "identity_over_learned"], how="all")

    labels = [
        f"{row.hardware}\n{row.target}"
        for row in rows.itertuples(index=False)
    ]

    x = range(len(rows))
    width = 0.35

    plt.figure(figsize=(max(9, len(rows) * 1.2), 5))
    plt.bar(
        [i - width / 2 for i in x],
        rows["identity_over_hadamardD"],
        width,
        label="Identity / Hadamard-D",
    )
    plt.bar(
        [i + width / 2 for i in x],
        rows["identity_over_learned"],
        width,
        label="Identity / Learned",
    )
    plt.yscale("log")
    plt.ylabel("Improvement ratio, log scale")
    plt.title("Analog PPL improvement over identity")
    plt.xticks(list(x), labels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()

    out = output_dir / "improvement_ratios.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


def plot_learned_vs_hadamard(pivot: pd.DataFrame, output_dir: Path) -> None:
    rows = pivot.dropna(subset=["hadamardD_over_learned"])

    labels = [
        f"{row.hardware}\n{row.target}"
        for row in rows.itertuples(index=False)
    ]

    plt.figure(figsize=(max(9, len(rows) * 1.2), 5))
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.bar(labels, rows["hadamardD_over_learned"])
    plt.ylabel("Hadamard-D PPL / Learned PPL")
    plt.title("Learned rotation vs fixed Hadamard-D\n>1 means learned is better")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = output_dir / "learned_vs_hadamardD.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/ablation")
    parser.add_argument("--output-dir", default="results/ablation_plots")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(input_dir)
    pivot = make_pivot(df)

    raw_csv = output_dir / "ablation_raw.csv"
    summary_csv = output_dir / "ablation_summary.csv"

    df.to_csv(raw_csv, index=False)
    pivot.to_csv(summary_csv, index=False)

    print(f"saved {raw_csv}")
    print(f"saved {summary_csv}")

    plot_ppl_bars(df, output_dir)
    plot_improvement_bars(pivot, output_dir)
    plot_learned_vs_hadamard(pivot, output_dir)


if __name__ == "__main__":
    main()