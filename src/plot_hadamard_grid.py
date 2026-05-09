#!/usr/bin/env python3
"""Plot Hadamard-D grid-search results."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import pandas as pd

from src.wandb_config import WANDB_ENTITY, WANDB_MODE, WANDB_PROJECT


def load_summary(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"No rows in {input_path}")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["ppl"] = pd.to_numeric(df["ppl"], errors="coerce")
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


def plot_hadamard_heatmaps(hadamard: pd.DataFrame, output_dir: Path, metric: str) -> list[Path]:
    label = metric_label(metric)
    outputs = []
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
        outputs.append(out)
    return outputs


def plot_all_method_improvement_heatmaps(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    plot_df = df[df["run_kind"].isin(["identity", "hadamard_D", "checkpoint"])].copy()
    plot_df["series"] = [run_label(row) for row in plot_df.itertuples(index=False)]
    plot_df["method_improvement_ratio"] = plot_df["improvement_ratio"]
    plot_df.loc[plot_df["run_kind"] == "identity", "method_improvement_ratio"] = 1.0
    plot_df = plot_df.dropna(
        subset=["method_improvement_ratio", "ir_drop", "input_bits", "weight_noise"]
    )
    if plot_df.empty:
        raise ValueError("No identity, Hadamard-D, or checkpoint rows with improvement values.")

    outputs = []
    for weight_noise, sub in plot_df.groupby("weight_noise"):
        bits_order = input_bits_order(pd.Index(sub["input_bits"].unique()))
        ir_order = sorted(sub["ir_drop"].unique())
        preferred = ["Identity", "Hadamard-D"]
        series_order = [name for name in preferred if name in set(sub["series"])]
        series_order.extend(
            name for name in sorted(sub["series"].unique()) if name not in series_order
        )

        fig, axes = plt.subplots(
            1,
            len(series_order),
            figsize=(max(5 * len(series_order), 7), max(4, 0.6 * len(bits_order))),
            squeeze=False,
        )
        values = sub["method_improvement_ratio"]
        norm = None
        if values.min() < 1.0 < values.max():
            norm = TwoSlopeNorm(vcenter=1.0, vmin=values.min(), vmax=values.max())

        image = None
        for axis, series in zip(axes[0], series_order):
            pivot = sub[sub["series"] == series].pivot_table(
                index="input_bits",
                columns="ir_drop",
                values="method_improvement_ratio",
                aggfunc="first",
            )
            pivot = pivot.reindex(index=bits_order, columns=ir_order)
            image = axis.imshow(
                pivot.values,
                aspect="auto",
                origin="upper",
                cmap="RdYlGn",
                norm=norm,
            )
            axis.set_title(series)
            axis.set_xticks(range(len(ir_order)), [f"{value:g}" for value in ir_order])
            axis.set_yticks(range(len(bits_order)), [str(int(value)) for value in bits_order])
            axis.set_xlabel("IR drop")
            axis.set_ylabel("Input bits (-1 = off)")
            for row_idx, input_bits in enumerate(pivot.index):
                for col_idx, ir_drop in enumerate(pivot.columns):
                    value = pivot.loc[input_bits, ir_drop]
                    if pd.notna(value):
                        axis.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center")

        if image is not None:
            fig.colorbar(image, ax=axes[0].tolist(), label="Identity PPL / Method PPL")
        fig.suptitle(f"Improvement Ratio by Method, weight_noise={weight_noise:g}")
        out = output_dir / f"all_methods_improvement_ratio_wnoise{f'{weight_noise:g}'.replace('-', 'm').replace('.', 'p')}.png"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")
        outputs.append(out)
    return outputs


def run_label(row) -> str:
    if row.run_kind == "identity":
        return "Identity"
    if row.run_kind == "hadamard_D":
        return "Hadamard-D"
    if row.run_kind == "checkpoint":
        checkpoint_label = getattr(row, "checkpoint_label", "")
        if isinstance(checkpoint_label, str) and checkpoint_label:
            return checkpoint_label
        checkpoint_path = getattr(row, "checkpoint_path", "")
        if isinstance(checkpoint_path, str) and checkpoint_path:
            return Path(checkpoint_path).stem
        return "Checkpoint"
    return str(row.run_kind)


def plot_absolute_ppl_bars(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    plot_df = df[df["run_kind"].isin(["identity", "hadamard_D", "checkpoint"])].copy()
    plot_df = plot_df.dropna(subset=["ppl", "ir_drop", "input_bits", "weight_noise"])
    if plot_df.empty:
        raise ValueError("No identity, Hadamard-D, or checkpoint rows with PPL values.")

    plot_df["series"] = [run_label(row) for row in plot_df.itertuples(index=False)]

    outputs = []
    for weight_noise, sub in plot_df.groupby("weight_noise"):
        sub = sub.copy()
        bits_order = input_bits_order(pd.Index(sub["input_bits"].unique()))
        sub["input_bits"] = pd.Categorical(sub["input_bits"], categories=bits_order, ordered=True)

        pivot = sub.pivot_table(
            index=["input_bits", "ir_drop"],
            columns="series",
            values="ppl",
            aggfunc="first",
            observed=False,
        )
        pivot = pivot.sort_index(level=["input_bits", "ir_drop"])

        preferred = ["Identity", "Hadamard-D"]
        series_order = [name for name in preferred if name in pivot.columns]
        series_order.extend(name for name in pivot.columns if name not in series_order)
        pivot = pivot[series_order]

        x = list(range(len(pivot)))
        width = min(0.8 / max(len(pivot.columns), 1), 0.28)
        labels = [f"bits={int(bits)}\nir={ir:g}" for bits, ir in pivot.index]

        plt.figure(figsize=(max(10, 0.65 * len(pivot)), 5))
        for series_idx, series in enumerate(pivot.columns):
            offset = (series_idx - (len(pivot.columns) - 1) / 2) * width
            plt.bar(
                [value + offset for value in x],
                pivot[series],
                width=width,
                label=series,
            )
        plt.xticks(x, labels, rotation=60, ha="right")
        plt.ylabel("Perplexity")
        plt.ylim(top=150)
        plt.xlabel("Hardware cell")
        plt.title(f"Absolute Perplexity by Rotation, weight_noise={weight_noise:g}")
        plt.legend()
        plt.tight_layout()
        out = output_dir / f"absolute_ppl_bars_wnoise{f'{weight_noise:g}'.replace('-', 'm').replace('.', 'p')}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"saved {out}")
        outputs.append(out)
    return outputs


def log_to_wandb(
    *,
    input_path: Path,
    output_dir: Path,
    hadamard_csv: Path,
    image_paths: list[Path],
    metric: str,
    max_ir_drop: float,
    run_name: str | None,
) -> None:
    import wandb

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        mode=WANDB_MODE,
        name=run_name or f"hadamard_grid_plots_{metric}",
        job_type="hadamard_grid_plots",
        config={
            "input": str(input_path),
            "output_dir": str(output_dir),
            "metric": metric,
            "max_ir_drop": max_ir_drop,
        },
    )

    log_payload = {
        path.stem: wandb.Image(str(path))
        for path in image_paths
        if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    if log_payload:
        wandb.log(log_payload)

    artifact = wandb.Artifact(
        name=f"hadamard-grid-plots-{metric}",
        type="plots",
        metadata={
            "input": str(input_path),
            "metric": metric,
            "max_ir_drop": max_ir_drop,
        },
    )
    artifact.add_file(str(hadamard_csv))
    for path in image_paths:
        if path.exists():
            artifact.add_file(str(path))
    run.log_artifact(artifact)
    wandb.finish()
    print(f"uploaded {len(image_paths)} plots and {hadamard_csv} to wandb")


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
    parser.add_argument(
        "--max-ir-drop",
        type=float,
        default=2.0,
        help="Only plot hardware cells with ir_drop <= this value.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Do not upload generated plots to W&B.")
    parser.add_argument("--wandb-name", default=None, help="Optional W&B run name.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(input_path)
    df = df[df["ir_drop"] <= args.max_ir_drop].copy()
    if df.empty:
        raise ValueError(f"No rows remain after filtering ir_drop <= {args.max_ir_drop:g}.")
    hadamard = hadamard_table(df, args.metric)
    hadamard_csv = output_dir / "hadamard_by_cell.csv"
    hadamard.to_csv(hadamard_csv, index=False)
    print(f"saved {hadamard_csv}")

    image_paths = plot_hadamard_heatmaps(hadamard, output_dir, args.metric)
    if args.metric == "improvement_ratio":
        image_paths.extend(plot_all_method_improvement_heatmaps(df, output_dir))
    image_paths.extend(plot_absolute_ppl_bars(df, output_dir))

    if not args.no_wandb:
        log_to_wandb(
            input_path=input_path,
            output_dir=output_dir,
            hadamard_csv=hadamard_csv,
            image_paths=image_paths,
            metric=args.metric,
            max_ir_drop=args.max_ir_drop,
            run_name=args.wandb_name,
        )


if __name__ == "__main__":
    main()
