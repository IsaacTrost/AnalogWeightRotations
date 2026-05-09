"""Convert a SpinQuant R.bin file into this repo's rotation checkpoint format.

SpinQuant stores rotations as a flat state dict:
  - R1
  - model.layers.{idx}.self_attn.R2

The analog evaluation path accepts a checkpoint-shaped dict:
  - R1: global residual-stream rotation
  - layers: per-layer R2 rotations keyed by model.layers.{idx}.self_attn.R2
  - metadata: lightweight provenance for result logs
"""
import argparse
import os
from collections.abc import Mapping

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(
    REPO_ROOT,
    "rotations",
    "spinquant-tinyllama-1.1b-w16a4kv4",
    "R.bin",
)


def _default_output_path(input_path: str) -> str:
    """Place the converted checkpoint next to the SpinQuant rotation folder."""
    rotation_dir = os.path.dirname(os.path.abspath(input_path))
    parent_dir = os.path.dirname(rotation_dir)
    return os.path.join(parent_dir, f"{os.path.basename(rotation_dir)}.pt")


def _find_r1_key(state: Mapping[str, torch.Tensor]) -> str:
    """Find the global R1 key while tolerating wrapper prefixes from training."""
    candidates = [key for key in state if key == "R1" or key.endswith(".R1")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one R1 tensor, found {candidates}.")
    return candidates[0]


def _collect_r2_layers(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Collect per-layer OV rotations in the key format used by the evaluator."""
    layers = {}
    for key, value in state.items():
        if "self_attn.R2" not in key:
            continue
        if ".layers." in key:
            suffix = key[key.index("model.layers.") :] if "model.layers." in key else key
            layers[suffix] = value
            continue
        if key.startswith("layer_"):
            layer_idx = int(key.split("_", 1)[1])
            layers[f"model.layers.{layer_idx}.self_attn.R2"] = value

    if not layers:
        raise ValueError("No per-layer self_attn.R2 tensors found in the SpinQuant checkpoint.")
    return dict(
        sorted(
            layers.items(),
            key=lambda item: int(item[0].split(".layers.", 1)[1].split(".", 1)[0]),
        )
    )


def convert_spinquant_rotation(input_path: str, output_path: str) -> dict:
    """Load SpinQuant rotations and write a checkpoint consumable by analog eval."""
    state = torch.load(input_path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected {input_path} to contain a mapping, got {type(state)!r}.")

    r1_key = _find_r1_key(state)
    layers = _collect_r2_layers(state)
    converted = {
        "R1": state[r1_key],
        "layers": layers,
        "metadata": {
            "source_format": "spinquant_R_bin",
            "rotate_mode": "spinquant_checkpoint",
            "r2_mode": "spinquant_checkpoint",
            "source_path": os.path.abspath(input_path),
            "num_r2_layers": len(layers),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(converted, output_path)
    return converted


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI used to convert SpinQuant rotation binaries."""
    parser = argparse.ArgumentParser(
        description="Convert SpinQuant R.bin rotations to eval_analog_perplexity.py format."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the SpinQuant R.bin file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination .pt checkpoint. Defaults to rotations/<spinquant-folder>.pt.",
    )
    return parser


def main() -> None:
    """Convert the requested file and print the output path for shell workflows."""
    args = build_arg_parser().parse_args()
    output_path = args.output or _default_output_path(args.input)
    converted = convert_spinquant_rotation(args.input, output_path)
    print(f"Saved converted rotation to {output_path}")
    print(f"R1 shape: {tuple(converted['R1'].shape)}")
    print(f"R2 layers: {len(converted['layers'])}")


if __name__ == "__main__":
    main()
