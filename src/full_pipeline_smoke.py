import json
import pathlib
import sys
from dataclasses import dataclass
from unittest.mock import patch

import torch

# Add the repo root so the smoke script can be run directly from the repository.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.full_model_pipeline import PipelineConfig, run_pipeline


@dataclass
class FakeModel:
    """Track which pipeline stages ran during the smoke test."""

    prepared: bool = False
    rotated: bool = False
    analog_converted: bool = False


def _fake_forward_payload(model: FakeModel) -> dict:
    """Return deterministic tensors so the pipeline comparison code stays real."""
    analog_offset = 0.25 if model.analog_converted else 0.0
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32) + analog_offset
    hidden = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32) + analog_offset
    module_output = torch.tensor([[0.5, 0.6]], dtype=torch.float32) + analog_offset
    return {
        "inputs": {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)},
        "logits": logits,
        "hidden_states": (hidden,),
        "module_outputs": {"lm_head": module_output},
    }


def _fake_load_model_and_tokenizer(*_args, **_kwargs):
    """Return a small fake pair so the smoke test does not need model downloads."""
    return FakeModel(), object()


def _fake_prepare_model_for_rotation(model: FakeModel) -> None:
    """Record the preparation stage without changing the test tensors."""
    model.prepared = True


def _fake_rotate_model(model: FakeModel, *_args, **_kwargs) -> dict:
    """Record the rotation stage and hand back a checkpoint-shaped rotation payload."""
    model.rotated = True
    return {
        "R1": torch.eye(2, dtype=torch.float64),
        "layers": {"model.layers.0.self_attn.R2": torch.eye(1, dtype=torch.float64)},
        "metadata": {
            "rotate_mode": "identity",
            "r2_mode": "identity",
            "seed": 0,
            "r2_seed_offset": 1,
            "hidden_size": 2,
            "num_attention_heads": 2,
            "head_dim": 1,
        },
    }


def _fake_enable_runtime_attention_rotations(model: FakeModel, *_args, **_kwargs) -> dict:
    """Record the runtime rotation stage and hand back the same checkpoint-shaped payload."""
    model.rotated = True
    return _fake_rotate_model(model)


def _fake_convert_llama_linears_to_analog(model: FakeModel, *_args, **_kwargs) -> list[str]:
    """Record the analog conversion stage and report one converted module."""
    model.analog_converted = True
    return ["model.layers.0.mlp.down_proj"]


def run_smoke_tests() -> dict:
    """Exercise the float and analog branches of the pipeline with real comparisons."""
    with patch("src.full_model_pipeline.load_model_and_tokenizer", side_effect=_fake_load_model_and_tokenizer), patch(
        "src.full_model_pipeline.is_llama_like_model", return_value=True
    ), patch(
        "src.full_model_pipeline.prepare_model_for_rotation",
        side_effect=_fake_prepare_model_for_rotation,
    ), patch("src.full_model_pipeline.rotate_model", side_effect=_fake_rotate_model), patch(
        "src.full_model_pipeline.enable_runtime_attention_rotations",
        side_effect=_fake_enable_runtime_attention_rotations,
    ), patch(
        "src.full_model_pipeline.run_verification_forward",
        side_effect=lambda model, tokenizer, **kwargs: _fake_forward_payload(model),
    ), patch(
        "src.full_model_pipeline.convert_llama_linears_to_analog",
        side_effect=_fake_convert_llama_linears_to_analog,
    ):
        float_only = run_pipeline(
            PipelineConfig(
                model_name="smoke/fake-llama",
                rotate_mode="identity",
                max_length=16,
                texts=("hello world",),
                prepare_model=True,
                convert_analog=False,
            )
        )
        runtime_only = run_pipeline(
            PipelineConfig(
                model_name="smoke/fake-llama",
                rotate_mode="identity",
                rotation_backend="runtime",
                max_length=16,
                texts=("hello world",),
                prepare_model=True,
                convert_analog=False,
            )
        )
        analog_enabled = run_pipeline(
            PipelineConfig(
                model_name="smoke/fake-llama",
                rotate_mode="identity",
                max_length=16,
                texts=("hello world",),
                prepare_model=True,
                convert_analog=True,
                analog_targets=("down_proj",),
            )
        )

    assert float_only["float_equivalence"]["next_token_match"] is True
    assert float_only["float_equivalence"]["logits"]["max_abs"] == 0.0
    assert float_only["rotation_backend"] == "static"
    assert "model.layers.0.self_attn.R2" in float_only["rotation_state"]["layers"]
    assert runtime_only["float_equivalence"]["next_token_match"] is True
    assert runtime_only["float_equivalence"]["logits"]["max_abs"] == 0.0
    assert runtime_only["rotation_backend"] == "runtime"
    assert "model.layers.0.self_attn.R2" in runtime_only["rotation_state"]["layers"]
    assert float_only["analog_targets"] == []
    assert analog_enabled["analog_targets"] == ["model.layers.0.mlp.down_proj"]
    assert analog_enabled["analog_comparison"]["logits"]["max_abs"] > 0.0

    return {
        "float_only": float_only,
        "runtime_only": runtime_only,
        "analog_enabled": analog_enabled,
    }


def main() -> None:
    """Run the smoke tests and print the pipeline results as formatted JSON."""
    results = run_smoke_tests()
    print(
        json.dumps(
            results,
            indent=2,
            default=lambda value: value.tolist() if isinstance(value, torch.Tensor) else str(value),
        )
    )


if __name__ == "__main__":
    main()
