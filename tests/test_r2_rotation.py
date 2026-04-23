import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch


# Add the repo root so the tests can import the local `src` modules without packaging changes.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llama_rotation import (  # noqa: E402
    _rotate_blockwise_input_weight,
    _rotate_blockwise_output_weight,
    rotate_input_weight_tensor,
    rotate_output_bias_tensor,
    rotate_output_weight_tensor,
    rotate_model,
)
from src.rotation_precision import ROTATION_COMPUTE_DTYPE  # noqa: E402
from src.rotation_utils import get_rotation_matrix  # noqa: E402
from src.runtime_rotation import build_runtime_linear_weight_and_bias  # noqa: E402


class _FakeAttention(torch.nn.Module):
    """Mirror the LLaMA attention projection layout used by the rotation helpers."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)


class _FakeMlp(torch.nn.Module):
    """Provide the MLP modules that `rotate_model()` expects on each decoder layer."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)


class _FakeLayer(torch.nn.Module):
    """Bundle the attention and MLP modules into a LLaMA-shaped decoder block."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = _FakeAttention(hidden_size)
        self.mlp = _FakeMlp(hidden_size)


class _FakeInnerModel(torch.nn.Module):
    """Expose the embedding table and layer list under the standard `model.*` names."""

    def __init__(self, hidden_size: int, num_layers: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(_FakeLayer(hidden_size) for _ in range(num_layers))


class _FakeLlama(torch.nn.Module):
    """Provide the minimal config and module layout consumed by the rotation pipeline."""

    def __init__(self, hidden_size: int = 8, num_heads: int = 2, num_layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, num_attention_heads=num_heads)
        self.model = _FakeInnerModel(hidden_size, num_layers)
        self.lm_head = torch.nn.Linear(hidden_size, hidden_size, bias=False)


class R2RotationTests(unittest.TestCase):
    """Check that the OV-path `R2` helper preserves correct math and exposes stable state."""

    def setUp(self) -> None:
        """Use a fixed seed so the negative tests fail only when the math is actually wrong."""
        torch.manual_seed(0)

    def test_correct_r2_pair_preserves_ov_output(self) -> None:
        """Applying `R2` to `v_proj` and the paired inverse to `o_proj` should preserve the OV path."""
        hidden_size = 12
        head_dim = 4
        batch_size = 5
        inputs = torch.randn(batch_size, hidden_size, dtype=torch.float32)
        v_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
        o_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
        r2 = get_rotation_matrix(
            head_dim,
            mode="random",
            device="cpu",
            dtype=ROTATION_COMPUTE_DTYPE,
            seed=11,
        ).to(torch.float32)

        baseline = (inputs @ v_weight.T) @ o_weight.T
        rotated_v = _rotate_blockwise_output_weight(v_weight, r2, head_dim)
        rotated_o = _rotate_blockwise_input_weight(o_weight, r2, head_dim)
        rotated = (inputs @ rotated_v.T) @ rotated_o.T

        self.assertTrue(torch.allclose(baseline, rotated, atol=1e-5, rtol=1e-5))

    def test_incorrect_r2_application_changes_ov_output(self) -> None:
        """Rotating only one side of the OV pair should break equivalence and move the output."""
        hidden_size = 12
        head_dim = 4
        batch_size = 5
        inputs = torch.randn(batch_size, hidden_size, dtype=torch.float32)
        v_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
        o_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
        r2 = get_rotation_matrix(
            head_dim,
            mode="random",
            device="cpu",
            dtype=ROTATION_COMPUTE_DTYPE,
            seed=17,
        ).to(torch.float32)

        baseline = (inputs @ v_weight.T) @ o_weight.T
        rotated_v_only = _rotate_blockwise_output_weight(v_weight, r2, head_dim)
        broken = (inputs @ rotated_v_only.T) @ o_weight.T

        self.assertGreater((baseline - broken).abs().max().item(), 1e-3)

    def test_rotate_model_records_named_r2_and_updates_ov_weights(self) -> None:
        """`rotate_model()` should emit stable `R2` names and apply them to the OV weights only."""
        model = _FakeLlama(hidden_size=8, num_heads=2, num_layers=2)
        identity_r1 = torch.eye(model.config.hidden_size, dtype=ROTATION_COMPUTE_DTYPE)

        before_q = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
        before_v = model.model.layers[0].self_attn.v_proj.weight.detach().clone()
        before_o = model.model.layers[0].self_attn.o_proj.weight.detach().clone()

        rotation_state = rotate_model(
            model,
            rotation=identity_r1,
            rotate_mode="identity",
            r2_mode="random",
            seed=3,
            r2_seed_offset=100,
        )

        self.assertIn("R1", rotation_state)
        self.assertIn("layers", rotation_state)
        self.assertIn("metadata", rotation_state)
        self.assertEqual(rotation_state["metadata"]["head_dim"], 4)
        self.assertEqual(
            set(rotation_state["layers"].keys()),
            {
                "model.layers.0.self_attn.R2",
                "model.layers.1.self_attn.R2",
            },
        )
        self.assertTrue(torch.equal(before_q, model.model.layers[0].self_attn.q_proj.weight))
        self.assertFalse(torch.equal(before_v, model.model.layers[0].self_attn.v_proj.weight))
        self.assertFalse(torch.equal(before_o, model.model.layers[0].self_attn.o_proj.weight))

    def test_runtime_weight_builder_matches_manual_r1_r2_math(self) -> None:
        """The runtime helper should compose `R1` and blockwise `R2` in the same order as the static path."""
        hidden_size = 8
        head_dim = 4
        weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32)
        bias = torch.randn(hidden_size, dtype=torch.float32)
        r1 = get_rotation_matrix(
            hidden_size,
            mode="random",
            device="cpu",
            dtype=ROTATION_COMPUTE_DTYPE,
            seed=23,
        )
        r2 = get_rotation_matrix(
            head_dim,
            mode="random",
            device="cpu",
            dtype=ROTATION_COMPUTE_DTYPE,
            seed=29,
        )

        expected_v_weight = _rotate_blockwise_output_weight(rotate_input_weight_tensor(weight, r1), r2, head_dim)
        runtime_v_weight, runtime_v_bias = build_runtime_linear_weight_and_bias(
            weight,
            None,
            r1=r1,
            apply_r1="input",
            r2=r2,
            apply_r2="output",
            head_dim=head_dim,
        )
        self.assertTrue(torch.allclose(expected_v_weight, runtime_v_weight))
        self.assertIsNone(runtime_v_bias)

        expected_o_weight = _rotate_blockwise_input_weight(rotate_output_weight_tensor(weight, r1), r2, head_dim)
        expected_o_bias = rotate_output_bias_tensor(bias, r1)
        runtime_o_weight, runtime_o_bias = build_runtime_linear_weight_and_bias(
            weight,
            bias,
            r1=r1,
            apply_r1="output",
            r2=r2,
            apply_r2="input",
            head_dim=head_dim,
        )
        self.assertTrue(torch.allclose(expected_o_weight, runtime_o_weight))
        self.assertTrue(torch.allclose(expected_o_bias, runtime_o_bias))


if __name__ == "__main__":
    unittest.main()
