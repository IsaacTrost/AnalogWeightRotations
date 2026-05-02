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
    bake_rotation_state_into_model,
    identity_rotation_state,
    rotate_input_weight_tensor,
    rotate_output_bias_tensor,
    rotate_output_weight_tensor,
    rotate_model,
)
from src.rotation_precision import ROTATION_COMPUTE_DTYPE  # noqa: E402
from src.rotation_utils import get_rotation_matrix, hadamard_matrix  # noqa: E402
from src.runtime_rotation import build_runtime_linear_weight_and_bias  # noqa: E402
from src.analog_llama import apply_block_hadamard  # noqa: E402


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

    def test_explicit_checkpoint_baker_matches_runtime_weight_builder(self) -> None:
        """Baking a checkpoint-shaped R1/R2 state should match the runtime effective weights."""
        model = _FakeLlama(hidden_size=8, num_heads=2, num_layers=1)
        before = {
            "q": model.model.layers[0].self_attn.q_proj.weight.detach().clone(),
            "v": model.model.layers[0].self_attn.v_proj.weight.detach().clone(),
            "o": model.model.layers[0].self_attn.o_proj.weight.detach().clone(),
            "down": model.model.layers[0].mlp.down_proj.weight.detach().clone(),
            "head": model.lm_head.weight.detach().clone(),
        }
        r1 = get_rotation_matrix(8, mode="random", device="cpu", dtype=ROTATION_COMPUTE_DTYPE, seed=31)
        r2 = get_rotation_matrix(4, mode="random", device="cpu", dtype=ROTATION_COMPUTE_DTYPE, seed=37)

        bake_rotation_state_into_model(
            model,
            {"R1": r1, "R2": {"layer_0": r2}, "metadata": {"rotate_mode": "checkpoint"}},
        )

        expected_q, _ = build_runtime_linear_weight_and_bias(before["q"], None, r1, "input")
        expected_v, _ = build_runtime_linear_weight_and_bias(before["v"], None, r1, "input", r2, "output", 4)
        expected_o, _ = build_runtime_linear_weight_and_bias(before["o"], None, r1, "output", r2, "input", 4)
        expected_down, _ = build_runtime_linear_weight_and_bias(before["down"], None, r1, "output")
        expected_head, _ = build_runtime_linear_weight_and_bias(before["head"], None, r1, "input")

        layer = model.model.layers[0]
        self.assertTrue(torch.allclose(layer.self_attn.q_proj.weight, expected_q))
        self.assertTrue(torch.allclose(layer.self_attn.v_proj.weight, expected_v))
        self.assertTrue(torch.allclose(layer.self_attn.o_proj.weight, expected_o))
        self.assertTrue(torch.allclose(layer.mlp.down_proj.weight, expected_down))
        self.assertTrue(torch.allclose(model.lm_head.weight, expected_head))

    def test_identity_checkpoint_baker_leaves_weights_unchanged(self) -> None:
        """Identity R1/R2 should make the explicit baker a no-op on linear weights."""
        model = _FakeLlama(hidden_size=8, num_heads=2, num_layers=2)
        before = {name: module.weight.detach().clone() for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}

        bake_rotation_state_into_model(model, identity_rotation_state(model))

        after = {name: module.weight.detach().clone() for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
        self.assertEqual(set(before), set(after))
        for name in before:
            self.assertTrue(torch.equal(before[name], after[name]), msg=name)

    def test_online_hadamard_weight_and_activation_pair_is_float_equivalent(self) -> None:
        """The R3/R4-style input Hadamard is exact when the weight side is pre-rotated."""
        x = torch.randn(2, 3, 8, dtype=torch.float32)
        weight = torch.randn(5, 8, dtype=torch.float32)
        bias = torch.randn(5, dtype=torch.float32)
        hadamard = hadamard_matrix(4, device="cpu", dtype=ROTATION_COMPUTE_DTYPE)

        baseline = torch.nn.functional.linear(x, weight, bias)
        rotated_x = apply_block_hadamard(x, hadamard)
        rotated_weight = apply_block_hadamard(weight, hadamard)
        rotated = torch.nn.functional.linear(rotated_x, rotated_weight, bias)

        self.assertTrue(torch.allclose(baseline, rotated, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
