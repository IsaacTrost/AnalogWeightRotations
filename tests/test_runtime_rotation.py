import copy
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
    rotate_blockwise_input_weight_tensor,
    rotate_blockwise_output_weight_tensor,
    rotate_input_weight_tensor,
    rotate_output_bias_tensor,
    rotate_output_weight_tensor,
)
from src.rotation_precision import ROTATION_COMPUTE_DTYPE  # noqa: E402
from src.rotation_utils import get_rotation_matrix  # noqa: E402
from src.runtime_rotation import (  # noqa: E402
    RotationParameters,
    RuntimeRotatedEmbedding,
    RuntimeRotatedLinear,
    enable_runtime_attention_rotations,
)


class _FakeAttention(torch.nn.Module):
    """Mirror the LLaMA attention projection layout used by the runtime rotation wrappers."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)


class _FakeMlp(torch.nn.Module):
    """Provide the MLP modules expected by the surrounding fake LLaMA model."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)


class _FakeLayer(torch.nn.Module):
    """Bundle the attention and MLP modules into one decoder block."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = _FakeAttention(hidden_size)
        self.mlp = _FakeMlp(hidden_size)


class _FakeInnerModel(torch.nn.Module):
    """Expose embeddings and decoder layers under the standard `model.*` names."""

    def __init__(self, hidden_size: int, num_layers: int, vocab_size: int = 32) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(_FakeLayer(hidden_size) for _ in range(num_layers))


class _FakeLlama(torch.nn.Module):
    """Provide the minimal config and module layout needed by the runtime rotation helpers."""

    def __init__(self, hidden_size: int = 8, num_heads: int = 2, num_layers: int = 1) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, num_attention_heads=num_heads)
        self.model = _FakeInnerModel(hidden_size, num_layers)
        self.lm_head = torch.nn.Linear(hidden_size, hidden_size, bias=False)


class RuntimeRotationTests(unittest.TestCase):
    """Check that runtime attention rotations match the existing static rewrite formulas."""

    def setUp(self) -> None:
        """Use a fixed seed so runtime and static models start from identical weights."""
        torch.manual_seed(0)

    def test_enable_runtime_attention_rotations_wraps_attention_projections(self) -> None:
        """The runtime path should wrap embeddings, attention, MLP, and head modules in one pass."""
        model = _FakeLlama(hidden_size=8, num_heads=2, num_layers=2)
        rotation_parameters = RotationParameters.for_model(
            model,
            rotate_mode="identity",
            r2_mode="identity",
        )

        rotation_state = enable_runtime_attention_rotations(model, rotation_parameters=rotation_parameters)

        self.assertIsInstance(model.model.embed_tokens, RuntimeRotatedEmbedding)
        self.assertIsInstance(model.model.layers[0].self_attn.q_proj, RuntimeRotatedLinear)
        self.assertIsInstance(model.model.layers[0].self_attn.k_proj, RuntimeRotatedLinear)
        self.assertIsInstance(model.model.layers[0].self_attn.v_proj, RuntimeRotatedLinear)
        self.assertIsInstance(model.model.layers[0].self_attn.o_proj, RuntimeRotatedLinear)
        self.assertIsInstance(model.model.layers[0].mlp.up_proj, RuntimeRotatedLinear)
        self.assertIsInstance(model.model.layers[0].mlp.gate_proj, RuntimeRotatedLinear)
        self.assertIsInstance(model.model.layers[0].mlp.down_proj, RuntimeRotatedLinear)
        self.assertIsInstance(model.lm_head, RuntimeRotatedLinear)
        self.assertIn("R1", rotation_state)
        self.assertEqual(
            set(rotation_state["layers"].keys()),
            {
                "model.layers.0.self_attn.R2",
                "model.layers.1.self_attn.R2",
            },
        )

    def test_runtime_attention_outputs_match_static_rewrite(self) -> None:
        """For fixed `R1` and `R2`, wrapped runtime attention projections should match static rewrites."""
        static_model = _FakeLlama(hidden_size=8, num_heads=2, num_layers=1)
        runtime_model = copy.deepcopy(static_model)
        hidden_states = torch.randn(2, 3, 8, dtype=torch.float32)
        attn_output = torch.randn(2, 3, 8, dtype=torch.float32)
        head_dim = 4
        r1 = get_rotation_matrix(
            8,
            mode="random",
            device="cpu",
            dtype=ROTATION_COMPUTE_DTYPE,
            seed=7,
        )
        r2 = get_rotation_matrix(
            head_dim,
            mode="random",
            device="cpu",
            dtype=ROTATION_COMPUTE_DTYPE,
            seed=13,
        )

        static_layer = static_model.model.layers[0]
        static_layer.self_attn.q_proj.weight.data = rotate_input_weight_tensor(static_layer.self_attn.q_proj.weight, r1)
        static_layer.self_attn.k_proj.weight.data = rotate_input_weight_tensor(static_layer.self_attn.k_proj.weight, r1)
        static_layer.self_attn.v_proj.weight.data = rotate_blockwise_output_weight_tensor(
            rotate_input_weight_tensor(static_layer.self_attn.v_proj.weight, r1),
            r2,
            head_dim,
        )
        static_layer.self_attn.o_proj.weight.data = rotate_blockwise_input_weight_tensor(
            rotate_output_weight_tensor(static_layer.self_attn.o_proj.weight, r1),
            r2,
            head_dim,
        )
        static_layer.self_attn.o_proj.bias.data = rotate_output_bias_tensor(static_layer.self_attn.o_proj.bias, r1)

        rotation_parameters = RotationParameters.for_model(
            runtime_model,
            rotate_mode="identity",
            r2_mode="identity",
        )
        with torch.no_grad():
            rotation_parameters.R1.copy_(r1)
            rotation_parameters.get_layer_r2(0).copy_(r2)
        enable_runtime_attention_rotations(runtime_model, rotation_parameters=rotation_parameters)

        runtime_layer = runtime_model.model.layers[0]
        self.assertTrue(torch.allclose(static_layer.self_attn.q_proj(hidden_states), runtime_layer.self_attn.q_proj(hidden_states)))
        self.assertTrue(torch.allclose(static_layer.self_attn.k_proj(hidden_states), runtime_layer.self_attn.k_proj(hidden_states)))
        self.assertTrue(torch.allclose(static_layer.self_attn.v_proj(hidden_states), runtime_layer.self_attn.v_proj(hidden_states)))
        self.assertTrue(torch.allclose(static_layer.self_attn.o_proj(attn_output), runtime_layer.self_attn.o_proj(attn_output)))

    def test_runtime_non_attention_outputs_match_static_rewrite(self) -> None:
        """The runtime path should match static rewrites for embeddings, MLP projections, and lm_head."""
        static_model = _FakeLlama(hidden_size=8, num_heads=2, num_layers=1)
        runtime_model = copy.deepcopy(static_model)
        token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        hidden_states = torch.randn(2, 3, 8, dtype=torch.float32)
        r1 = get_rotation_matrix(
            8,
            mode="random",
            device="cpu",
            dtype=ROTATION_COMPUTE_DTYPE,
            seed=19,
        )

        static_model.model.embed_tokens.weight.data = rotate_input_weight_tensor(
            static_model.model.embed_tokens.weight,
            r1,
        )
        static_layer = static_model.model.layers[0]
        static_layer.mlp.up_proj.weight.data = rotate_input_weight_tensor(static_layer.mlp.up_proj.weight, r1)
        static_layer.mlp.gate_proj.weight.data = rotate_input_weight_tensor(
            static_layer.mlp.gate_proj.weight,
            r1,
        )
        static_layer.mlp.down_proj.weight.data = rotate_output_weight_tensor(
            static_layer.mlp.down_proj.weight,
            r1,
        )
        static_model.lm_head.weight.data = rotate_input_weight_tensor(static_model.lm_head.weight, r1)

        rotation_parameters = RotationParameters.for_model(
            runtime_model,
            rotate_mode="identity",
            r2_mode="identity",
        )
        with torch.no_grad():
            rotation_parameters.R1.copy_(r1)
        enable_runtime_attention_rotations(runtime_model, rotation_parameters=rotation_parameters)

        runtime_layer = runtime_model.model.layers[0]
        self.assertTrue(
            torch.allclose(
                static_model.model.embed_tokens(token_ids),
                runtime_model.model.embed_tokens(token_ids),
            )
        )
        self.assertTrue(torch.allclose(static_layer.mlp.up_proj(hidden_states), runtime_layer.mlp.up_proj(hidden_states)))
        self.assertTrue(
            torch.allclose(static_layer.mlp.gate_proj(hidden_states), runtime_layer.mlp.gate_proj(hidden_states))
        )
        self.assertTrue(
            torch.allclose(static_layer.mlp.down_proj(hidden_states), runtime_layer.mlp.down_proj(hidden_states))
        )
        self.assertTrue(torch.allclose(static_model.lm_head(hidden_states), runtime_model.lm_head(hidden_states)))


if __name__ == "__main__":
    unittest.main()
