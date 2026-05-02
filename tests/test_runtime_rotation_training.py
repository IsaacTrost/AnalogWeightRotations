import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F


# Add the repo root so the tests can import the local `src` modules without packaging changes.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration_data import build_calibration_batches, load_calibration_texts  # noqa: E402
from src.cayley_optimizer import CayleySGDG  # noqa: E402
from src.rotation_utils import get_rotation_matrix, orthogonality_error  # noqa: E402
from src.runtime_rotation import enable_runtime_attention_rotations  # noqa: E402
from src.train_runtime_rotation import (  # noqa: E402
    RuntimeRotationTrainingConfig,
    build_cli_results,
    freeze_non_rotation_parameters,
    run_runtime_rotation_training,
)


class _FakeTokenizer:
    """Map short strings into small token ids so tests avoid transformer dependencies."""

    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=128):
        encoded = []
        for text in texts:
            token_ids = [((ord(char) - 96) % 15) + 1 for char in text.lower()[:max_length] if char.strip()]
            encoded.append(token_ids or [1])

        max_tokens = max(len(row) for row in encoded)
        padded = []
        mask = []
        for row in encoded:
            pad_length = max_tokens - len(row)
            padded.append(row + [self.pad_token_id] * pad_length)
            mask.append([1] * len(row) + [0] * pad_length)

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class _FakeAttention(torch.nn.Module):
    """Provide the four attention projections expected by the runtime wrappers."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        mixed = torch.tanh(query + key) * torch.sigmoid(value)
        return self.o_proj(mixed)


class _FakeMlp(torch.nn.Module):
    """Provide the MLP projections expected by the runtime wrappers."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = torch.relu(self.up_proj(hidden_states)) * torch.sigmoid(self.gate_proj(hidden_states))
        return self.down_proj(gated)


class _FakeLayer(torch.nn.Module):
    """Combine attention, MLP, and one extra frozen mix to make CE depend on the rotations."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = torch.nn.LayerNorm(hidden_size)
        self.self_attn = _FakeAttention(hidden_size)
        self.mlp = _FakeMlp(hidden_size)
        self.extra_mix = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_input = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + self.self_attn(attn_input)
        mlp_input = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(mlp_input)
        return self.extra_mix(hidden_states)


class _FakeInnerModel(torch.nn.Module):
    """Expose the embedding table and layer list under the standard `model.*` names."""

    def __init__(self, hidden_size: int, num_layers: int, vocab_size: int = 16) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(_FakeLayer(hidden_size) for _ in range(num_layers))
        self.norm = torch.nn.LayerNorm(hidden_size)


class _FakeLlamaForCausalLM(torch.nn.Module):
    """Provide a small causal-LM forward that matches the runtime training entrypoint."""

    def __init__(self, hidden_size: int = 8, num_heads: int = 2, num_layers: int = 1, vocab_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
        )
        self.model = _FakeInnerModel(hidden_size, num_layers, vocab_size=vocab_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ):
        hidden_states = self.model.embed_tokens(input_ids)
        all_hidden_states = [hidden_states]

        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
            all_hidden_states.append(hidden_states)

        hidden_states = self.model.norm(hidden_states)
        all_hidden_states.append(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else tuple(all_hidden_states),
            loss=loss,
        )


class RuntimeRotationTrainingTests(unittest.TestCase):
    """Check the calibration loader, optimizer, and runtime rotation training entrypoint."""

    def setUp(self) -> None:
        """Seed the tiny synthetic models so the optimizer behavior stays deterministic."""
        torch.manual_seed(0)

    def test_load_calibration_texts_supports_inline_jsonl_and_plain_text(self) -> None:
        """Calibration text resolution should work for inline text, JSONL rows, and plain lines."""
        self.assertEqual(load_calibration_texts(texts=("alpha", "beta")), ["alpha", "beta"])

        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = pathlib.Path(temp_dir) / "calibration.jsonl"
            jsonl_path.write_text(json.dumps({"text": "gamma"}) + "\n" + json.dumps("delta") + "\n")
            self.assertEqual(load_calibration_texts(data_path=str(jsonl_path)), ["gamma", "delta"])

            text_path = pathlib.Path(temp_dir) / "calibration.txt"
            text_path.write_text("epsilon\n\nzeta\n")
            self.assertEqual(load_calibration_texts(data_path=str(text_path)), ["epsilon", "zeta"])

    def test_build_calibration_batches_creates_masked_labels(self) -> None:
        """Padding positions should become `-100` in the generated labels."""
        tokenizer = _FakeTokenizer()
        batches = build_calibration_batches(
            tokenizer,
            texts=("ab", "abcd", "abc"),
            batch_size=2,
            device="cpu",
            max_length=8,
        )

        self.assertEqual(len(batches), 2)
        first_batch = batches[0]
        self.assertIn("labels", first_batch)
        self.assertTrue(torch.equal(first_batch["labels"][0, 2:], torch.full((2,), -100, dtype=torch.long)))

    def test_freeze_non_rotation_parameters_only_leaves_runtime_rotations_trainable(self) -> None:
        """Only `R1` and the per-layer `R2` matrices should stay trainable after freezing."""
        model = _FakeLlamaForCausalLM()
        enable_runtime_attention_rotations(model, rotate_mode="random", r2_mode="random")

        trainable_names = freeze_non_rotation_parameters(model)
        active_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

        self.assertEqual(
            set(active_names),
            {
                "runtime_rotation_parameters.R1",
                "runtime_rotation_parameters.layer_R2.layer_0",
            },
        )
        self.assertEqual(set(trainable_names), set(active_names))

    def test_cayley_optimizer_preserves_orthogonality_and_reduces_simple_loss(self) -> None:
        """The Cayley optimizer should keep an orthogonal matrix on-manifold while descending a loss."""
        target = torch.eye(4, dtype=torch.float64)
        parameter = torch.nn.Parameter(
            get_rotation_matrix(4, mode="random", device="cpu", dtype=torch.float64, seed=7)
        )
        optimizer = CayleySGDG([parameter], lr=0.25, momentum=0.0, stiefel=True)
        losses = []

        for _ in range(8):
            optimizer.zero_grad()
            loss = ((parameter - target) ** 2).sum()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        self.assertLess(losses[-1], losses[0])
        self.assertGreater((parameter - target).abs().max().item(), 0.0)
        self.assertLess(orthogonality_error(parameter.detach()), 1e-5)

    def test_build_cli_results_omits_rotation_state_and_adds_history_summary(self) -> None:
        """The CLI view should stay compact while still surfacing the loss trend."""
        cli_results = build_cli_results(
            {
                "history": [
                    {"step": 1, "loss": 3.0},
                    {"step": 2, "loss": 2.5},
                    {"step": 3, "loss": 2.0},
                ],
                "initial_equivalence": {
                    "logits": {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0},
                    "next_token_match": True,
                    "hidden_states": [{"layer_index": 0, "rel_l2": 0.0}],
                    "module_outputs": {"lm_head": {"rel_l2": 0.0}},
                },
                "final_equivalence": {
                    "logits": {"max_abs": 0.1, "mean_abs": 0.01, "rel_l2": 0.001},
                    "next_token_match": True,
                    "hidden_states": [{"layer_index": 0, "rel_l2": 0.001}],
                    "module_outputs": {"lm_head": {"rel_l2": 0.001}},
                },
                "evaluation_history": [
                    {
                        "step": 1,
                        "float_equivalence": {
                            "logits": {"max_abs": 0.2, "mean_abs": 0.02, "rel_l2": 0.002},
                            "next_token_match": True,
                            "hidden_states": [{"layer_index": 0, "rel_l2": 0.002}],
                            "module_outputs": {"lm_head": {"rel_l2": 0.002}},
                        },
                    }
                ],
                "rotation_state": {"R1": torch.eye(2)},
                "rotation_summary": {"R1": {"orthogonality_error": 0.0}},
            }
        )

        self.assertNotIn("rotation_state", cli_results)
        self.assertEqual(cli_results["history_summary"]["num_steps"], 3)
        self.assertEqual(cli_results["history_summary"]["best_loss"], 2.0)
        self.assertTrue(cli_results["history_summary"]["monotonic_nonincreasing"])
        self.assertNotIn("hidden_states", cli_results["final_equivalence"])
        self.assertNotIn("module_outputs", cli_results["evaluation_history"][0]["float_equivalence"])

    @patch("src.train_runtime_rotation.load_model_and_tokenizer")
    def test_runtime_rotation_training_runs_and_saves_checkpoint(self, mock_loader) -> None:
        """A tiny synthetic training run should produce history, eval metrics, and a saved checkpoint."""
        mock_loader.return_value = (_FakeLlamaForCausalLM(), _FakeTokenizer())

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = pathlib.Path(temp_dir) / "rotations.pt"
            results = run_runtime_rotation_training(
                RuntimeRotationTrainingConfig(
                    model_name="fake/runtime-llama",
                    rotate_mode="random",
                    r2_mode="random",
                    calibration_texts=("abcd", "bcde", "cdef", "defg"),
                    batch_size=2,
                    num_steps=3,
                    learning_rate=0.1,
                    eval_every=1,
                    prepare_model=False,
                    save_rotation_path=str(save_path),
                )
            )

            self.assertEqual(len(results["history"]), 3)
            self.assertEqual(len(results["evaluation_history"]), 3)
            self.assertIn("runtime_rotation_parameters.R1", results["trainable_parameter_names"])
            self.assertTrue(save_path.exists())
            saved_state = torch.load(save_path, weights_only=True)
            self.assertIn("R1", saved_state)
            self.assertIn("model.layers.0.self_attn.R2", saved_state["layers"])
            self.assertIn("final_equivalence", results)


if __name__ == "__main__":
    unittest.main()
