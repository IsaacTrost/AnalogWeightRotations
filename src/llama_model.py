from typing import Any, Iterable, List, Optional, Sequence, TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
else:
    PreTrainedModel = Any
    PreTrainedTokenizerBase = Any


DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Large language models can be rotated without changing the exact float computation.",
    "Analog hardware errors depend on how energy is distributed across rows and columns.",
    "TinyLlama is a practical small checkpoint for validating a LLaMA-shaped pipeline.",
]

TORCH_DTYPE_CHOICES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def get_default_device() -> str:
    """Pick the fastest available device while keeping CPU as a safe fallback."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    """Map a CLI-friendly dtype name to the torch dtype used for model weights."""
    if dtype_name is None or dtype_name == "auto":
        return None
    if dtype_name not in TORCH_DTYPE_CHOICES:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return TORCH_DTYPE_CHOICES[dtype_name]


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Hugging Face causal LM and normalize its tokenizer padding config."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_device = device or get_default_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    model.to(target_device)
    model.eval()
    return model, tokenizer


def build_inputs(
    tokenizer: PreTrainedTokenizerBase,
    texts: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    max_length: int = 128,
) -> dict:
    """Tokenize a small text batch for forward-pass verification."""
    batch_texts = list(texts or DEFAULT_TEXTS)
    encoded = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    target_device = device or get_default_device()
    return {name: tensor.to(target_device) for name, tensor in encoded.items()}


def get_decoder_layers(model: PreTrainedModel) -> List[torch.nn.Module]:
    """Return the decoder layers for a standard Hugging Face LLaMA layout."""
    return list(model.model.layers)


def is_llama_like_model(model: PreTrainedModel) -> bool:
    """Check the module layout required by the SpinQuant-style pipeline."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return False
    if not model.model.layers:
        return False

    first_layer = model.model.layers[0]
    required_names = [
        ("self_attn", "q_proj"),
        ("self_attn", "k_proj"),
        ("self_attn", "v_proj"),
        ("self_attn", "o_proj"),
        ("mlp", "up_proj"),
        ("mlp", "gate_proj"),
        ("mlp", "down_proj"),
    ]
    return all(hasattr(getattr(first_layer, parent, object()), child) for parent, child in required_names)


def get_module_names_for_verification(model: PreTrainedModel) -> Iterable[str]:
    """Select a representative set of modules for hook-based equivalence checks."""
    layers = get_decoder_layers(model)
    if not layers:
        return []

    last_idx = len(layers) - 1
    return [
        "model.embed_tokens",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
        f"model.layers.{last_idx}.self_attn.o_proj",
        f"model.layers.{last_idx}.mlp.down_proj",
        "lm_head",
    ]
