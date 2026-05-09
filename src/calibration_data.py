import json
import pathlib
from typing import Optional, Sequence

import torch

from src.llama_model import DEFAULT_TEXTS, get_default_device


def load_calibration_texts(
    texts: Optional[Sequence[str]] = None,
    data_path: Optional[str] = None,
) -> list[str]:
    """Resolve calibration texts from inline input or a simple file-backed dataset."""
    if texts is not None and data_path is not None:
        raise ValueError("Provide either inline calibration texts or a calibration data path, not both.")

    if texts is not None:
        resolved = [text for text in texts if text]
        if not resolved:
            raise ValueError("Calibration texts cannot be empty.")
        return resolved

    if data_path is None:
        return list(DEFAULT_TEXTS)

    path = pathlib.Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration data path does not exist: {data_path}")

    if path.suffix == ".jsonl":
        resolved = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, str):
                resolved.append(record)
            elif isinstance(record, dict) and isinstance(record.get("text"), str):
                resolved.append(record["text"])
            else:
                raise ValueError("JSONL calibration rows must be strings or objects with a `text` field.")
        if not resolved:
            raise ValueError("Calibration JSONL file did not contain any usable text rows.")
        return resolved

    resolved = [line for line in path.read_text().splitlines() if line.strip()]
    if not resolved:
        raise ValueError("Calibration text file did not contain any non-empty lines.")
    return resolved


def build_calibration_batches(
    tokenizer,
    texts: Optional[Sequence[str]] = None,
    data_path: Optional[str] = None,
    batch_size: int = 1,
    device: Optional[str] = None,
    max_length: int = 128,
) -> list[dict[str, torch.Tensor]]:
    """Tokenize calibration text into deterministic minibatches with next-token labels."""
    if batch_size <= 0:
        raise ValueError(f"Calibration batch size must be positive, got {batch_size}.")

    resolved_texts = load_calibration_texts(texts=texts, data_path=data_path)
    target_device = device or get_default_device()
    batches = []

    for start in range(0, len(resolved_texts), batch_size):
        batch_texts = resolved_texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batch = {name: tensor.to(target_device) for name, tensor in encoded.items()}
        labels = batch["input_ids"].clone()

        # Padding positions should not contribute to the causal-LM loss.
        if "attention_mask" in batch:
            labels = labels.masked_fill(batch["attention_mask"] == 0, -100)

        batch["labels"] = labels
        batches.append(batch)

    return batches
