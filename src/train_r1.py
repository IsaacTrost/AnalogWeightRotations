import argparse
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import torch

from src.llama_model import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEXTS,
    TORCH_DTYPE_CHOICES,
    build_inputs,
    get_default_device,
    is_llama_like_model,
    load_model_and_tokenizer,
    resolve_torch_dtype,
)
from src.llama_prepare import prepare_model_for_rotation
from src.optimizer import SGDG
from src.rotation_losses import setpoint_repulsion_loss
from src.trainable_rotation import (
    collect_effective_weights,
    freeze_non_rotation_params,
    install_trainable_rotation,
)


@dataclass
class TrainR1Config:
    """Knobs for an R1-only training run with a set-point repulsion auxiliary loss."""

    model_name: str = DEFAULT_MODEL_NAME
    torch_dtype: Optional[torch.dtype] = torch.float32
    init_mode: str = "identity"
    seed: int = 0

    lr: float = 1e-3
    momentum: float = 0.9
    num_steps: int = 50
    max_length: int = 128
    texts: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_TEXTS))

    aux_lambda: float = 0.01
    aux_set_points: Sequence[float] = (0.0,)
    aux_sigma: float = 0.05
    aux_kind: str = "gaussian"
    aux_margin: float = 0.1
    aux_target_suffixes: Sequence[str] = ("down_proj",)

    log_every: int = 1


def build_training_batches(
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    device: str,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """Yield (input_ids, labels, attention_mask) by repeating the calibration texts.

    Labels shift-left is handled by the HF CausalLM loss; we pass input_ids as
    labels so the model computes next-token loss internally, then mask out
    padding positions with -100 so they do not contribute to the loss.
    """
    encoded = build_inputs(tokenizer, texts=texts, device=device, max_length=max_length)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    labels = input_ids.clone()
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    while True:
        yield input_ids, labels, attention_mask


def train_r1(config: TrainR1Config) -> dict:
    """Run the R1-only training loop and return the final metrics."""
    device = get_default_device()
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model_name,
        device=device,
        torch_dtype=config.torch_dtype,
    )
    if not is_llama_like_model(model):
        raise ValueError(f"{config.model_name} does not expose a LLaMA-style layout.")

    prepare_model_for_rotation(model)
    rotation = install_trainable_rotation(
        model,
        init_mode=config.init_mode,
        seed=config.seed,
        dtype=torch.float32,
    )
    freeze_non_rotation_params(model, rotation)
    model.train()

    optimizer = SGDG(
        [rotation.R],
        lr=config.lr,
        momentum=config.momentum,
        stiefel=True,
    )

    batches = build_training_batches(
        tokenizer,
        texts=list(config.texts),
        max_length=config.max_length,
        device=device,
    )

    history = []
    for step in range(config.num_steps):
        input_ids, labels, attention_mask = next(batches)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        task_loss = outputs.loss

        effective_weights = [
            w for _, w in collect_effective_weights(model, config.aux_target_suffixes)
        ]
        aux_loss = setpoint_repulsion_loss(
            effective_weights,
            set_points=config.aux_set_points,
            sigma=config.aux_sigma,
            kind=config.aux_kind,
            margin=config.aux_margin,
        )

        total_loss = task_loss + config.aux_lambda * aux_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        record = {
            "step": step,
            "task_loss": float(task_loss.detach()),
            "aux_loss": float(aux_loss.detach()),
            "total_loss": float(total_loss.detach()),
        }
        history.append(record)
        if step % config.log_every == 0:
            print(
                f"step {step:4d}  task={record['task_loss']:.4f}  "
                f"aux={record['aux_loss']:.4f}  total={record['total_loss']:.4f}"
            )

    return {
        "rotation": rotation.R.detach().cpu(),
        "history": history,
        "final_task_loss": history[-1]["task_loss"] if history else None,
        "final_aux_loss": history[-1]["aux_loss"] if history else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train R1 with an auxiliary set-point repulsion loss.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["auto", *TORCH_DTYPE_CHOICES.keys()],
    )
    parser.add_argument(
        "--init-mode",
        default="identity",
        choices=["identity", "sign_flip", "random", "hadamard", "block_hadamard", "hadamard_D"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--aux-lambda", type=float, default=0.01)
    parser.add_argument("--aux-set-points", type=float, nargs="+", default=[0.0])
    parser.add_argument("--aux-sigma", type=float, default=0.05)
    parser.add_argument("--aux-kind", default="gaussian", choices=["gaussian", "hinge"])
    parser.add_argument("--aux-margin", type=float, default=0.1)
    parser.add_argument(
        "--aux-target-suffixes",
        nargs="+",
        default=["down_proj"],
        help="Module name suffixes whose effective weights feed the auxiliary loss.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = TrainR1Config(
        model_name=args.model_name,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        init_mode=args.init_mode,
        seed=args.seed,
        lr=args.lr,
        momentum=args.momentum,
        num_steps=args.num_steps,
        max_length=args.max_length,
        aux_lambda=args.aux_lambda,
        aux_set_points=tuple(args.aux_set_points),
        aux_sigma=args.aux_sigma,
        aux_kind=args.aux_kind,
        aux_margin=args.aux_margin,
        aux_target_suffixes=tuple(args.aux_target_suffixes),
        log_every=args.log_every,
    )
    result = train_r1(config)
    print(f"final task_loss={result['final_task_loss']:.4f}  aux_loss={result['final_aux_loss']:.4f}")


if __name__ == "__main__":
    main()
