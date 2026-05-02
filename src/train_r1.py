import argparse
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import wandb

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
from src.rotation_losses import setpoint_repulsion_loss, simulated_quant_error
from src.runtime_rotation import (
    RotationParameters,
    RuntimeRotatedLinear,
    enable_runtime_attention_rotations,
)
from src.wandb_config import WANDB_ENTITY, WANDB_PROJECT, WANDB_MODE


@dataclass
class TrainR1Config:
    """Knobs for a rotation training run with a set-point repulsion auxiliary loss."""

    model_name: str = DEFAULT_MODEL_NAME
    torch_dtype: Optional[torch.dtype] = torch.float32
    init_mode: str = "identity"
    seed: int = 0

    train_r2: bool = True
    r2_mode: Optional[str] = None
    r2_seed_offset: int = 1

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

    quant_lambda: float = 1.0
    quant_num_bits: int = 4

    log_every: int = 1
    use_wandb: bool = True
    checkpoint_path: Optional[str] = None


def _freeze_non_rotation_params(
    model: torch.nn.Module,
    runtime_params: RotationParameters,
    train_r2: bool,
) -> None:
    trainable_ids = {id(runtime_params.R1)}
    if train_r2:
        trainable_ids.update(id(p) for p in runtime_params.layer_R2.values())
    for param in model.parameters():
        param.requires_grad = id(param) in trainable_ids


def _collect_effective_weights(
    model: torch.nn.Module,
    target_suffixes: Optional[Sequence[str]] = None,
) -> List[Tuple[str, torch.Tensor]]:
    suffix_tuple = tuple(target_suffixes) if target_suffixes else None
    result = []
    for name, module in model.named_modules():
        if not isinstance(module, RuntimeRotatedLinear):
            continue
        if suffix_tuple is not None and not name.endswith(suffix_tuple):
            continue
        result.append((name, module.effective_weight()))
    return result


def build_training_batches(
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    device: str,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """Yield one random text per step so the gradient varies across steps."""
    import random
    # Pre-encode each text individually so we can sample one per step.
    encoded_texts = []
    for text in texts:
        enc = build_inputs(tokenizer, texts=[text], device=device, max_length=max_length)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)
        encoded_texts.append((input_ids, labels, attention_mask))
    while True:
        yield random.choice(encoded_texts)


def train_r1(config: TrainR1Config) -> dict:
    """Train R1 (and optionally per-layer R2) with a set-point repulsion auxiliary loss."""
    if config.use_wandb:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            mode=WANDB_MODE,
            name=f"train_rotations_{config.init_mode}_r2={config.train_r2}",
            config={
                "model_name": config.model_name,
                "init_mode": config.init_mode,
                "train_r2": config.train_r2,
                "r2_mode": config.r2_mode or config.init_mode,
                "r2_seed_offset": config.r2_seed_offset,
                "lr": config.lr,
                "momentum": config.momentum,
                "num_steps": config.num_steps,
                "aux_lambda": config.aux_lambda,
                "aux_set_points": list(config.aux_set_points),
                "aux_kind": config.aux_kind,
                "aux_target_suffixes": list(config.aux_target_suffixes),
                "quant_lambda": config.quant_lambda,
                "quant_num_bits": config.quant_num_bits,
                "seed": config.seed,
            },
        )

    device = get_default_device()
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model_name,
        device=device,
        torch_dtype=config.torch_dtype,
    )
    if not is_llama_like_model(model):
        raise ValueError(f"{config.model_name} does not expose a LLaMA-style layout.")

    prepare_model_for_rotation(model)
    enable_runtime_attention_rotations(
        model,
        rotate_mode=config.init_mode,
        r2_mode=config.r2_mode,
        seed=config.seed,
        r2_seed_offset=config.r2_seed_offset,
    )
    runtime_params = model.runtime_rotation_parameters
    _freeze_non_rotation_params(model, runtime_params, config.train_r2)
    model.train()

    params_to_train = [runtime_params.R1]
    if config.train_r2:
        params_to_train += list(runtime_params.layer_R2.values())

    optimizer = SGDG(
        params_to_train,
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

        effective_weights = [
            w for _, w in _collect_effective_weights(model, config.aux_target_suffixes)
        ]
        analog_loss = setpoint_repulsion_loss(
            effective_weights,
            set_points=config.aux_set_points,
            sigma=config.aux_sigma,
            kind=config.aux_kind,
            margin=config.aux_margin,
        )
        quant_loss = simulated_quant_error(effective_weights, num_bits=config.quant_num_bits)
        total_loss = config.aux_lambda * analog_loss + config.quant_lambda * quant_loss

        # Monitor LM loss without including it in the gradient — a correct rotation
        # is transparent to the model, so task_loss should stay near its baseline.
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            task_loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        r1_grad_norm = float(runtime_params.R1.grad.norm()) if runtime_params.R1.grad is not None else float("nan")
        eye_r1 = torch.eye(runtime_params.R1.shape[0], device=runtime_params.R1.device, dtype=runtime_params.R1.dtype)
        r1_dist = float((runtime_params.R1.detach() - eye_r1).norm())

        record = {
            "step": step,
            "task_loss": float(task_loss),
            "total_loss": float(total_loss.detach()),
            "analog_loss": float(analog_loss.detach()),
            "quant_loss": float(quant_loss.detach()),
            "r1_grad_norm": r1_grad_norm,
            "r1_dist": r1_dist,
        }

        if config.train_r2:
            r2_params = list(runtime_params.layer_R2.values())
            r2_grad_norms = [float(p.grad.norm()) for p in r2_params if p.grad is not None]
            head_dim = r2_params[0].shape[0]
            eye_r2 = torch.eye(head_dim, device=r2_params[0].device, dtype=r2_params[0].dtype)
            r2_dists = [float((p.detach() - eye_r2).norm()) for p in r2_params]
            record["r2_grad_norm_mean"] = float(torch.tensor(r2_grad_norms).mean()) if r2_grad_norms else float("nan")
            record["r2_dist_mean"] = float(torch.tensor(r2_dists).mean())

        history.append(record)
        if config.use_wandb:
            wandb.log(record)
        if step % config.log_every == 0:
            r2_info = (
                f"  r2_grad={record['r2_grad_norm_mean']:.3e}  |R2-I|={record['r2_dist_mean']:.3e}"
                if config.train_r2 else ""
            )
            print(
                f"step {step:4d}  total={record['total_loss']:.4f}  "
                f"quant={record['quant_loss']:.4f}  analog={record['analog_loss']:.4f}  "
                f"task={record['task_loss']:.4f} (monitor)  "
                f"r1_grad={r1_grad_norm:.3e}  |R1-I|={r1_dist:.3e}{r2_info}"
            )

    if config.use_wandb:
        wandb.finish()

    result = {
        "R1": runtime_params.R1.detach().cpu(),
        "R2": {k: v.detach().cpu() for k, v in runtime_params.layer_R2.items()} if config.train_r2 else {},
        "history": history,
        "final_analog_loss": history[-1]["analog_loss"] if history else None,
        "final_task_loss": history[-1]["task_loss"] if history else None,
    }

    if config.checkpoint_path:
        os.makedirs(os.path.dirname(os.path.abspath(config.checkpoint_path)), exist_ok=True)
        torch.save({"R1": result["R1"], "R2": result["R2"]}, config.checkpoint_path)
        print(f"Saved checkpoint to {config.checkpoint_path}")

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train R1 and R2 with a set-point repulsion auxiliary loss.")
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
    parser.add_argument("--no-train-r2", action="store_true", help="Train R1 only, skip R2.")
    parser.add_argument(
        "--r2-mode",
        default=None,
        choices=["identity", "sign_flip", "random", "hadamard", "block_hadamard", "hadamard_D"],
        help="R2 init mode. Defaults to --init-mode.",
    )
    parser.add_argument("--r2-seed-offset", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--aux-lambda", type=float, default=0.01)
    parser.add_argument("--aux-set-points", type=float, nargs="+", default=[0.0])
    parser.add_argument("--aux-sigma", type=float, default=0.05)
    parser.add_argument("--aux-kind", default="gaussian", choices=["gaussian", "hinge"])
    parser.add_argument("--aux-margin", type=float, default=0.1)
    parser.add_argument("--quant-lambda", type=float, default=1.0)
    parser.add_argument("--quant-num-bits", type=int, default=4)
    parser.add_argument("--checkpoint", default=None, help="Save final rotation matrices to this .pt path.")
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
        train_r2=not args.no_train_r2,
        r2_mode=args.r2_mode,
        r2_seed_offset=args.r2_seed_offset,
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
        quant_lambda=args.quant_lambda,
        quant_num_bits=args.quant_num_bits,
        checkpoint_path=args.checkpoint,
        log_every=args.log_every,
        use_wandb=not args.no_wandb,
    )
    result = train_r1(config)
    print(f"final analog_loss={result['final_analog_loss']:.4f}  task_loss={result['final_task_loss']:.4f}")


if __name__ == "__main__":
    main()
