#!/usr/bin/env python3
"""
train_rotation_first_order.py

First-order autograd training of an orthogonal rotation R against the
torch-backed aihwkit inference tile (TorchInferenceRPUConfigIRDropT).

Compared to train_rotation_matrix.py (SPSA on the C++ tile):
  - Real first-order gradients via autograd through the analog forward
  - SGDG with Cayley retraction on the Stiefel manifold (matches train_analog.py)
  - Targets the torch tile so W stays in the autograd graph
  - Covers IR drop + DAC/ADC quantization + additive weight noise (configurable)
  - Does NOT support PCM programming noise (program_analog_weights() is non-
    differentiable and the torch IR-drop tile does not model it). For full_pcm,
    keep using train_rotation_matrix.py.

Mechanism
---------
The tile is built once. At every step we form W_rot = W @ R.T and inject it
directly into tile._parameters["weight"] (same trick as src/train_analog.py).
This bypasses set_weights(), which would otherwise sever the autograd graph,
and lets dL/dR flow back through both the analog forward and the float ideal.

Loss matches train_rotation_matrix.py:
    rel_error = ||y_analog - y_ideal||_F / ||y_ideal||_F
"""

import argparse
import gc
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

# Repo layout: import from sibling 1layertests/explore_rotations and ../src
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explore_rotations import (
    HF_CACHE_DIR,
    make_analog_layer,
    eval_analog,
    make_identity, make_block_hadamard, make_rand_orth,
)
from src.optimizer import SGDG
from src.train_analog import _load_weight_into_tile

from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# TinyLlama loader with RMSNorm absorption + WikiText calibration split
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama_v1.1"

_PROJ_TO_NORM_ATTR = {
    # MLP input projections sit after post_attention_layernorm
    "up_proj":   ("mlp",       "post_attention_layernorm"),
    "gate_proj": ("mlp",       "post_attention_layernorm"),
    # Attention input projections sit after input_layernorm
    "q_proj":    ("self_attn", "input_layernorm"),
    "k_proj":    ("self_attn", "input_layernorm"),
    "v_proj":    ("self_attn", "input_layernorm"),
}


def _load_wikitext_text(dataset_name: str, split: str) -> str:
    """Same loader pattern as src/eval_analog_perplexity.py."""
    from datasets import load_dataset
    if dataset_name == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    elif dataset_name == "wikitext-103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return "\n\n".join(t for t in (ex["text"] for ex in ds) if t.strip())


def _capture_activations(model, target, in_dim, tokenizer, text,
                         n_tokens, seq_len, seq_batch, device, label):
    """
    Tokenize `text`, take the first `n_tokens`, chunk into sequences of
    `seq_len` tokens, run them through `model` in batches of `seq_batch`,
    and capture every value the hook on `target` sees.

    Returns x: (samples, in_dim) on CPU, float32. Samples are token-row
    activations from positions 0..n_tokens-1 (excluding any that were
    truncated by chunking — i.e., the last partial chunk is dropped).
    """
    saved_max = tokenizer.model_max_length
    tokenizer.model_max_length = 10 ** 9
    try:
        all_ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    finally:
        tokenizer.model_max_length = saved_max

    ids = torch.tensor(all_ids[:n_tokens], dtype=torch.long)
    n_full = (ids.shape[0] // seq_len) * seq_len
    if n_full == 0:
        raise RuntimeError(f"{label}: not enough tokens ({ids.shape[0]}) for one "
                           f"sequence of {seq_len}.")
    if n_full < ids.shape[0]:
        print(f"  {label}: dropping {ids.shape[0] - n_full} trailing tokens "
              f"to fit {seq_len}-token chunks")
    chunks = ids[:n_full].view(-1, seq_len)                   # (n_seq, seq_len)

    captured = []
    def hook(_mod, inp, _out):
        captured.append(inp[0].detach().reshape(-1, in_dim).cpu().float())
    handle = target.register_forward_hook(hook)

    with torch.no_grad():
        for start in range(0, chunks.shape[0], seq_batch):
            batch_ids = chunks[start:start + seq_batch].to(device)
            model(input_ids=batch_ids)
    handle.remove()

    x = torch.cat(captured, dim=0).float()                    # (n_full, in_dim)
    print(f"  {label}: captured {x.shape[0]} activation rows from "
          f"{chunks.shape[0]} sequences x {seq_len} tokens  "
          f"mean={x.mean():.3f} std={x.std():.3f}")
    return x


def load_llama_layer_and_inputs(
    model_id=DEFAULT_MODEL_ID,
    layer_idx=0,
    proj="up_proj",
    dataset="wikitext-2",
    n_train_tokens=16384,
    n_eval_tokens=4096,
    seq_len=256,
    seq_batch=4,
    absorb_rmsnorm=True,
):
    """
    Load a LLaMA-style model and capture post-RMSNorm activations entering
    the chosen input projection from disjoint train/eval text splits.

    Calibration source: WikiText (raw). The `train` split feeds activations
    into the rotation training loop; the `validation` split feeds the held-out
    evaluation. The two splits are completely disjoint by construction, so the
    eval number reflects generalization, not training-set fit.

    Gamma absorption (if absorb_rmsnorm=True) is applied identically to both
    train and eval activations and to the weight matrix, so all three sides
    of the comparison live in the same SpinQuant-compatible basis.

    Returns (W, b, x_train, x_eval):
      - W       : (out, in) with gamma folded in if absorb_rmsnorm.
      - b       : bias if any (LLaMA projections don't carry one) -> None.
      - x_train : (n_train_rows, in) activations from the train split.
      - x_eval  : (n_eval_rows,  in) activations from the validation split.
    """
    if proj not in _PROJ_TO_NORM_ATTR:
        raise ValueError(
            f"{proj!r} is not preceded by an RMSNorm. Pick an input projection "
            f"(one of {list(_PROJ_TO_NORM_ATTR)}); down_proj/o_proj sit after "
            "non-norm intermediates and have nothing to absorb."
        )

    print(f"\nLoading {model_id} ...")
    model = AutoModel.from_pretrained(
        model_id, cache_dir=HF_CACHE_DIR, torch_dtype=torch.float32
    )
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    capture_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(capture_device)

    block = model.layers[layer_idx]
    submodule_attr, norm_attr = _PROJ_TO_NORM_ATTR[proj]
    target = getattr(getattr(block, submodule_attr), proj)
    norm   = getattr(block, norm_attr)

    W = target.weight.detach().float().cpu().clone()          # (out, in)
    b = (target.bias.detach().float().cpu().clone()
         if target.bias is not None else None)
    in_dim = W.shape[1]
    gamma = norm.weight.detach().float().cpu().clone()        # (in,)
    print(f"  {proj} weight: {tuple(W.shape)}  RMSNorm gamma: ({gamma.shape[0]},)  "
          f"min={gamma.min():.3f}  max={gamma.max():.3f}  mean={gamma.mean():.3f}")

    print(f"\nCalibration: {dataset}")
    print(f"  train: {n_train_tokens} tokens from train split")
    print(f"  eval:  {n_eval_tokens} tokens from validation split (disjoint)")

    train_text = _load_wikitext_text(dataset, "train")
    x_train = _capture_activations(
        model, target, in_dim, tok, train_text,
        n_train_tokens, seq_len, seq_batch, capture_device, label="train",
    )

    eval_text = _load_wikitext_text(dataset, "validation")
    x_eval = _capture_activations(
        model, target, in_dim, tok, eval_text,
        n_eval_tokens, seq_len, seq_batch, capture_device, label="eval",
    )

    # Sanity: float forward must match before/after absorption (same identity
    # holds on both pools, but we only need to check one).
    y_before = x_eval @ W.T

    if absorb_rmsnorm:
        if gamma.abs().min() < 1e-6:
            print(f"  WARNING: smallest |gamma| = {gamma.abs().min():.2e}, "
                  "absorption may amplify noise on those dimensions.")
        x_train = x_train / gamma.unsqueeze(0)
        x_eval  = x_eval  / gamma.unsqueeze(0)
        W = W * gamma.unsqueeze(0)
        y_after = x_eval @ W.T
        max_drift = (y_after - y_before).abs().max().item()
        print(f"  Absorbed RMSNorm gamma into W. "
              f"Max forward drift on eval pool: {max_drift:.2e} (should be ~0)")
    else:
        print("  Skipping RMSNorm absorption — trained R will not be SpinQuant-compatible.")

    del model, tok, target, norm, block
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return W, b, x_train, x_eval

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Hardware config (torch tile only — required for autograd through the weight)
# ---------------------------------------------------------------------------

def build_torch_rpu_config(ir_drop, ir_drop_segments, ir_drop_v_read,
                           inp_bits, out_bits, w_noise):
    """
    TorchInferenceRPUConfigIRDropT with knobs for IR drop, ADC/DAC quant, and
    constant additive weight noise. inp_bits / out_bits = 0 disables quant on
    that side; w_noise = 0 disables weight noise.
    """
    from aihwkit.simulator.configs import TorchInferenceRPUConfigIRDropT
    from aihwkit.simulator.parameters.enums import (
        BoundManagementType, NoiseManagementType, WeightNoiseType,
    )
    rpu = TorchInferenceRPUConfigIRDropT()
    rpu.forward.ir_drop          = ir_drop
    rpu.forward.ir_drop_segments = ir_drop_segments
    rpu.forward.ir_drop_v_read   = ir_drop_v_read
    rpu.forward.inp_res          = (2 ** inp_bits - 2) if inp_bits > 0 else -1
    rpu.forward.out_res          = (2 ** out_bits - 2) if out_bits > 0 else -1
    rpu.forward.out_bound        = 1.0
    rpu.forward.w_noise          = w_noise
    rpu.forward.w_noise_type     = (WeightNoiseType.ADDITIVE_CONSTANT
                                    if w_noise > 0 else WeightNoiseType.NONE)
    rpu.forward.bound_management = BoundManagementType.NONE
    rpu.forward.noise_management = NoiseManagementType.ABS_MAX
    return rpu


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rotation(W, b, x_train, rpu_config, *, init, num_steps, lr, momentum,
                   batch, log_every, sample_seed=0):
    """
    Train R via first-order autograd through the torch aihwkit tile.

    Each step draws `batch` random rows (without replacement) from x_train,
    so over many steps R sees the whole calibration pool, not just the first
    `batch` rows.
    """
    n = W.shape[1]

    if init == "identity":
        R0 = make_identity(n)
    elif init == "hadamard":
        R0 = make_block_hadamard(n)
    elif init == "random":
        R0 = make_rand_orth(n, seed=0)
    else:
        raise ValueError(f"Unknown init {init!r}")

    R = R0.to(device=DEVICE, dtype=torch.float32).clone().detach().requires_grad_(True)
    opt = SGDG([R], lr=lr, momentum=momentum, stiefel=True)

    W       = W.to(device=DEVICE, dtype=torch.float32)
    b_dev   = b.to(device=DEVICE, dtype=torch.float32) if b is not None else None
    x_pool  = x_train.to(device=DEVICE, dtype=torch.float32)
    pool_n  = x_pool.shape[0]
    actual_batch = min(batch, pool_n)

    rng = torch.Generator(device=DEVICE).manual_seed(sample_seed)

    # Build the analog tile once; weight is overwritten in-place every step.
    layer = make_analog_layer(W, b_dev, rpu_config)

    print(f"\nTraining R ({init} init, {num_steps} steps, lr={lr}, "
          f"batch={actual_batch} of {pool_n} train rows) on torch tile ...")

    for step in range(1, num_steps + 1):
        opt.zero_grad(set_to_none=True)

        idx = torch.randperm(pool_n, generator=rng, device=DEVICE)[:actual_batch]
        x   = x_pool[idx]

        x_rot   = x @ R.T
        W_rot   = W @ R.T
        y_ideal = F.linear(x_rot, W_rot, b_dev)

        _load_weight_into_tile(layer, W_rot)
        y_analog = layer(x_rot)

        denom = y_ideal.detach().pow(2).mean() + 1e-12
        loss  = (y_analog - y_ideal).pow(2).mean() / denom

        loss.backward()
        opt.step()

        if step == 1 or step % log_every == 0:
            with torch.no_grad():
                orth_err  = (R @ R.T - torch.eye(n, device=DEVICE)).norm().item()
                grad_norm = R.grad.norm().item() if R.grad is not None else float("nan")
            print(f"  step {step:4d}/{num_steps}  "
                  f"loss={loss.item():.6f}  "
                  f"|grad|={grad_norm:.3e}  "
                  f"orth_err={orth_err:.2e}")

    R_trained = R.detach().cpu()
    final_orth = (R_trained @ R_trained.T - torch.eye(n)).norm().item()
    print(f"\n  Final ||R R^T - I||_F = {final_orth:.2e}")
    return R_trained


# ---------------------------------------------------------------------------
# Evaluation (same tile we trained on — zero modeling gap)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_rotation(W, b, x_eval, R, rpu_config, n_trials, eval_batch=32):
    """
    Evaluate R on the held-out activation pool. The same eval batch is reused
    across n_trials hardware noise realisations, so any spread reflects only
    the analog noise model, not activation variance.
    """
    R     = R.to(device=DEVICE, dtype=torch.float32)
    W     = W.to(device=DEVICE, dtype=torch.float32)
    b_dev = b.to(device=DEVICE, dtype=torch.float32) if b is not None else None
    x     = x_eval.to(device=DEVICE, dtype=torch.float32)[:eval_batch]

    x_rot   = x @ R.T
    W_rot   = W @ R.T
    y_ideal = F.linear(x_rot, W_rot, b_dev)

    metrics = {"rel_error": [], "snr_db": [], "cos_sim": []}
    for _ in range(n_trials):
        layer = make_analog_layer(W_rot, b_dev, rpu_config)
        m = eval_analog(layer, x_rot, y_ideal)
        for k in metrics:
            metrics[k].append(m[k])
        del layer
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="First-order training of a rotation R against the torch "
                    "aihwkit tile (IR drop + quantization + weight noise).",
    )
    p.add_argument("--init", default="identity",
                   choices=["identity", "hadamard", "random"])
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--batch", type=int, default=8,
                   help="Activation samples per step. The torch IR-drop tile "
                        "allocates ~batch*in*out*segments*4B; for TinyLlama "
                        "up_proj (2048->5632, segments=4) that's ~1.5 GB at "
                        "batch=8, doubled by autograd. Bump up if you have VRAM.")

    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help="HuggingFace model id (default: TinyLlama 1.1B Chat).")
    p.add_argument("--layer-idx", type=int, default=0,
                   help="Transformer block index to extract from.")
    p.add_argument("--proj", default="up_proj",
                   choices=list(_PROJ_TO_NORM_ATTR.keys()),
                   help="Which input projection to use. Must sit after an "
                        "RMSNorm so gamma absorption is well-defined.")
    p.add_argument("--no-absorb-rmsnorm", dest="absorb_rmsnorm",
                   action="store_false",
                   help="Skip RMSNorm gamma absorption. The trained R will "
                        "no longer be SpinQuant-compatible — use only for "
                        "ablations.")
    p.set_defaults(absorb_rmsnorm=True)

    p.add_argument("--ir-drop", type=float, default=1.0)
    p.add_argument("--ir-drop-segments", type=int, default=4)
    p.add_argument("--ir-drop-v-read", type=float, default=0.4)
    p.add_argument("--inp-bits", type=int, default=8,
                   help="Input DAC bits (0 disables input quantization).")
    p.add_argument("--out-bits", type=int, default=8,
                   help="Output ADC bits (0 disables output quantization).")
    p.add_argument("--w-noise", type=float, default=0.0,
                   help="Additive constant weight noise sigma (0 disables).")

    p.add_argument("--dataset", default="wikitext-2",
                   choices=["wikitext-2", "wikitext-103"],
                   help="Calibration source. Train activations come from the "
                        "train split, eval from the validation split.")
    p.add_argument("--calib-train-tokens", type=int, default=16384,
                   help="Tokens to draw from the train split for calibration.")
    p.add_argument("--calib-eval-tokens", type=int, default=4096,
                   help="Tokens to draw from the validation split for eval.")
    p.add_argument("--calib-seq-len", type=int, default=256,
                   help="Sequence length used when chunking calibration text.")
    p.add_argument("--calib-seq-batch", type=int, default=4,
                   help="Sequences batched per forward pass during capture.")
    p.add_argument("--eval-batch", type=int, default=32,
                   help="Eval activation rows per hardware-noise trial.")

    p.add_argument("--n-eval-trials", type=int, default=10)
    p.add_argument("--save", default="results/trained_R_first_order.pt")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--sample-seed", type=int, default=0,
                   help="Seed for per-step random row sampling from x_train.")
    return p


def main():
    args = build_arg_parser().parse_args()
    os.makedirs("results", exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"HF cache: {HF_CACHE_DIR}")
    W, b, x_train, x_eval = load_llama_layer_and_inputs(
        model_id=args.model_id,
        layer_idx=args.layer_idx,
        proj=args.proj,
        dataset=args.dataset,
        n_train_tokens=args.calib_train_tokens,
        n_eval_tokens=args.calib_eval_tokens,
        seq_len=args.calib_seq_len,
        seq_batch=args.calib_seq_batch,
        absorb_rmsnorm=args.absorb_rmsnorm,
    )
    in_dim = W.shape[1]
    print(f"Layer: W={tuple(W.shape)}  x_train={tuple(x_train.shape)}  "
          f"x_eval={tuple(x_eval.shape)}  bias={'yes' if b is not None else 'no'}")

    rpu_config = build_torch_rpu_config(
        ir_drop=args.ir_drop,
        ir_drop_segments=args.ir_drop_segments,
        ir_drop_v_read=args.ir_drop_v_read,
        inp_bits=args.inp_bits,
        out_bits=args.out_bits,
        w_noise=args.w_noise,
    )
    print(f"Hardware: ir_drop={args.ir_drop}, segments={args.ir_drop_segments}, "
          f"inp_bits={args.inp_bits}, out_bits={args.out_bits}, "
          f"w_noise={args.w_noise}")

    R_trained = train_rotation(
        W, b, x_train, rpu_config,
        init=args.init,
        num_steps=args.num_steps,
        lr=args.lr,
        momentum=args.momentum,
        batch=args.batch,
        log_every=args.log_every,
        sample_seed=args.sample_seed,
    )
    torch.save(R_trained, args.save)
    print(f"\nSaved trained R to {args.save}")

    rotations = {
        "identity":  make_identity(in_dim),
        "hadamard":  make_block_hadamard(in_dim),
        "rand_orth": make_rand_orth(in_dim, seed=7),
        "trained_R": R_trained,
    }

    print(f"\nEvaluating on held-out wikitext activations "
          f"({args.n_eval_trials} hardware trials each, eval_batch={args.eval_batch}):")
    results = {}
    for name, R in rotations.items():
        res = eval_rotation(W, b, x_eval, R, rpu_config, args.n_eval_trials,
                            eval_batch=args.eval_batch)
        results[name] = res
        rm, rs = res["rel_error"]
        sm, ss = res["snr_db"]
        print(f"  {name:12s}  rel_err={rm:.4f} ± {rs:.4f}   "
              f"snr={sm:+.2f} ± {ss:.2f} dB")

    print("\nImprovement vs identity:")
    base = results["identity"]["rel_error"][0]
    for name in ("hadamard", "rand_orth", "trained_R"):
        val = results[name]["rel_error"][0]
        pct = 100.0 * (base - val) / (base + 1e-12)
        print(f"  {name:12s}  {pct:+.2f}%   ({base:.4f} -> {val:.4f})")


if __name__ == "__main__":
    main()
