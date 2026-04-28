#!/usr/bin/env python3
"""
explore_rotations.py

Explores whether applying orthogonal rotations to the input of an analog
linear layer — with the corresponding inverse rotation applied to the
weight matrix — can reduce the output error caused by hardware non-idealities.

The key identity:  y = W x  =  (W R^T)(R x)  for any orthogonal R.
In exact float arithmetic the rotation is transparent. On analog hardware,
however, R redistributes the energy across crossbar rows/columns, which
can mitigate (or worsen) IR drop, weight noise, and ADC quantization.

Layer under test
----------------
GPT-2 small (124 M), first transformer block, MLP feed-forward projection:
    c_fc : (batch, 768) -> (batch, 3072)   [W shape (3072, 768)]

The weight matrix and a batch of real input activations are extracted from
the model with a forward-hook, so statistics are realistic.

Rotations (all orthogonal, so R R^T = I and the math is exact in float)
------------------------------------------------------------------------
  identity     - no rotation (baseline)
  sign_flip    - random diagonal ±1 matrix D
  rand_orth    - random orthogonal from QR(randn(n, n))
  hadamard     - block-diagonal Hadamard H = blkdiag(H256, H256, H256)/√256
  hadamard_D   - H @ D  (QuaRot-style: combines incoherence + sign spreading)
  sorted_perm  - permutation that sorts inputs by mean |activation| (energy balancing)

Analog configs
--------------
  irdrop_only   - InferenceRPUConfig, ir_drop=1.0, all other noise disabled
  w_noise_only  - InferenceRPUConfig, weight noise σ=0.02, no IR drop
  inp_quant     - InferenceRPUConfig, 8-bit input + output quantization only
  full_pcm      - InferenceRPUConfig + PCMLikeNoiseModel + IR drop (realistic)
  adv_irdrop    - TorchInferenceRPUConfigIRDropT (time-dependent IR drop)

Metrics (averaged over N_TRIALS noise realisations)
----------------------------------------------------
  rel_error     - ||y_analog - y_ideal||_F / ||y_ideal||_F
  cos_sim       - mean per-sample cosine similarity
  snr_db        - 10 log10(||y_ideal||² / ||y_analog - y_ideal||²)
"""

import gc
import os
import sys
import datetime
import warnings
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import wandb
from wandb_config import WANDB_ENTITY, WANDB_PROJECT, WANDB_MODE

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import InferenceRPUConfig, TorchInferenceRPUConfigIRDropT
from aihwkit.simulator.parameters.enums import (
    WeightNoiseType, BoundManagementType, NoiseManagementType
)
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
N_TRIALS      = 20    # noise realisations per (config, rotation) to average over
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
HF_CACHE_DIR  = os.getenv("HF_CACHE_DIR", "/mnt/bigdisk/models")
print(f"Using device: {DEVICE}")
print(f"HF cache:     {HF_CACHE_DIR}")


def _log_mem(tag: str):
    """Print system RAM and GPU memory at a checkpoint."""
    try:
        with open("/proc/self/status") as f:
            rss = next(l for l in f if l.startswith("VmRSS"))
        rss_gb = int(rss.split()[1]) / 1e6
    except Exception:
        rss_gb = -1
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / 1e9
        gpu_res   = torch.cuda.memory_reserved()  / 1e9
        print(f"  [mem:{tag}]  RAM={rss_gb:.1f}GB  "
              f"GPU alloc={gpu_alloc:.2f}GB  reserved={gpu_res:.2f}GB", flush=True)
    else:
        print(f"  [mem:{tag}]  RAM={rss_gb:.1f}GB", flush=True)

# ---------------------------------------------------------------------------
# Layer specifications
# ---------------------------------------------------------------------------

@dataclass
class LayerSpec:
    model_id: str           # HuggingFace model ID
    display_name: str       # human-readable label for plots/logs
    slug: str               # short filesystem-safe name
    in_dim: int
    out_dim: int
    get_module: Callable    # (model) -> nn.Module  (for hooking)
    get_weights: Callable   # (model) -> (W: Tensor(out,in), b: Tensor(out)|None)


# GPT-2 uses Conv1D which stores weight as (in, out) — must transpose.
# Modern models (Llama, Qwen) use standard nn.Linear, weight is (out, in).

LAYER_SPECS = [
    LayerSpec(
        model_id="gpt2",
        display_name="GPT-2s  h[0].mlp.c_fc  768→3072  (early MLP)",
        slug="gpt2s_h0_mlp",
        in_dim=768, out_dim=3072,
        get_module=lambda m: m.h[0].mlp.c_fc,
        get_weights=lambda m: (
            m.h[0].mlp.c_fc.weight.detach().T.float(),   # Conv1D: (in,out)→(out,in)
            m.h[0].mlp.c_fc.bias.detach().float(),
        ),
    ),
    LayerSpec(
        model_id="gpt2",
        display_name="GPT-2s  h[11].mlp.c_fc  768→3072  (late MLP)",
        slug="gpt2s_h11_mlp",
        in_dim=768, out_dim=3072,
        get_module=lambda m: m.h[11].mlp.c_fc,
        get_weights=lambda m: (
            m.h[11].mlp.c_fc.weight.detach().T.float(),
            m.h[11].mlp.c_fc.bias.detach().float(),
        ),
    ),
    LayerSpec(
        model_id="HuggingFaceTB/SmolLM2-135M",   # ~270 MB download
        display_name="SmolLM2-135M  layers[0].mlp.gate  576→1536  (small/recent)",
        slug="smollm2_135m_l0_gate",
        in_dim=576, out_dim=1536,
        get_module=lambda m: m.layers[0].mlp.gate_proj,
        get_weights=lambda m: (
            m.layers[0].mlp.gate_proj.weight.detach().float(),
            None,
        ),
    ),
    LayerSpec(
        model_id="meta-llama/Llama-3.2-1B",      # ~2.5 GB  (requires HF access token)
        display_name="Llama-3.2-1B  layers[0].attn.o_proj  2048→2048  (Llama/recent)",
        slug="llama32_1b_l0_oproj",
        in_dim=2048, out_dim=2048,
        get_module=lambda m: m.layers[0].self_attn.o_proj,
        get_weights=lambda m: (
            m.layers[0].self_attn.o_proj.weight.detach().float(),
            None,
        ),
    ),
    LayerSpec(
        model_id="Qwen/Qwen2-1.5B",              # ~3.1 GB
        display_name="Qwen2-1.5B  layers[0].attn.q_proj  1536→1536  (Qwen2/recent)",
        slug="qwen2_15b_l0_qproj",
        in_dim=1536, out_dim=1536,
        get_module=lambda m: m.layers[0].self_attn.q_proj,
        get_weights=lambda m: (
            m.layers[0].self_attn.q_proj.weight.detach().float(),
            None,
        ),
    ),
]

# ---------------------------------------------------------------------------
# 1. Build rotation matrices
# ---------------------------------------------------------------------------

def make_identity(n: int) -> torch.Tensor:
    return torch.eye(n)


def make_sign_flip(n: int, seed: int = 0) -> torch.Tensor:
    """Random diagonal ±1 matrix."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    d = torch.randint(0, 2, (n,), generator=rng).float() * 2 - 1
    return torch.diag(d)


def make_rand_orth(n: int, seed: int = 0) -> torch.Tensor:
    """Random orthogonal matrix via QR decomposition."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    G = torch.randn(n, n, generator=rng)
    Q, _ = torch.linalg.qr(G)
    return Q


def make_block_hadamard(n: int) -> torch.Tensor:
    """
    Block-diagonal Hadamard for dimension n.

    For n = k * b where b is a power of 2, returns blkdiag(H_b, ..., H_b) / sqrt(b).
    For GPT-2's 768 = 3 * 256, this gives a clean 768×768 orthogonal matrix
    with entries ±1/16, similar to the construction used in QuaRot and QuIP#.
    """
    # Decompose n into blocks: find largest power-of-2 divisor
    b = 1
    while b * 2 <= n and n % (b * 2) == 0:
        b *= 2
    # b is the largest power-of-2 that evenly divides n
    k = n // b
    H_b = torch.tensor(hadamard(b), dtype=torch.float32) / (b ** 0.5)
    return torch.block_diag(*([H_b] * k))


def make_hadamard_D(n: int, seed: int = 0) -> torch.Tensor:
    """
    H @ D where H is block-diagonal Hadamard and D is random ±1 diagonal.
    This is the QuaRot-style rotation: incoherence (from H) + sign spreading (from D).
    """
    H = make_block_hadamard(n)
    D = make_sign_flip(n, seed=seed)
    return H @ D


def make_sorted_perm(n: int, ref_activations: torch.Tensor) -> torch.Tensor:
    """
    Permutation matrix that sorts input dimensions by mean absolute activation
    (descending). Places high-energy inputs into the first rows of the crossbar,
    which can reduce IR drop effects on subsequent rows.

    ref_activations : (batch, n) tensor of reference input activations.
    """
    mean_abs = ref_activations.abs().mean(dim=0)  # (n,)
    order = torch.argsort(mean_abs, descending=True)
    P = torch.zeros(n, n)
    for new_idx, old_idx in enumerate(order):
        P[new_idx, old_idx] = 1.0
    return P


# ---------------------------------------------------------------------------
# 2. Build analog RPU configs
# ---------------------------------------------------------------------------

def cfg_irdrop_only() -> InferenceRPUConfig:
    """IR drop only — all weight/input/output noise disabled."""
    cfg = InferenceRPUConfig()
    cfg.forward.ir_drop       = 1.0
    cfg.forward.w_noise       = 0.0
    cfg.forward.w_noise_type  = WeightNoiseType.NONE
    cfg.forward.inp_noise     = 0.0
    cfg.forward.out_noise     = 0.0
    cfg.forward.inp_res       = -1.0   # no input quantization
    cfg.forward.out_res       = -1.0   # no output quantization
    cfg.forward.out_bound     = -1.0
    cfg.forward.bound_management  = BoundManagementType.NONE
    cfg.forward.noise_management  = NoiseManagementType.NONE
    return cfg


def cfg_w_noise_only() -> InferenceRPUConfig:
    """Additive weight noise only — no IR drop or quantization."""
    cfg = InferenceRPUConfig()
    cfg.forward.ir_drop       = 0.0
    cfg.forward.w_noise       = 0.02
    cfg.forward.w_noise_type  = WeightNoiseType.ADDITIVE_CONSTANT
    cfg.forward.inp_noise     = 0.0
    cfg.forward.out_noise     = 0.0
    cfg.forward.inp_res       = -1.0
    cfg.forward.out_res       = -1.0
    cfg.forward.out_bound     = -1.0
    cfg.forward.bound_management  = BoundManagementType.NONE
    cfg.forward.noise_management  = NoiseManagementType.NONE
    return cfg


def cfg_inp_quant() -> InferenceRPUConfig:
    """8-bit input + output quantization only — no noise, no IR drop."""
    cfg = InferenceRPUConfig()
    cfg.forward.ir_drop       = 0.0
    cfg.forward.w_noise       = 0.0
    cfg.forward.w_noise_type  = WeightNoiseType.NONE
    cfg.forward.inp_noise     = 0.0
    cfg.forward.out_noise     = 0.0
    cfg.forward.inp_res       = 2**8 - 2   # 8-bit DAC
    cfg.forward.out_res       = 2**8 - 2   # 8-bit ADC
    cfg.forward.bound_management  = BoundManagementType.NONE
    cfg.forward.noise_management  = NoiseManagementType.NONE
    return cfg


def cfg_full_pcm() -> InferenceRPUConfig:
    """
    Realistic PCM inference: PCM-like weight noise, IR drop, and 10-bit ADC/DAC.
    Models a typical Phase-Change Memory crossbar at inference time.
    """
    cfg = InferenceRPUConfig()
    cfg.noise_model = PCMLikeNoiseModel(
        g_max=25.0,
        prog_noise_scale=1.0,
        read_noise_scale=1.0,
        drift_scale=0.0,          # no temporal drift for this comparison
        g_converter=SinglePairConductanceConverter(g_min=0.1, g_max=25.0),
    )
    cfg.forward.ir_drop           = 0.5
    cfg.forward.w_noise           = 0.0   # weight noise comes from noise_model
    cfg.forward.inp_noise         = 0.0
    cfg.forward.out_noise         = 0.0
    cfg.forward.inp_res           = 2**10 - 2
    cfg.forward.out_res           = 2**10 - 2
    cfg.forward.bound_management  = BoundManagementType.NONE
    cfg.forward.noise_management  = NoiseManagementType.NONE
    cfg.drift_compensation        = GlobalDriftCompensation()
    return cfg


def cfg_adv_irdrop() -> TorchInferenceRPUConfigIRDropT:
    """
    Advanced time-dependent IR drop (TorchInferenceRPUConfigIRDropT).
    Models resistive-line voltage drop that evolves during the PWM pulse.

    Note: segment count kept low (4) — the 4D Thevenin tensor is
    O(segments × batch × out × in), which OOMs at large layer sizes.
    """
    cfg = TorchInferenceRPUConfigIRDropT()
    cfg.forward.ir_drop           = 1.0
    cfg.forward.ir_drop_segments  = 4    # was 16; memory = segments*batch*out*in
    cfg.forward.ir_drop_v_read    = 0.4
    cfg.forward.w_noise           = 0.0
    cfg.forward.w_noise_type      = WeightNoiseType.NONE
    cfg.forward.inp_noise         = 0.0
    cfg.forward.out_noise         = 0.0
    cfg.forward.inp_res           = 2**10 - 2
    cfg.forward.out_res           = -1.0
    cfg.forward.out_bound         = -1.0
    cfg.forward.bound_management  = BoundManagementType.NONE
    cfg.forward.noise_management  = NoiseManagementType.NONE
    return cfg


# ---------------------------------------------------------------------------
# 3. Load layer weights + real activations (generic)
# ---------------------------------------------------------------------------

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models are transforming the field of natural language processing.",
    "In the beginning was the word, and the word was with God.",
    "To be or not to be, that is the question.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael. Some years ago, I thought I would sail about a little.",
    "The only way to do great work is to love what you do.",
    "In mathematics, you don't understand things, you just get used to them.",
    "Science is not only a disciple of reason but also one of romance.",
    "The universe is not only stranger than we imagine, it is stranger than we can imagine.",
    "Two roads diverged in a yellow wood, and I took the one less traveled by.",
    "Ask not what your country can do for you; ask what you can do for your country.",
    "Elementary, my dear Watson. The game is afoot.",
    "I think therefore I am. Cogito ergo sum.",
    "The medium is the message in the age of electronic communication.",
    "Language is the house of being. In its home man dwells.",
    "The unexamined life is not worth living, said Socrates.",
    "Neurons that fire together wire together during learning.",
    "Attention is all you need for sequence to sequence modelling.",
]


def load_layer_and_inputs(spec: LayerSpec, n_texts: int = 20):
    """
    Load a model, extract the target linear layer's weights, and capture
    real input activations via a forward hook.

    Returns:
        W : (out_dim, in_dim) float32 — weight matrix
        b : (out_dim,) float32 or None — bias
        x : (n_samples, in_dim) float32 — captured activations
    """
    print(f"\nLoading {spec.model_id} for layer '{spec.display_name}' ...")
    model = AutoModel.from_pretrained(spec.model_id, cache_dir=HF_CACHE_DIR)
    tok   = AutoTokenizer.from_pretrained(spec.model_id, cache_dir=HF_CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()

    W, b = spec.get_weights(model)
    module = spec.get_module(model)
    print(f"  W shape: {W.shape}  bias: {'yes' if b is not None else 'none'}")

    captured = []
    def hook(mod, inp, out):
        captured.append(inp[0].detach().reshape(-1, spec.in_dim).cpu().float())

    handle = module.register_forward_hook(hook)
    with torch.no_grad():
        for text in TEXTS[:n_texts]:
            tokens = tok(text, return_tensors="pt", truncation=True, max_length=64)
            model(**tokens)
    handle.remove()

    x = torch.cat(captured, dim=0).float()
    print(f"  Captured {x.shape[0]} activation samples  "
          f"mean={x.mean():.3f}  std={x.std():.3f}")

    # Free the model — we only need W, b, x going forward.
    # Must also del `module` since it holds a reference to a model submodule,
    # which would keep the entire model alive despite `del model`.
    del model, tok, captured, module, handle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log_mem("after model free")

    return W, b, x


# ---------------------------------------------------------------------------
# 3b. Build rotation matrices for a given input dimension
# ---------------------------------------------------------------------------

def build_rotations(in_dim: int, x: torch.Tensor) -> dict:
    """Return all rotation matrices sized for in_dim."""
    print(f"\nBuilding rotations for in_dim={in_dim} ...")
    rotations = {
        "identity":    make_identity(in_dim),
        "sign_flip":   make_sign_flip(in_dim, seed=7),
        "rand_orth":   make_rand_orth(in_dim, seed=7),
        "hadamard":    make_block_hadamard(in_dim),
        "hadamard_D":  make_hadamard_D(in_dim, seed=7),
        "sorted_perm": make_sorted_perm(in_dim, x),
    }
    for name, R in rotations.items():
        err = (R @ R.T - torch.eye(in_dim)).norm().item()
        print(f"  {name:15s}  ||R R^T - I||_F = {err:.2e}")
    return rotations


# ---------------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------------

def make_analog_layer(W_rot: torch.Tensor, b: torch.Tensor,
                      rpu_config) -> AnalogLinear:
    """
    Create an AnalogLinear layer pre-loaded with (rotated) weights.
    W_rot: (out, in), b: (out,)
    """
    out_f, in_f = W_rot.shape
    has_bias = (b is not None)
    layer = AnalogLinear(in_f, out_f, bias=has_bias, rpu_config=rpu_config)
    # set_weights expects CPU tensors
    W_cpu = W_rot.cpu()
    b_cpu = b.cpu() if has_bias else None
    for _, tile in layer.named_analog_layers():
        if has_bias:
            tile.set_weights(W_cpu, b_cpu)
        else:
            tile.set_weights(W_cpu)
    if DEVICE != "cpu":
        layer = layer.cuda()
    layer.eval()
    return layer


@torch.no_grad()
def eval_analog(layer: AnalogLinear, x_rot: torch.Tensor,
                y_ideal: torch.Tensor) -> dict:
    """
    Run a single forward pass and compute error metrics vs y_ideal.
    x_rot  : (batch, in)
    y_ideal: (batch, out)  — float reference output (no rotation applied)
    """
    y_analog = layer(x_rot)
    err = y_analog - y_ideal
    rel  = err.norm() / y_ideal.norm()
    snr  = 10 * torch.log10(y_ideal.norm()**2 / (err.norm()**2 + 1e-12))
    cos  = torch.nn.functional.cosine_similarity(y_analog, y_ideal, dim=1).mean()
    return {
        "rel_error": rel.item(),
        "snr_db":    snr.item(),
        "cos_sim":   cos.item(),
    }


def run_trials(W: torch.Tensor, b: torch.Tensor, x: torch.Tensor,
               R: torch.Tensor, rpu_config_fn, n_trials: int,
               post_init=None) -> dict:
    """
    Apply rotation R, create N_TRIALS independent analog layers (different
    noise seeds), evaluate each, and return mean ± std of metrics.

    post_init: optional callable(layer) called after layer creation,
               e.g. lambda l: l.program_analog_weights() for PCM noise.
    """
    # Cap batch size to avoid OOM on memory-hungry configs (adv_irdrop).
    # TorchInferenceRPUConfigIRDropT allocates O(segments * batch * out * in).
    MAX_BATCH = 32
    x = x[:MAX_BATCH]

    # Move to device
    W = W.to(DEVICE)
    b = b.to(DEVICE) if b is not None else None
    x = x.to(DEVICE)
    R = R.to(DEVICE)

    # Rotated inputs and weights
    x_rot = x @ R.T               # (batch, in)
    W_rot = W @ R.T               # (out, in)

    # Float ideal (unrotated — same as rotated in exact arithmetic)
    y_ideal = x @ W.T + (b.unsqueeze(0) if b is not None else 0)

    metrics = {"rel_error": [], "snr_db": [], "cos_sim": []}

    for _ in range(n_trials):
        cfg   = rpu_config_fn()
        layer = make_analog_layer(W_rot, b, cfg)
        if post_init is not None:
            post_init(layer)
        m     = eval_analog(layer, x_rot, y_ideal)
        for k in metrics:
            metrics[k].append(m[k])
        layer.cpu()  # move tile back to CPU → releases CUDA buffers immediately
        del layer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Explicitly free CUDA tensors created at the top of this function
    del W_rot, x_rot, y_ideal
    if b is not None:
        del b
    del W, x, R
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return {
        k: (float(np.mean(v)), float(np.std(v)))
        for k, v in metrics.items()
    }


# ---------------------------------------------------------------------------
# 5. Main experiment
# ---------------------------------------------------------------------------

# Analog hardware configs — shared across all layers.
# Each entry: (config_fn, post_init_fn or None)
CONFIGS = {
    "irdrop_only":  (cfg_irdrop_only,  None),
    "w_noise_only": (cfg_w_noise_only, None),
    "inp_quant":    (cfg_inp_quant,    None),
    "full_pcm":     (cfg_full_pcm,     lambda l: l.program_analog_weights()),
}
CFG_NAMES = list(CONFIGS.keys())


def _run_layer(spec: LayerSpec, W, b, x) -> dict:
    """Run all (config × rotation) combos for one layer. Returns results dict."""
    rotations = build_rotations(spec.in_dim, x)

    n_combos = len(CONFIGS) * len(rotations)
    print(f"\nRunning {N_TRIALS} trials × {n_combos} combos "
          f"({len(CONFIGS)} configs × {len(rotations)} rotations) ...")

    results = {}
    for cfg_name, (cfg_fn, post_init) in CONFIGS.items():
        results[cfg_name] = {}
        for rot_name, R in rotations.items():
            _log_mem(f"{cfg_name}×{rot_name} start")
            print(f"  {cfg_name} × {rot_name} ...", end=" ", flush=True)
            res = run_trials(W, b, x, R, cfg_fn, N_TRIALS, post_init=post_init)
            results[cfg_name][rot_name] = res
            print(f"rel_err={res['rel_error'][0]:.4f} ± {res['rel_error'][1]:.4f}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    # Print summary
    rot_names = list(rotations.keys())
    print("\n" + "=" * 90)
    print(f"SUMMARY [{spec.display_name}]  —  Relative L2 error (mean ± std)")
    print("=" * 90)
    print(f"{'Config':15s}" + "".join(f" {r:>13s}" for r in rot_names))
    print("-" * 90)
    for cfg_name in CONFIGS:
        row = f"{cfg_name:15s}"
        for rot_name in rot_names:
            m, s = results[cfg_name][rot_name]["rel_error"]
            row += f"  {m:.4f}±{s:.4f}"
        print(row)
    print("=" * 90)

    return results, rotations


def _save_layer_plots(spec, results, rotations, W, b, x, out_dir):
    cfg_fns_only = {k: v[0] for k, v in CONFIGS.items()}
    _plot_bar_charts(results, rotations, CFG_NAMES, metric="rel_error",
                     ylabel="Relative L2 error  ||y_a - y_f|| / ||y_f||",
                     title=f"Rotation vs. Analog Error — {spec.display_name}",
                     fname=f"{out_dir}/rel_error.png")
    _plot_bar_charts(results, rotations, CFG_NAMES, metric="snr_db",
                     ylabel="Output SNR (dB)",
                     title=f"Rotation vs. SNR — {spec.display_name}",
                     fname=f"{out_dir}/snr_db.png", higher_is_better=True)
    _plot_bar_charts(results, rotations, CFG_NAMES, metric="cos_sim",
                     ylabel="Cosine Similarity",
                     title=f"Rotation vs. Cos-Sim — {spec.display_name}",
                     fname=f"{out_dir}/cos_sim.png", higher_is_better=True)
    _plot_improvement_heatmap(results, rotations, CFG_NAMES,
                              fname=f"{out_dir}/improvement_heatmap.png")
    try:
        _plot_output_distributions(W, b, x, rotations, cfg_fns_only,
                                   fname=f"{out_dir}/output_distributions.png")
    except Exception as e:
        print(f"  WARNING: output_distributions plot failed ({e}) — skipping")
    print(f"  Plots saved to {out_dir}/")


def _log_layer_wandb(spec, results, out_dir):
    """Log results to wandb. results[cfg][rot] = {"metric": [mean, std]}."""
    slug = spec.slug
    cols = ["layer", "config", "rotation",
            "rel_error_mean", "rel_error_std",
            "snr_db_mean", "snr_db_std",
            "cos_sim_mean", "cos_sim_std"]
    table = wandb.Table(columns=cols)
    for cfg_name, rot_dict in results.items():
        for rot_name, metrics in rot_dict.items():
            table.add_data(spec.display_name, cfg_name, rot_name,
                           *metrics["rel_error"], *metrics["snr_db"], *metrics["cos_sim"])
    wandb.log({f"{slug}/results_table": table})
    wandb.log({
        f"{slug}/rel_error_chart":     wandb.Image(f"{out_dir}/rel_error.png"),
        f"{slug}/snr_db_chart":        wandb.Image(f"{out_dir}/snr_db.png"),
        f"{slug}/cos_sim_chart":       wandb.Image(f"{out_dir}/cos_sim.png"),
        f"{slug}/improvement_heatmap": wandb.Image(f"{out_dir}/improvement_heatmap.png"),
    })


def _layer_worker(spec_idx: int, out_dir_base: str) -> None:
    """
    Subprocess entry point for one LayerSpec.

    Runs in an isolated process so that the aihwkit C++ RPU tile backend
    (which accumulates CUDA memory outside PyTorch's allocator) is fully
    released when this process exits, preventing cross-layer GPU/RAM leaks.

    Writes results to {out_dir_base}/{spec.slug}/results.json and saves all plots.
    """
    import json
    import sys

    # Ensure prints appear immediately in the parent's tee pipe.
    sys.stdout.reconfigure(line_buffering=True)

    spec = LAYER_SPECS[spec_idx]
    out_dir = f"{out_dir_base}/{spec.slug}"
    os.makedirs(out_dir, exist_ok=True)

    W, b, x = load_layer_and_inputs(spec)
    results, rotations = _run_layer(spec, W, b, x)

    # Write JSON first — before plots — so results survive a plot-time OOM kill.
    serializable = {
        cfg: {
            rot: {k: list(v) for k, v in metrics.items()}
            for rot, metrics in rot_dict.items()
        }
        for cfg, rot_dict in results.items()
    }
    result_path = f"{out_dir}/results.json"
    with open(result_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results written to {result_path}")

    _save_layer_plots(spec, results, rotations, W, b, x, out_dir)


def main():
    import torch.multiprocessing as tmp
    import json

    run_name = f"multilayer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        name=run_name,
        config={
            "n_trials": N_TRIALS,
            "n_layers": len(LAYER_SPECS),
            "layers": [s.display_name for s in LAYER_SPECS],
            "device": DEVICE,
        },
    )

    # Use 'spawn' so each worker gets a clean CUDA context.
    # When the worker process exits, the OS reclaims ALL GPU memory allocated
    # by the aihwkit C++ tile backend — the only reliable way to avoid the
    # cross-layer VRAM leak caused by RPU tiles holding CUDA buffers outside
    # PyTorch's allocator.
    ctx = tmp.get_context("spawn")

    for i, spec in enumerate(LAYER_SPECS):
        print(f"\n{'#'*70}")
        print(f"# {spec.display_name}")
        print(f"{'#'*70}")

        out_dir = f"results/{spec.slug}"
        os.makedirs(out_dir, exist_ok=True)

        p = ctx.Process(target=_layer_worker, args=(i, "results"))
        p.start()
        p.join()

        if p.exitcode != 0:
            print(f"  ERROR: subprocess for '{spec.display_name}' "
                  f"exited with code {p.exitcode} — skipping wandb log")
            continue

        _log_mem("after layer subprocess")

        # Read scalar results back and log to wandb.
        result_path = f"{out_dir}/results.json"
        with open(result_path) as f:
            results = json.load(f)
        _log_layer_wandb(spec, results, out_dir)

    print("\nAll layers done.")
    wandb.finish()


# ---------------------------------------------------------------------------
# 6. Plotting helpers
# ---------------------------------------------------------------------------

ROT_LABELS = {
    "identity":    "Identity",
    "sign_flip":   "Sign Flip (D)",
    "rand_orth":   "Rand. Orth. (QR)",
    "hadamard":    "Block Hadamard",
    "hadamard_D":  "Hadamard + D\n(QuaRot-style)",
    "sorted_perm": "Sorted Perm.",
}

CFG_COLORS = {
    "irdrop_only":  "#e06c75",
    "w_noise_only": "#61afef",
    "inp_quant":    "#98c379",
    "full_pcm":     "#c678dd",
    "adv_irdrop":   "#e5c07b",
}


def _plot_bar_charts(results, rotations, configs, metric, ylabel, title, fname,
                     higher_is_better=False):
    # configs may be a list of names or a dict; normalise to list
    cfg_names = list(configs) if not isinstance(configs, list) else configs
    n_cfg = len(cfg_names)
    n_rot = len(rotations)
    rot_names = list(rotations.keys())

    x_pos = np.arange(n_rot)
    width = 0.8 / n_cfg

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, cfg_name in enumerate(cfg_names):  # noqa: E741
        means = [results[cfg_name][r][metric][0] for r in rot_names]
        stds  = [results[cfg_name][r][metric][1] for r in rot_names]
        offset = (i - n_cfg / 2 + 0.5) * width
        ax.bar(x_pos + offset, means, width * 0.9, yerr=stds,
               label=cfg_name, color=CFG_COLORS[cfg_name],
               error_kw=dict(elinewidth=1.2, capsize=3), alpha=0.9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ROT_LABELS[r] for r in rot_names], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def _plot_improvement_heatmap(results, rotations, configs, fname):
    """
    Heatmap of relative improvement (%) in rel_error vs. identity baseline.
    Positive = improvement (rotation reduces error).
    """
    rot_names = list(rotations.keys())
    cfg_names = list(configs) if not isinstance(configs, list) else configs

    # Exclude identity column itself
    rot_names_no_id = [r for r in rot_names if r != "identity"]

    data = np.zeros((len(cfg_names), len(rot_names_no_id)))
    for i, cfg_name in enumerate(cfg_names):
        baseline = results[cfg_name]["identity"]["rel_error"][0]
        for j, rot_name in enumerate(rot_names_no_id):
            val = results[cfg_name][rot_name]["rel_error"][0]
            data[i, j] = 100 * (baseline - val) / (baseline + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                   vmin=-max(abs(data.min()), abs(data.max())),
                   vmax=max(abs(data.min()), abs(data.max())))
    ax.set_xticks(range(len(rot_names_no_id)))
    ax.set_xticklabels([ROT_LABELS[r] for r in rot_names_no_id], fontsize=9)
    ax.set_yticks(range(len(cfg_names)))
    ax.set_yticklabels(cfg_names)

    for i in range(len(cfg_names)):
        for j in range(len(rot_names_no_id)):
            ax.text(j, i, f"{data[i,j]:+.1f}%", ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="black" if abs(data[i, j]) < 30 else "white")

    plt.colorbar(im, ax=ax, label="Relative improvement (%) vs. identity")
    ax.set_title("Rotation improvement over no-rotation baseline\n"
                 "(green = rotation reduces analog error, red = worsens)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def _plot_output_distributions(W, b, x, rotations, configs, fname):
    """
    For each (rotation, config) show histograms of per-element output error
    for a single noise realisation. Shows the error distribution shape.
    """
    n_rot = len(rotations)
    n_cfg = len(configs)
    rot_names = list(rotations.keys())
    cfg_names  = list(configs.keys())

    x = x[:32].cpu()  # cap for memory safety; keep on CPU for numpy
    W = W.cpu()
    b_cpu = b.cpu() if b is not None else None
    # Float ideal
    if b_cpu is not None:
        y_ideal = (x @ W.T + b_cpu.unsqueeze(0)).detach().numpy().flatten()
    else:
        y_ideal = (x @ W.T).detach().numpy().flatten()

    fig, axes = plt.subplots(n_cfg, n_rot, figsize=(3 * n_rot, 2.5 * n_cfg),
                              sharex=False, sharey=False)

    for i, cfg_name in enumerate(cfg_names):
        for j, rot_name in enumerate(rot_names):
            ax = axes[i][j]
            R     = rotations[rot_name].cpu()
            x_rot = x @ R.T
            W_rot = W @ R.T

            cfg   = configs[cfg_name]()
            layer = make_analog_layer(W_rot, b_cpu, cfg)
            with torch.no_grad():
                y_a = layer(x_rot.to(DEVICE)).cpu().numpy().flatten()

            # Free analog tile immediately — 24 layers across this plot otherwise
            layer.cpu()
            del layer, cfg, W_rot, x_rot, R
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            err = y_a - y_ideal
            ax.hist(err, bins=60, density=True, alpha=0.75,
                    color=CFG_COLORS[cfg_name])
            mu, sigma = err.mean(), err.std()
            ax.set_title(f"{cfg_name}\n{ROT_LABELS[rot_name]}", fontsize=7)
            ax.text(0.97, 0.92, f"μ={mu:.3f}\nσ={sigma:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6)
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.tick_params(labelsize=6)

    fig.suptitle("Output error distribution: y_analog − y_ideal  (1 noise realisation)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
