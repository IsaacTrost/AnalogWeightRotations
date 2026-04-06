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

import os
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from transformers import GPT2Model, GPT2Tokenizer

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
N_TRIALS  = 20    # noise realisations per (config, rotation) to average over
IN_DIM    = 768   # GPT-2 hidden size
OUT_DIM   = 3072  # GPT-2 MLP intermediate size
DEVICE    = "cpu"

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
# 3. Load GPT-2 layer + real activations
# ---------------------------------------------------------------------------

def load_gpt2_layer_and_inputs(n_texts: int = 20):
    """
    Returns:
        W  : (OUT_DIM, IN_DIM) float32 tensor — weight matrix of GPT-2 c_fc
        b  : (OUT_DIM,) bias vector
        x  : (n_samples, IN_DIM) float32 tensor — real activations at c_fc input
    """
    print("Loading GPT-2 small ...")
    model = GPT2Model.from_pretrained("gpt2")
    tok   = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    # GPT-2's c_fc uses Conv1D with weight stored as (in, out)
    mlp = model.h[0].mlp
    W = mlp.c_fc.weight.detach().T.float()   # (out, in) = (3072, 768)
    b = mlp.c_fc.bias.detach().float()        # (3072,)
    print(f"  Extracted W shape: {W.shape}, bias shape: {b.shape}")

    # Capture real input activations via forward hook
    captured = []
    def hook(module, inp, out):
        captured.append(inp[0].detach().reshape(-1, IN_DIM))

    handle = mlp.c_fc.register_forward_hook(hook)

    texts = [
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

    with torch.no_grad():
        for text in texts[:n_texts]:
            tokens = tok(text, return_tensors="pt", truncation=True, max_length=64)
            model(**tokens)

    handle.remove()

    x = torch.cat(captured, dim=0).float()
    print(f"  Captured {x.shape[0]} activation samples, "
          f"mean={x.mean():.3f}, std={x.std():.3f}")
    return W, b, x


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
    for _, tile in layer.named_analog_layers():
        if has_bias:
            tile.set_weights(W_rot, b)
        else:
            tile.set_weights(W_rot)
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

    # Rotated inputs and weights
    x_rot = x @ R.T               # (batch, in)
    W_rot = W @ R.T               # (out, in)

    # Float ideal (unrotated — same as rotated in exact arithmetic)
    y_ideal = x @ W.T + b.unsqueeze(0)

    metrics = {"rel_error": [], "snr_db": [], "cos_sim": []}

    for _ in range(n_trials):
        cfg   = rpu_config_fn()
        layer = make_analog_layer(W_rot, b, cfg)
        if post_init is not None:
            post_init(layer)
        m     = eval_analog(layer, x_rot, y_ideal)
        for k in metrics:
            metrics[k].append(m[k])

    return {
        k: (float(np.mean(v)), float(np.std(v)))
        for k, v in metrics.items()
    }


# ---------------------------------------------------------------------------
# 5. Main experiment
# ---------------------------------------------------------------------------

def main():
    W, b, x = load_gpt2_layer_and_inputs()

    # --- Build rotation matrices ---
    print("\nBuilding rotation matrices ...")
    rotations = {
        "identity":    make_identity(IN_DIM),
        "sign_flip":   make_sign_flip(IN_DIM, seed=7),
        "rand_orth":   make_rand_orth(IN_DIM, seed=7),
        "hadamard":    make_block_hadamard(IN_DIM),
        "hadamard_D":  make_hadamard_D(IN_DIM, seed=7),
        "sorted_perm": make_sorted_perm(IN_DIM, x),
    }

    # Verify all rotations are orthogonal (R R^T ≈ I)
    for name, R in rotations.items():
        err = (R @ R.T - torch.eye(IN_DIM)).norm().item()
        print(f"  {name:15s}  R R^T - I  Frobenius norm = {err:.2e}")

    # --- Define analog configs ---
    # Each entry: (config_fn, post_init_fn or None)
    # post_init is called on the layer after weight loading to activate
    # noise models that require an explicit programming step.
    configs = {
        "irdrop_only":  (cfg_irdrop_only,  None),
        "w_noise_only": (cfg_w_noise_only, None),
        "inp_quant":    (cfg_inp_quant,    None),
        "full_pcm":     (cfg_full_pcm,     lambda l: l.program_analog_weights()),
        # adv_irdrop excluded: TorchInferenceRPUConfigIRDropT is too slow for
        # large layers (768x3072) — each trial takes ~10 min on CPU.
    }

    # --- Run experiments ---
    print(f"\nRunning {N_TRIALS} trials per (config × rotation) combo "
          f"[{len(configs)} × {len(rotations)} = {len(configs)*len(rotations)} combos] ...")

    results = {}   # results[config][rotation] = {metric: (mean, std)}
    for cfg_name, (cfg_fn, post_init) in configs.items():
        results[cfg_name] = {}
        for rot_name, R in rotations.items():
            print(f"  {cfg_name} × {rot_name} ...", end=" ", flush=True)
            res = run_trials(W, b, x, R, cfg_fn, N_TRIALS, post_init=post_init)
            results[cfg_name][rot_name] = res
            print(f"rel_err={res['rel_error'][0]:.4f} ± {res['rel_error'][1]:.4f}")

    # --- Print summary table ---
    print("\n" + "=" * 85)
    print("SUMMARY  —  Relative L2 error  (mean ± std)")
    print("=" * 85)
    header = f"{'Config':15s}" + "".join(f" {r:>13s}" for r in rotations)
    print(header)
    print("-" * 85)
    for cfg_name in configs:
        row = f"{cfg_name:15s}"
        for rot_name in rotations:
            m, s = results[cfg_name][rot_name]["rel_error"]
            row += f"  {m:.4f}±{s:.4f}"
        print(row)
    print("=" * 85)

    # --- Plots ---
    # For plotting we only need the config names (not the post_init fns)
    cfg_names_only = list(configs.keys())
    cfg_fns_only   = {k: v[0] for k, v in configs.items()}

    _plot_bar_charts(results, rotations, cfg_names_only, metric="rel_error",
                     ylabel="Relative L2 error  ||y_a - y_f|| / ||y_f||",
                     title="Effect of Rotation on Analog Layer Relative Error",
                     fname="results/rel_error.png")

    _plot_bar_charts(results, rotations, cfg_names_only, metric="snr_db",
                     ylabel="Output SNR (dB)",
                     title="Effect of Rotation on Analog Layer SNR",
                     fname="results/snr_db.png", higher_is_better=True)

    _plot_bar_charts(results, rotations, cfg_names_only, metric="cos_sim",
                     ylabel="Cosine Similarity (per-sample mean)",
                     title="Effect of Rotation on Analog Layer Cosine Similarity",
                     fname="results/cos_sim.png", higher_is_better=True)

    _plot_improvement_heatmap(results, rotations, cfg_names_only,
                              fname="results/improvement_heatmap.png")

    _plot_output_distributions(W, b, x, rotations, cfg_fns_only,
                                fname="results/output_distributions.png")

    print("\nPlots saved to ./results/")


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

    x = x[:32]  # cap for memory safety (matches run_trials cap)
    # Float ideal
    y_ideal = (x @ W.T + b.unsqueeze(0)).detach().numpy().flatten()

    fig, axes = plt.subplots(n_cfg, n_rot, figsize=(3 * n_rot, 2.5 * n_cfg),
                              sharex=False, sharey=False)

    for i, cfg_name in enumerate(cfg_names):
        for j, rot_name in enumerate(rot_names):
            ax = axes[i][j]
            R     = rotations[rot_name]
            x_rot = x @ R.T
            W_rot = W @ R.T

            cfg   = configs[cfg_name]()
            layer = make_analog_layer(W_rot, b, cfg)
            with torch.no_grad():
                y_a = layer(x_rot).numpy().flatten()

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
