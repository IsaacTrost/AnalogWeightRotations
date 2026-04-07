# How to use
## My workflow
`
(base) ao2844_columbia_edu@instance-20260325-210342:~/AnalogWeightRotations/aimc-docker$ sudo docker build --no-cache -t aihwkit-min .
`

`
$ export WANDB_API_KEY=<your-api-key>
`

`
$ export WANDB_ENTITY=ao2844-columbia-university
`

`
$ export WANDB_PROJECT=aimc-rotations
`

Mind the directory I run it in isn't aimc-docker:
`
(base) ao2844_columbia_edu@instance-20260325-210342:~/AnalogWeightRotations$ sudo docker run --gpus all -it --rm   -p 8888:8888   -v $(pwd):/workspace   -e WANDB_API_KEY=$WANDB_API_KEY   -e WANDB_ENTITY=$WANDB_ENTITY   -e WANDB_PROJECT=$WANDB_PROJECT   aihwkit-min
`

`
$ conda run -n aihwkit python -m src.baseline_forward
`

## Jupyter
```
$ source /opt/conda/etc/profile.d/conda.sh
$ conda activate aihwkit
$ conda run -n aihwkit jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

# Overview
`
src/baseline_forward.py
`

Loads a Hugging Face model (currently GPT-2)
Runs a baseline forward pass
Captures intermediate activations via hooks
Logs basic metrics to W&B
Purpose: establish a reference point before applying rotations

`
src/baseline_forward.py
`

Loads a Hugging Face model (currently GPT-2)
Runs a baseline forward pass
Captures intermediate activations via hooks
Logs basic metrics to W&B
Purpose: establish a reference point before applying rotations

`
src/rotation_utils.py
`

Core math utilities:
random_orthogonal_matrix(n)
hadamard_matrix(n) (power-of-2 only)
apply_rotation(x, R)
helper functions (orthonormal checks, etc.)
Purpose: define and apply rotation matrices

`
src/apply_rotation.py
`

Runs baseline forward pass
Applies rotation → inverse rotation
Compares outputs with baseline
Logs error metrics to W&B
Purpose: verify rotations preserve model computation (sanity check)

`
experiments/rotation_experiments.ipynb
`

Interactive experimentation
Activation visualization (histograms, stats)
Quick testing of rotations
Purpose: exploration + generating plots for reports

---

# 1-Layer Rotation Experiments (`1layertests/`)

Systematic evaluation of whether orthogonal input rotations reduce output error in a single analog linear layer extracted from GPT-2 small.

**Key identity:** `y = Wx = (WR^T)(Rx)` for any orthogonal R. Exact in float; on analog hardware the rotation redistributes energy across crossbar rows/columns, potentially mitigating IR drop, weight noise, and ADC quantization error.

## Layer under test

GPT-2 small (124M), first transformer block, MLP feed-forward projection:
- `c_fc`: `(batch, 768) → (batch, 3072)`, weight shape `(3072, 768)`
- Weights and input activations extracted via forward hook (realistic statistics)

## Rotations compared

| Name | Description |
|------|-------------|
| `identity` | No rotation (baseline) |
| `sign_flip` | Random diagonal ±1 matrix D |
| `rand_orth` | Random orthogonal via QR decomposition |
| `hadamard` | Block-diagonal Hadamard `blkdiag(H₂₅₆, H₂₅₆, H₂₅₆) / √256` |
| `hadamard_D` | `H @ D` — QuaRot-style: incoherence + sign spreading |
| `sorted_perm` | Permutation sorting inputs by mean \|activation\| (energy balancing) |

All rotations verified orthogonal (`R R^T ≈ I`) before use.

## Analog hardware configs

| Config | Description |
|--------|-------------|
| `irdrop_only` | IR drop = 1.0, all other noise disabled |
| `w_noise_only` | Additive weight noise σ=0.02, no IR drop |
| `inp_quant` | 8-bit input + output quantization only |
| `full_pcm` | PCM-like noise model + IR drop = 0.5 + 10-bit ADC/DAC (realistic) |

## Metrics

Each `(config, rotation)` combination is averaged over 20 independent noise trials:
- `rel_error` — `‖y_analog − y_ideal‖_F / ‖y_ideal‖_F`
- `snr_db` — `10 log₁₀(‖y_ideal‖² / ‖y_analog − y_ideal‖²)`
- `cos_sim` — mean per-sample cosine similarity

## Key results

Relative L2 error (mean):

| Config | identity | sign_flip | rand_orth | hadamard | hadamard_D | sorted_perm |
|--------|----------|-----------|-----------|----------|------------|-------------|
| IR drop only | 6.85% | 6.85% | **0.13%** | **0.13%** | **0.13%** | 6.82% |
| Weight noise | 9.07% | 9.07% | **6.24%** | **6.24%** | **6.24%** | 9.05% |
| 8-bit quant | 7.34% | 7.34% | **2.82%** | **2.83%** | **2.83%** | 7.35% |
| Full PCM | 14.83% | 14.84% | 6.29% | 6.40% | **6.10%** | 15.99% |

**Takeaways:**
- `rand_orth`, `hadamard`, and `hadamard_D` all yield large error reductions (~50–98%) across every config
- `sign_flip` and `sorted_perm` provide no meaningful benefit — they don't redistribute energy incoherently
- `hadamard_D` (QuaRot-style) is the best single rotation under the realistic full PCM config (−58.8% vs identity)
- Results are deterministic for `irdrop_only` / `inp_quant` (no stochastic noise), and low-variance for the others

## Running the experiment

```bash
# Full experiment (loads GPT-2, runs 20 trials × 4 configs × 6 rotations)
$ conda run -n aihwkit python -m 1layertests.explore_rotations
```

Or inside the Docker container (from `AnalogWeightRotations/`):

```bash
$ conda run -n aihwkit python 1layertests/explore_rotations.py
```

Results and plots are saved to `1layertests/results/` and logged to W&B under the configured project.

## Regenerating plots without re-running

`plot_results.py` hardcodes the completed run's output and regenerates all figures locally:

```bash
$ conda run -n aihwkit python 1layertests/plot_results.py
```

Produces: `results/rel_error.png`, `results/snr_db.png`, `results/improvement_heatmap.png`, `results/snr_panel.png`, `results/summary_table.png`
