# HPML Final Project: Analog Weight Rotations

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** Analog Hardware Rotations (team 13)
- **Members:**
  - William Trost (wit2102) — rotation training, AIHWKit integration, evaluation
  - Alvis Ou — experiments, analysis, reporting
  - David Wang — experiments, analysis, reporting

## Submission

- **GitHub repository:** [https://github.com/IsaacTrost/AnalogWeightRotations](https://github.com/IsaacTrost/AnalogWeightRotations)
- **Final report:** [`deliverables/HPML_Final_Report.pdf`](deliverables/HPML_Final_Report.pdf)
- **Final presentation:** [`deliverables/HPML_Final_Presentation.pptx`](deliverables/HPML_Final_Presentation.pptx) Please note that the slides reflect outdated results with lower IR values, we did not change the config until later. The results are still valid, but focus more on IR than quantization
- **Experiment-tracking dashboard:** [W&B Final report](https://wandb.ai/ao2844-columbia-university/aimc-rotations/reports/Final-report--VmlldzoxNjgyNTE5Mw)

The final report PDF and the presentation file are checked into the `deliverables/` folder of this repository **and** uploaded to CourseWorks.

The dashboard is not excessive, as our project was not conducive to excessive logging. It does show the training runs we made with various configs, however, as well as the grids showing performance of the trained checkpoints compared to the baseline which is representative of the effectivness of this method. 

---

## 1. Problem Statement

We are building a pipeline to automatically train and apply rotation matrices to LLaMA-style models. The goal is to apply custom rotations to TinyLlama so the ideal floating-point output is preserved without full fine-tuning, while redistributing activations and weights to reduce sensitivity to analog hardware non-idealities.

---

## 2. Model/Application Description

- **Model architecture:** TinyLlama/TinyLlama_v1.1
- **Framework:** PyTorch 2.10, IBM/AIHWKIT 1.1.0
- **Dataset:** WikiText-2 validation split, loaded through Hugging Face `datasets`
- **Custom layers or modifications:** Rotation matrices R1-R4 following the SpinQuant structure, with R1/R2 trained directly against hardware non-idealities
- **Hardware target:** In-memory analog compute with quantization, IR drop, and related non-idealities simulated through AIHWKit.

---

## 3. Final Results Summary

Perplexity on WikiText-2 validation under analog non-idealities. Lower PPL is better; improvement is identity PPL divided by learned-rotation PPL.

Moderate IR-drop setting (`results/new/summary.csv`):

| Hardware cell | Identity R1/R2 PPL | Fixed Hadamard-D PPL | Learned rotations PPL | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 8-bit DAC/ADC, IR drop 0.0 | 20.17 | 13.69 | 13.69 | 1.47× |
| 8-bit DAC/ADC, IR drop 0.5 | 19.84 | 13.66 | 13.66 | 1.45× |
| 8-bit DAC/ADC, IR drop 1.0 | 20.45 | 13.68 | 13.67 | 1.50× |
| 6-bit DAC/ADC, IR drop 0.0 | 22.43 | 13.93 | 13.94 | 1.61× |
| 6-bit DAC/ADC, IR drop 0.5 | 22.63 | 13.95 | 13.92 | 1.63× |
| 6-bit DAC/ADC, IR drop 1.0 | 22.43 | 13.94 | 13.94 | 1.61× |

High IR-drop stress test (`results/high_ir/summary.csv`, 8-bit DAC/ADC) (note that IR drop of 50 was above the trained value):

| IR drop | Identity R1/R2 PPL | Fixed Hadamard-D PPL | Learned rotations PPL | Improvement |
| ---: | ---: | ---: | ---: | ---: |
| 5.0 | 21.15 | 14.63 | 14.28 | 1.48× |
| 10.0 | 21.70 | 16.58 | 14.64 | 1.48× |
| 20.0 | 21.43 | 48.47 | 16.15 | 1.33× |
| 50.0 | 103.92 | 2.31e6 | 147.65 | 0.70× |

**Hardware:** 1x Nvidia 3090

**Headline result (one sentence):** *By applying the rotation matrices, we were able to achieve substantially lower perplexity on the wikitext-2 dataset compared to the non-rotated model for configs we trained on*

---

## 4. Repository Structure

```
.
├── README.md
├── LICENSE
├── configs/                # JSON configs for training/eval sweeps
├── results/                # Experiment JSON/CSV summaries and generated plots
├── checkpoints/            # Trained rotation checkpoints
├── deliverables/           # Final report and presentation submitted to CourseWorks
│   ├── HPML_Final_Report.pdf
│   └── HPML_Final_Presentation.pptx
├── aimc-docker/            # AIHWKit Docker environment
│   └── smoke_test.py
├── experiments/            # Plotting and ablation helper scripts
│   ├── plot_ablation_results.py
│   ├── plot_hadamard_grid.py
│   └── run_eval_ablations.sh
├── 1layertests/            # Early single-layer rotation experiments
│   ├── explore_rotations.py
│   ├── plot_combined.py
│   ├── plot_results.py
│   ├── stuck_at_faults.py
│   ├── train_rotation_first_order.py
│   ├── train_rotation_matrix.py
│   └── train_weight_matrix.py
├── src/
│   ├── analog_llama.py
│   ├── analyze_rotation_channel_concentration.py
│   ├── calibration_data.py
│   ├── cayley_optimizer.py
│   ├── eval_analog_perplexity.py
│   ├── eval_rotation.py
│   ├── full_model_pipeline.py
│   ├── hardware_config_smoke.py
│   ├── hardware_configs.py
│   ├── lean_analog_inference.py
│   ├── llama_model.py
│   ├── llama_prepare.py
│   ├── llama_rotation.py
│   ├── llama_verify.py
│   ├── optimizer.py
│   ├── plot_hadamard_grid.py
│   ├── plot_rotation_activation_heatmaps.py
│   ├── plot_rotation_current_proxies.py
│   ├── plot_rotation_weight_heatmaps.py
│   ├── rotation_losses.py
│   ├── rotation_precision.py
│   ├── rotation_utils.py
│   ├── run_hadamard_grid.py
│   ├── runtime_rotation.py
│   ├── train_full_aihwkit.py
│   ├── train_full_analog.py
│   ├── train_runtime_rotation.py
│   ├── trainable_rotation.py
│   ├── wandb_config.py
│   ├── wandb_logging.py
│   └── legacy/
│       ├── __init__.py
│       ├── apply_rotation.py
│       ├── baseline_forward.py
│       ├── convert_spinquant_rotation.py
│       ├── eval_fake_quant_perplexity.py
│       ├── full_pipeline_smoke.py
│       ├── train_layerwise_analog.py
│       └── train_r1.py
├── tests/
│   ├── test_r2_rotation.py
│   ├── test_runtime_rotation.py
│   └── test_runtime_rotation_training.py
├── run.sh                  # Launch the AIHWKit Docker container
├── logs/                   # Runtime logs, if generated
├── profiles/               # Profiler traces, if generated
└── wandb/                  # Local Weights & Biases run metadata, if generated

```

---

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
# Clone
git clone https://github.com/IsaacTrost/AnalogWeightRotations.git 
cd AnalogWeightRotations

# Build the docker container

cd aimc-docker
docker build --no-cache -t aihwkit-min .

# run the docker container
./run.sh
```

**System requirements:** Docker or Podman with GPU passthrough enabled, NVIDIA Container Toolkit / NVIDIA container runtime, and a host NVIDIA driver compatible with the CUDA version in `nvcr.io/nvidia/pytorch:25.04-py3` (CUDA 12.x; 570-series drivers are recommended). CUDA itself does not need to be installed on the host because the container provides the CUDA user-space libraries. Use a GPU with at least 16 GB VRAM; 24 GB is preferred because AIHWKIT can be memory hungry and leaky, so random crashes in validation scripts are possible.

### B. Experiment Tracking Dashboard

Public experiment-tracking dashboard with training and evaluation metrics, system profiling, and baseline vs. optimized comparisons:

> **🔗 Dashboard:** [Final report](https://wandb.ai/ao2844-columbia-university/aimc-rotations/reports/Final-report--VmlldzoxNjgyNTE5Mw)
>
> *Platform used:* [Weights & Biases]

We did not run all of the eval scripts through wandb through a configuration error. Thus, we are forced to just upload the end heatmap results. The runs we had cached were uploaded late before we submitted.

### C. Dataset

The dataset is *not* committed to the repository. It is small, and manually pulled into the docker container huggingface cache if scripts need it.

### D. Training

To produce the trained r1 and r2 results

```bash
python3 -m src.train_full_aihwkit --config configs/train_full_aihwkit_high_ir_8_bit.json
```

### E. Evaluation

Validate the trained high-IR 8-bit checkpoint with the analog perplexity evaluator:

```bash
python3 -m src.eval_analog_perplexity --config configs/eval_high_ir_8bit.json
```

### F. Profiling

Profiling is enabled through the training config flags in `src/train_full_analog.py`. When `profile` is enabled, traces are written under `profiles/` and can be viewed in Chrome tracing or Perfetto.

### G. Quickstart: Reproduce the Headline Result

The following sequence reproduces the headline number in Section 3 end-to-end (≈ 30 minutes on a single 3090):

```bash
# 1. Build and enter the AIHWKit Docker environment
cd aimc-docker
docker build --no-cache -t aihwkit-min .
cd ..
./run.sh

# 2. Train the high-IR 8-bit rotation checkpoint
python3 -m src.train_full_aihwkit --config configs/train_full_aihwkit_high_ir_8_bit.json

# 3. Validate with analog perplexity eval
python3 -m src.eval_analog_perplexity --config configs/eval_high_ir_8bit.json
```

---

## 6. Results and Observations

- *Trained R1 residual-stream rotation:* Learning the global hidden-state rotation reduced sensitivity to analog non-idealities by redistributing large activation/weight coordinates before they reach the analog matrix-vector products. This was the most broadly useful learned component across the reported 8-bit and 6-bit settings.
- *Trained R2 attention value/output rotation:* Learning the per-layer OV-path rotations gave the model additional freedom to reshape attention-channel structure without changing the ideal floating-point function. This helped preserve perplexity under quantization and IR-drop, especially when used with the trained R1 checkpoint.
- *Online Hadamards R3/R4:* The fixed online Hadamards were intended to flatten attention-output and FFN intermediate outliers, but they did not work as well in our AIHWKit perplexity runs. In high-IR settings, the fixed Hadamard-D baseline often lagged the learned checkpoint and sometimes regressed badly, suggesting the untrained online transforms were too rigid for the simulated hardware errors.
- *Main takeaway:* Rotation helps most when the rotation is trained against the hardware loss rather than applied as a fixed preprocessing trick on realistic analog inconsistencies, but it must be trained against the specific hardware configuration.

![High-IR rotation perplexity heatmap](results/high_ir/all_methods_improvement_ratio_wnoise0.png)

---

## 7. Notes

- Source files live under `src/`, configuration under `configs/`
- Trained checkpoints are stored in this repo (each one is only a couple megabytes)
- All secrets (W&B tokens) are loaded from environment variables.
```
WANDB_API_KEY=<your_wandb_key>
WANDB_ENTITY=<your_entity>
WANDB_PROJECT=<project_name>
WANDB_MODE=online
```
### AI Use Disclosure

*Per the HPML AI Use Policy (posted on CourseWorks). Required for every submission.*

**Did your team use any AI tool in completing this project?**

- [ ] No, we did not use AI assistance.
- [x] Yes, we used AI assistance as described below.

**Tool(s) used:** *GitHub Copilot, Cursor*

**Specific purpose:** *Wrote boilerplate plotting code, helped explore OOM errors, fixed documentation, wrote clearer code comments when necessary, and proofread the report for errors and inconsistencies.*

**Sections affected:** *Plotting utilities, documentation, and code comments across selected `src/` files.*

**How we verified correctness:** *The small amount of fully AI-generated code was not used in critical training logic; plotting code, comments, and proofreading suggestions were read and verified by the team.*

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

If you build on this work, please cite:

```bibtex
@misc{teamname2026hpml,
  title  = {Analog weight rotations},
  author = {Trost, William and Ou Alvis and Wang, David},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/IsaacTrost/AnalogWeightRotations}
}
```

### Contact

Open a GitHub Issue or email *[wit2102@columbia.edu]*.

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*
