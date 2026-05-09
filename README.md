# HPML Final Project: [Project Title]

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** Analog Hardware Rotations (team 13)
- **Members:**
  - William Trost (wit2102) — **
  - Full Name 2 (UNI) — *role / area of contribution*
  - Full Name 3 (UNI) — *role / area of contribution*

## Submission

- **GitHub repository:** [https://github.com/&lt;org&gt;/&lt;repo&gt;](https://github.com/org/repo)
- **Final report:** [`deliverables/HPML_Final_Report.pdf`](deliverables/HPML_Final_Report.pdf)
- **Final presentation:** [`deliverables/HPML_Final_Presentation.pptx`](deliverables/HPML_Final_Presentation.pptx)
- **Experiment-tracking dashboard:** [link to public Wandb / MLflow / TensorBoard / Comet / Neptune dashboard]

The final report PDF and the presentation file are checked into the `deliverables/` folder of this repository **and** uploaded to CourseWorks.

---

## 1. Problem Statement

We are building a pipeline to automatically train and apply rotation matrices to llama models. The goal is to build out and apply custom rotation matrices to change the weights of the tinyllama model such that the output is unchanged in the ideal case, we dont have to fully finetune, and we distribute the weights such that the hardware non-idealities are less present.

---

## 2. Model/Application Description

Briefly describe the model(s) and stack you used:

- **Model architecture:** TinyLlama/TinyLlama_v1.1
- **Framework:** PyTorch 2.10, IBM/AIHWKIT 1.1.0
- **Dataset:** name, size, license, and link.
- **Custom layers or modifications:** The addition of rotation matrices r1-r4, as taught in Spinquant, but trained based on hardware non-idealities
- **Hardware target:** Generic In-memory compute chips with typical hardware non-idealities.

---

## 3. Final Results Summary

Replace the numbers below with your measured values. Add or remove rows to fit your study.

| Metric                       | Baseline | Optimized | Δ (Improvement) |
| ---------------------------- | -------- | --------- | --------------- |
| Top-1 Accuracy / Task Metric | XX.XX%   | XX.XX%    | ±X.XX pp        |
| Inference Latency (p50)      | XX.XX ms | XX.XX ms  | XX% faster      |
| Inference Throughput         | XXX tok/s| XXX tok/s | XX× higher      |
| Training Time / Epoch        | XX s     | XX s      | XX% faster      |
| Peak GPU Memory              | XX GB    | XX GB     | XX% less        |
| Model Size on Disk           | XX MB    | XX MB     | XX% smaller     |
| Energy / Sample (optional)   | X.XX J   | X.XX J    | XX% less        |

**Hardware:** 1x Nvidia 3090

**Headline result (one sentence):** *By applying the rotation matrices, we were able to achieve substantially lower perplexity on the wikitext-2 dataset compared to the non-rotated model*

---

## 4. Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── configs/                # JSON configs for every reported experiment
├── deliverables/           # Final report (PDF) and final presentation (PPT/PDF) — same files uploaded to CourseWorks
│   ├── HPML_Final_Report.pdf
│   └── HPML_Final_Presentation.pptx
├── scripts/
│   ├── download_dataset.sh
│   ├── run_baseline.sh
│   └── run_optimized.sh
├── src/
│   ├── analog_llama.py     # Analog LLaMA wrapper
│   ├── hardware_configs.py # Analog hardware presets
│   ├── train_full_analog.py
│   ├── train_layerwise_analog.py
│   ├── eval_analog_perplexity.py
│   ├── optimizer.py    
│   ├── rotation_utils.py    
│   └── ...                 # Rotation, quantization, pipeline, and evaluation helpers
├── tests/
│   ├── test_r2_rotation.py
│   └── test_runtime_rotation.py
├── checkpoints/            # Generated model checkpoints
├── logs/                   # Runtime logs
├── profiler_traces/        # Generated profiler traces
├── results/                # Experiment outputs, summaries, and plots
└── wandb/                  # Local Weights & Biases run metadata
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

> **🔗 Dashboard:** [https://wandb.ai/&lt;team&gt;/&lt;project&gt;](https://wandb.ai/team/project)
>
> *Platform used:* [Weights & Biases / MLflow / TensorBoard / Comet / Neptune / other]

Verify the link opens in an incognito browser. The dashboard includes a curated **report** that walks through the optimization story. If your platform does not support public links (e.g., self-hosted MLflow), a static export is committed under `results/dashboard/` instead.

### C. Dataset

The dataset is *not* committed to the repository. It is small, and manually pulled into the docker container huggingface cache if scripts need it.

### D. Training

To produce the trained r1 and r2 results

```bash
python3 -m src.train_full_analog --config configs/train_full_hadamard_d_ir0p5_bits8_steps300.json
```

### E. Evaluation

```bash
python src/eval.py --weights checkpoints/best_model.pth --config configs/optimized.yaml
```

### F. Profiling

To regenerate the profiler traces referenced in the report:

```bash
python src/profile.py --config configs/optimized.yaml --output results/trace.json
# View in chrome://tracing or perfetto.dev
```

### G. Quickstart: Reproduce the Headline Result

The following sequence reproduces the headline number in Section 3 end-to-end (≈ XX minutes on a single A100):

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. Download dataset
bash scripts/download_dataset.sh

# 3. Run optimized training (or skip if checkpoint provided in releases)
bash scripts/run_optimized.sh

# 4. Evaluate
python src/eval.py --weights checkpoints/best_model.pth
```

---

## 6. Results and Observations

A short narrative (3–6 bullets) summarizing what you found. Include 1–2 representative figures from `results/` directly in this README so a reader gets the gist without opening Wandb.

- *Optimization 1 (e.g., torch.compile + bfloat16):* X% latency reduction, attributable to [reason].
- *Optimization 2 (e.g., FlashAttention-2):* Y% memory reduction at long context lengths.
- *Optimization 3 (e.g., paged KV cache):* Z× throughput gain at batch size 32.
- *What did not work:* [briefly note any optimization that failed or regressed performance, and why you think it failed].

![Baseline vs Optimized latency](results/figures/latency_comparison.png)

---

## 7. Notes

- Source files live under `src/`, configuration under `configs/`, and scripts under `scripts/`.
- Trained checkpoints are stored in [GitHub Releases / Hugging Face Hub / external bucket] — see `docs/checkpoints.md`.
- All secrets (API keys, Wandb tokens) are loaded from environment variables. See `.env.example`.

### AI Use Disclosure

*Per the HPML AI Use Policy (posted on CourseWorks). Required for every submission.*

**Did your team use any AI tool in completing this project?**

- [ ] No, we did not use any AI tool.
- [ ] Yes, we used AI assistance as described below.

**Tool(s) used:** *e.g., ChatGPT, Claude, GitHub Copilot, Cursor*

**Specific purpose:** *e.g., debugged a CUDA OOM error, clarified SM occupancy, polished prose in the report's introduction*

**Sections affected:** *e.g., src/profile.py setup, README §6 results narrative, report §V Discussion*

**How we verified correctness:** *e.g., re-ran every reported experiment ourselves; confirmed profiler-trace interpretations against the raw traces in results/; rewrote AI-suggested code in our own words and confirmed it produces the same numbers as the version we hand-wrote.*

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

If you build on this work, please cite:

```bibtex
@misc{teamname2026hpml,
  title  = {[Project Title]},
  author = {Last1, First1 and Last2, First2 and Last3, First3},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/<org>/<repo>}
}
```

### Contact

Open a GitHub Issue or email *[wit2102@columbia.edu]*.

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*
