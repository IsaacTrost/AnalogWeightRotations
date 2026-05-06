#!/usr/bin/env bash
set -euo pipefail

MODEL="HuggingFaceTB/SmolLM2-135M"
CKPT="checkpoints/smollm2_135m_hadamardD_100steps.pt"
TOKENS=2048
MAX_LENGTH=64
BATCH_SIZE=1

mkdir -p results/ablation

HARDWARE_PRESETS=(
  # "quant_only"
  # "ir_drop_only"
  "weight_noise_only"
)

TARGET_NAMES=(
  "downproj"
  "mlp"
  "all_no_lm_head"
)

TARGET_ARGS=(
  "down_proj"
  "up_proj gate_proj down_proj"
  "q_proj k_proj v_proj o_proj up_proj gate_proj down_proj"
)

run_eval () {
  local hardware="$1"
  local target_name="$2"
  local targets="$3"
  local mode="$4"
  local extra_args="$5"

  local out="results/ablation/smollm2_135m_${hardware}_${target_name}_${mode}_${TOKENS}.json"

  echo
  echo "============================================================"
  echo "hardware=${hardware} targets=${targets} mode=${mode}"
  echo "output=${out}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="" conda run -n aihwkit python -m src.eval_analog_perplexity \
    --model-name "${MODEL}" \
    --device cpu \
    --torch-dtype float32 \
    --hardware-preset "${hardware}" \
    --analog-targets ${targets} \
    --max-eval-tokens "${TOKENS}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    --json-output "${out}" \
    --use-wandb \
    ${extra_args}
}

for hardware in "${HARDWARE_PRESETS[@]}"; do
  for i in "${!TARGET_NAMES[@]}"; do
    target_name="${TARGET_NAMES[$i]}"
    targets="${TARGET_ARGS[$i]}"

    run_eval "${hardware}" "${target_name}" "${targets}" "identity" "--identity-r1-r2"
    run_eval "${hardware}" "${target_name}" "${targets}" "hadamardD" "--rotation-mode hadamard_D"
    run_eval "${hardware}" "${target_name}" "${targets}" "learned" "--checkpoint ${CKPT}"
  done
done

echo
echo "All ablation runs complete. JSON files saved under results/ablation/"