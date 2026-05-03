#!/usr/bin/env bash
# Table30v2 training launcher.
#
# Usage:
#     bash examples/Table30v2/train_files/run_table30v2_train.sh <bucket>
# Where <bucket> is one of:
#     w1_base | w1_additional | aloha_base | aloha_additional | arx5_base | ur5_base
#
# Optional env overrides:
#     NUM_GPUS, BATCH, MAX_STEPS, SAVE_EVERY, LOG_EVERY, WARMUP,
#     WANDB_MODE (default disabled — set to "online" + ensure wandb_entity in yaml),
#     RUN_TAG (extra suffix appended to run_id).
#
# Run from repo root inside the `starvla` conda env.

set -euo pipefail

# --- args / defaults --------------------------------------------------------
BUCKET=${1:?bucket required: w1_base | w1_additional | aloha_base | aloha_additional | arx5_base | ur5_base}
YAML="examples/Table30v2/train_files/starvla_qwenpi_v3_table30v2_${BUCKET}.yaml"
[[ -f "$YAML" ]] || { echo "yaml not found: $YAML" >&2; exit 1; }

# --- conda env --------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "starvla" ]]; then
  source /primus_xpfs_workspace_T04/xcy/miniforge3/bin/activate starvla
  true
fi

# --- knobs ------------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch;print(torch.cuda.device_count())")}
BATCH=${BATCH:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_EVERY=${SAVE_EVERY:-5000}
LOG_EVERY=${LOG_EVERY:-50}
WARMUP=${WARMUP:-5000}
RUN_TAG=${RUN_TAG:-}

# --- run_id and output_dir --------------------------------------------------
DATE_TAG=$(date +%m%d_%H%M)
run_root_dir=./playground/Checkpoints
run_id=${DATE_TAG}_qwenpi_v3_table30v2_${BUCKET}${RUN_TAG:+_${RUN_TAG}}
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0"   "${output_dir}/"
cp "$YAML" "${output_dir}/"

# --- W&B (off by default) ---------------------------------------------------
export WANDB_MODE=${WANDB_MODE:-disabled}

# --- DeepSpeed nvcc ---------------------------------------------------------
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}

# --- launch -----------------------------------------------------------------
echo "[launch] bucket=${BUCKET}  gpus=${NUM_GPUS}  bs=${BATCH}  grad_accum=${GRAD_ACCUM}  effective=0  max_steps=${MAX_STEPS}"
echo "[launch] yaml=${YAML}"
echo "[launch] run_id=${run_id}  output_dir=${output_dir}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml "$YAML" \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_EVERY}" \
  --trainer.logging_frequency "${LOG_EVERY}" \
  --trainer.num_warmup_steps "${WARMUP}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
