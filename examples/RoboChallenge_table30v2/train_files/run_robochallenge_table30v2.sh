#!/usr/bin/env bash
# RoboChallenge Table30v2 — training launcher (UR5 / ARX5 / DOS-W1 / ALOHA).
# Usage (local):
#   bash examples/RoboChallenge_table30v2/train_files/run_robochallenge_table30v2.sh
# Usage (SLURM):
#   srun --jobid=<JOB_ID> --overlap --pty bash examples/RoboChallenge_table30v2/train_files/run_robochallenge_table30v2.sh
set -euo pipefail

# === Conda setup ===
if [[ "${CONDA_DEFAULT_ENV:-}" != "starVLA_dev" ]]; then
  _CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/.conda")"
  _CONDA_SH="${_CONDA_BASE}/etc/profile.d/conda.sh"
  if [[ -f "${_CONDA_SH}" ]]; then
    source "${_CONDA_SH}"
    conda activate starVLA_dev
  else
    echo "[WARN] conda.sh not found at ${_CONDA_SH}, skipping conda activate"
  fi
fi

# === CUDA setup — try common system paths first, then fall back to a stub wrapper ===
for cuda_path in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.4 \
                 /cm/shared/apps/cuda12.2/toolkit/12.2.2; do
  if [ -x "${cuda_path}/bin/nvcc" ]; then
    export CUDA_HOME="${cuda_path}"
    export PATH="${cuda_path}/bin:${PATH}"
    export LD_LIBRARY_PATH="${cuda_path}/lib64:${LD_LIBRARY_PATH:-}"
    break
  fi
done

if ! nvcc --version 2>&1 | grep -q "release"; then
  _WRAPPER_DIR="${CONDA_PREFIX}/cuda_compat/bin"
  mkdir -p "${_WRAPPER_DIR}" 2>/dev/null || true
  _TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "12.4")
  _MAJOR=$(echo "${_TORCH_CUDA_VER}" | cut -d. -f1)
  _MINOR=$(echo "${_TORCH_CUDA_VER}" | cut -d. -f2)
  cat > "${_WRAPPER_DIR}/nvcc" << NVCC_EOF
#!/bin/bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release ${_MAJOR}.${_MINOR}, V${_TORCH_CUDA_VER}"
NVCC_EOF
  chmod +x "${_WRAPPER_DIR}/nvcc"
  export CUDA_HOME="${CONDA_PREFIX}/cuda_compat"
  echo "[INFO] Created nvcc stub wrapper: CUDA ${_TORCH_CUDA_VER}"
fi

echo "[INFO] CUDA_HOME=${CUDA_HOME}"
nvcc --version 2>/dev/null || echo "[WARN] nvcc not found"

# === NCCL stability ===
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

###########################################################################################
# === Please modify the following variables for your environment ===

cd /home/jye624/Projcets/starVLA

# --- model ---
Framework_name=QwenOFT
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}

# --- data ---
data_root=./playground/Datasets/RoboChallenge_table30v2
data_mix=rc2_aloha_one
# Available mixtures (edit data_mix above to switch):
#   rc2_ur5_all
#   rc2_arx5_all
#   rc2_dosw1_all
#   rc2_aloha_all
#   rc2_all          (all embodiments combined)
#   rc2_aloha_one  (single-task walk-through)

# --- training ---
BATCH=${BATCH:-16}
MAX_STEPS=${MAX_STEPS:-130000}
SAVE_EVERY=${SAVE_EVERY:-10000}
EVAL_EVERY=${EVAL_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-100}
freeze_module_list=''

# --- output ---
run_root_dir=./results/Checkpoints
run_id=$(date +%m%d)_${data_mix}_${Framework_name}

# === End of environment-specific configuration ===
###########################################################################################

# export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_API_KEY=${WANDB_API_KEY:-}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29506}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}
accelerate_config_file=${ACCELERATE_CONFIG_FILE:-starVLA/config/deepseeds/deepspeed_zero2.yaml}

config_yaml=./examples/RoboChallenge_table30v2/train_files/starvla_qwenoft_robochallenge_table30v2.yaml

# Fix: ensure vonneumann1 group is active for NFS file access on compute nodes.
CONDA_BASE=$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX%/envs/*}")
CONDA_INIT="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_DEFAULT_ENV:-starVLA_dev}"

sg vonneumann1 -c "
${CONDA_INIT} && \
accelerate launch \
  --config_file ${accelerate_config_file} \
  --num_processes ${num_processes} \
  --main_process_port ${MAIN_PROCESS_PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.attn_implementation ${attn_implementation} \
  --datasets.vla_data.data_root_dir ${data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${BATCH} \
  --trainer.freeze_modules '${freeze_module_list}' \
  --trainer.max_train_steps ${MAX_STEPS} \
  --trainer.save_interval ${SAVE_EVERY} \
  --trainer.logging_frequency ${LOG_EVERY} \
  --trainer.eval_interval ${EVAL_EVERY} \
  --trainer.is_resume True \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_robochallenge_table30v2 \
  --wandb_entity jinhuiye
"
