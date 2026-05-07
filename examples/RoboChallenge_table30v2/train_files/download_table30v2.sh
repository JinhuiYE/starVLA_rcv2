#!/usr/bin/env bash
# =============================================================================
# RoboChallenge Table30v2 — background download + convert + cleanup (all tasks)
#
# Per-task pipeline:
#   1. Download .tar.part-* shards from HF, concatenate + extract to raw/<task>/
#   2. Convert raw/<task>/ -> lerobot/<task>/ (LeRobot v2.1 + gr00t modality)
#   3. Remove raw/<task>/ to free disk  (lerobot/ has its own video copies)
#
# At the end: auto-updates data_config.py task-sets.
#
# Usage (from repo root):
#   mkdir -p tmp/logs
#   nohup bash examples/RoboChallenge_table30v2/train_files/bg_download_convert_all.sh \
#       > tmp/logs/bg_dcall.log 2>&1 &
#   echo "PID $!"
#
# Re-run any time — already-converted tasks are skipped.
#
# Env vars (override if needed):
#   DATA_BASE   Base dir for data (default: playground/Datasets/RoboChallenge_table30v2)
#   ONLY_TASKS  Space-separated subset of tasks to process (default: all)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../../.."      # always run from repo root

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="tmp/logs/bg_dcall_${TS}.log"
mkdir -p tmp/logs

# Tee stdout+stderr to a timestamped log file in addition to the caller's pipe.
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[${TS}] bg_download_convert_all.sh starting"
echo "[info] log: ${LOG_FILE}"
echo "[info] pwd: $(pwd)"

# ---------------------------------------------------------------------------
# 1. Stop any stray old download process from the previous session
# ---------------------------------------------------------------------------
OLD_PIDS=$(pgrep -f "download_table30v2.py.*--raw-root" 2>/dev/null || true)
if [[ -n "${OLD_PIDS}" ]]; then
    echo "[init] Stopping old download process(es): ${OLD_PIDS}"
    kill ${OLD_PIDS} 2>/dev/null || true
    sleep 2
fi

# ---------------------------------------------------------------------------
# 2. Activate conda env (starVLA — no GPU needed for download/convert)
# ---------------------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "starVLA" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate starVLA
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# 3. Paths  (training YAML expects lerobot data at playground/Datasets/...)
# ---------------------------------------------------------------------------
DATA_BASE="${DATA_BASE:-$(pwd)/playground/Datasets/RoboChallenge_table30v2}"
RAW_ROOT="${DATA_BASE}/raw"
LEROBOT_ROOT="${DATA_BASE}/lerobot"
mkdir -p "${RAW_ROOT}" "${LEROBOT_ROOT}"
echo "[info] RAW_ROOT    = ${RAW_ROOT}"
echo "[info] LEROBOT_ROOT= ${LEROBOT_ROOT}"

# ---------------------------------------------------------------------------
# 4. Query full task list from HF
# ---------------------------------------------------------------------------
echo "[init] Querying HF task list ..."
TASK_LIST=$(python - <<'PYEOF'
from examples.RoboChallenge_table30v2.train_files.download_table30v2 import list_task_parts
for t in sorted(list_task_parts()):
    print(t)
PYEOF
)

echo "[init] $(echo "${TASK_LIST}" | wc -l) tasks found on HF"

# Optional subset filter
if [[ -n "${ONLY_TASKS:-}" ]]; then
    echo "[init] Filtering to: ${ONLY_TASKS}"
    TASK_LIST=$(echo "${TASK_LIST}" | grep -xF -f <(echo "${ONLY_TASKS}" | tr ' ' '\n') || true)
    echo "[init] After filter: $(echo "${TASK_LIST}" | wc -l) tasks"
fi

# ---------------------------------------------------------------------------
# 5. Per-task: download → convert → cleanup
# ---------------------------------------------------------------------------
CONVERTED=()
SKIPPED_ALOHA=()
SKIPPED_DONE=()
FAILED=()

while IFS= read -r task; do
    [[ -z "${task}" ]] && continue
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━ ${task} ━━━━━━━━━━━━━━━━━━━━━━"

    # ---- Already converted? ----
    if [[ -f "${LEROBOT_ROOT}/${task}/meta/info.json" ]]; then
        echo "[skip][done] ${task} — lerobot/meta/info.json exists"
        SKIPPED_DONE+=("${task}")
        # Clean up any residual raw dir
        if [[ -d "${RAW_ROOT}/${task}" ]]; then
            # rm -rf "${RAW_ROOT}/${task}" # 因为用 hard link 的方式， 所以不用rm 了， 直接保留着
            echo "[cleanup] removed stale raw/${task}"
        fi
        continue
    fi

    # ---- Download + extract ----
    if [[ ! -f "${RAW_ROOT}/${task}/meta/task_info.json" ]]; then
        echo "[download] ${task} — starting"
        if ! python examples/RoboChallenge_table30v2/train_files/download_table30v2.py \
                --raw-root "${RAW_ROOT}" --only "${task}"; then
            echo "[FAIL][download] ${task}"
            FAILED+=("${task}")
            continue
        fi
        echo "[download][OK] ${task}"
    else
        echo "[download][skip] ${task} — raw/task_info.json exists, skipping download"
    fi

    # ---- Convert ----
    echo "[convert] ${task} — starting"
    CONV_LOG="${RAW_ROOT}/${task}_convert.log"
    if python examples/RoboChallenge_table30v2/train_files/convert_robochallenge_to_lerobot.py \
            --raw-root "${RAW_ROOT}" \
            --task "${task}" \
            --out-root "${LEROBOT_ROOT}" \
        >"${CONV_LOG}" 2>&1; then
        echo "[convert][OK] ${task}"
        CONVERTED+=("${task}")
        rm -f "${CONV_LOG}"
        # ---- Remove raw to free disk (skip ALOHA so we can re-explore) ----
        if [[ -f "${RAW_ROOT}/${task}/meta/task_info.json" ]] && \
           grep -q '"ALOHA"' "${RAW_ROOT}/${task}/meta/task_info.json"; then
            echo "[cleanup][keep] raw/${task} retained (ALOHA, KEEP_RAW=1)"
        else
            # rm -rf "${RAW_ROOT}/${task}" # rm -rf "${RAW_ROOT}/${task}" # 因为用 hard link 的方式， 所以不用rm 了， 直接保留着
            echo "[cleanup][OK] removed raw/${task}"
        fi
    else
        echo "[FAIL][convert] ${task} — see ${CONV_LOG}"
        cat "${CONV_LOG}" | tail -20
        FAILED+=("${task}")
    fi

done <<< "${TASK_LIST}"

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━ SUMMARY ━━━━━━━━━━━━━━━━━━━━━━"
echo "Already done   (${#SKIPPED_DONE[@]}): ${SKIPPED_DONE[*]:-—}"
echo "Newly converted(${#CONVERTED[@]}): ${CONVERTED[*]:-—}"
echo "Skipped (ALOHA)(${#SKIPPED_ALOHA[@]}): ${SKIPPED_ALOHA[*]:-—}"
echo "FAILED         (${#FAILED[@]}): ${FAILED[*]:-—}"
echo ""
echo "Lerobot tasks on disk:"
ls "${LEROBOT_ROOT}"

# ---------------------------------------------------------------------------
# 7. Auto-update data_config.py task-sets
# ---------------------------------------------------------------------------
echo ""
echo "[update] Rebuilding data_config.py task-sets from disk ..."
python examples/RoboChallenge_table30v2/train_files/update_data_config.py \
    --lerobot-root "${LEROBOT_ROOT}"

echo ""
echo "[DONE] $(date '+%Y-%m-%d %H:%M:%S')"
