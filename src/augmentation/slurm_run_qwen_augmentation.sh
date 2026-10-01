#!/usr/bin/env bash
# High-throughput Qwen3-Coder-Next augmentation job for four GPUs on the 8gpus partition.
#
# This does not submit itself. Submit it from the REvolution/src directory with:
#   mkdir -p slurm_log
#   sbatch augmentation/slurm_run_qwen_augmentation.sh
#
# The job reuses a cached Qwen description dataset when it exists; otherwise it
# writes/refreshes descriptions first. Existing descriptions are reused unless
# DESC_OVERWRITE_EXISTING=1 is exported.
#
# By default, source rows are read from data/dataset_v6.2.0_yosys and generated
# datasets/work artifacts are written under /work/u4523475/revolution-clean.
# The wrapper starts an OpenAI-compatible vLLM server inside the Slurm allocation
# and points both description prep and augmentation at that server.
#
# This wrapper currently defaults to START=1000,LIMIT=1000 so the next run
# augments the second 1000-row deterministic shard in the same output/work tree.
# To rerun the first shard explicitly:
#   sbatch --export=ALL,START=0,LIMIT=1000 augmentation/slurm_run_qwen_augmentation.sh
#
# To run a quick 100-sample dev test:
#   sbatch --partition=dev --time=02:00:00 --export=ALL,LIMIT=100 augmentation/slurm_run_qwen_augmentation.sh
#
# To skip description preparation and consume an already-described INPUT_DATASET:
#   sbatch --export=ALL,RUN_DESCRIPTION_PREP=0 augmentation/slurm_run_qwen_augmentation.sh
#
# To use a separate OpenAI-compatible vLLM server instead of starting one here:
#   sbatch --export=ALL,API_BACKEND=vllm,START_VLLM_SERVER=0,API_BASE_URL=http://localhost:8000/v1 augmentation/slurm_run_qwen_augmentation.sh
#
# To use vLLM's in-process offline API instead of an HTTP server:
#   sbatch --export=ALL,API_BACKEND=local-vllm,START_VLLM_SERVER=0 augmentation/slurm_run_qwen_augmentation.sh

#SBATCH --job-name=qwen3-coder-aug
#SBATCH --partition=8gpus
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --gpus-per-node=4
#SBATCH --output=slurm_log/slurm-%x-%j.out
#SBATCH --error=slurm_log/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/openroad_env/bin/python}"
if [[ ! -x "${PYTHON_BIN}" && -x "${HOME}/anaconda3/envs/openroad_env/bin/python" ]]; then
  PYTHON_BIN="${HOME}/anaconda3/envs/openroad_env/bin/python"
fi
PYTHON_ENV_BIN="$(cd -- "$(dirname -- "${PYTHON_BIN}")" && pwd)"
PYTHON_ENV_DIR="$(cd -- "${PYTHON_ENV_BIN}/.." && pwd)"
export CONDA_PREFIX="${CONDA_PREFIX:-${PYTHON_ENV_DIR}}"
export PATH="${PYTHON_ENV_BIN}:${PATH}"

WORK_DIR="${WORK_DIR:-/work/u4523475}"
RUN_DIR="${RUN_DIR:-${WORK_DIR}/revolution-clean/augmentation}"
RUN_NAME="${RUN_NAME:-qwen3_coder_next_v6.2.0_yosys_random_prompt_v2}"
HF_HOME="${HF_HOME:-${WORK_DIR}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_DIR}/.cache}"
TMPDIR="${TMPDIR:-${WORK_DIR}/tmp/${SLURM_JOB_ID:-manual}}"

SOURCE_DATASET="${SOURCE_DATASET:-data/dataset_v6.2.0_yosys}"
DESC_OUTPUT="${DESC_OUTPUT:-${RUN_DIR}/data/dataset_v6.2.0_yosys_qwen_desc}"
RUN_DESCRIPTION_PREP="${RUN_DESCRIPTION_PREP:-auto}"
DESC_INPUT="${DESC_INPUT:-${SOURCE_DATASET}}"
DESC_OVERWRITE_EXISTING="${DESC_OVERWRITE_EXISTING:-0}"

if [[ -n "${INPUT_DATASET:-}" ]]; then
  if [[ "${RUN_DESCRIPTION_PREP}" == "auto" ]]; then
    RUN_DESCRIPTION_PREP=0
  fi
elif [[ "${RUN_DESCRIPTION_PREP}" == "0" || "${RUN_DESCRIPTION_PREP}" == "false" ]]; then
  INPUT_DATASET="${DESC_OUTPUT}"
elif [[ "${RUN_DESCRIPTION_PREP}" == "auto" ]]; then
  if [[ "${DESC_OVERWRITE_EXISTING}" == "1" || "${DESC_OVERWRITE_EXISTING}" == "true" ]]; then
    RUN_DESCRIPTION_PREP=1
    INPUT_DATASET="${SOURCE_DATASET}"
  elif [[ -f "${DESC_OUTPUT}" || -f "${DESC_OUTPUT}/train.jsonl" || -f "${DESC_OUTPUT}/data.jsonl" ]]; then
    RUN_DESCRIPTION_PREP=0
    INPUT_DATASET="${DESC_OUTPUT}"
  else
    RUN_DESCRIPTION_PREP=1
    INPUT_DATASET="${SOURCE_DATASET}"
  fi
else
  INPUT_DATASET="${SOURCE_DATASET}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/data/${RUN_NAME}}"
AUG_WORK_DIR="${AUG_WORK_DIR:-${RUN_DIR}/outputs/${RUN_NAME}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-Next}"
API_BACKEND="${API_BACKEND:-vllm}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
START_VLLM_SERVER="${START_VLLM_SERVER:-1}"
API_BASE_URL="${API_BASE_URL:-}"
if [[ "${API_BACKEND}" == "vllm" && -z "${API_BASE_URL}" ]]; then
  API_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
fi

TARGETS="${TARGETS:-area power delay}"
TARGET_SELECTION_MODE="${TARGET_SELECTION_MODE:-random}"
NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
CANDIDATES_PER_GENERATION="${CANDIDATES_PER_GENERATION:-5}"
START="${START:-1000}"
LIMIT="${LIMIT:-1000}"
RANDOM_SEED="${RANDOM_SEED:-42}"
RESUME="${RESUME:-1}"

CONCURRENCY="${CONCURRENCY:-16}"
LLM_CONCURRENCY="${LLM_CONCURRENCY:-16}"
SYNTHESIS_CONCURRENCY="${SYNTHESIS_CONCURRENCY:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-600}"

LOCAL_VLLM_BATCH_SIZE="${LOCAL_VLLM_BATCH_SIZE:-16}"
LOCAL_VLLM_BATCH_TIMEOUT_MS="${LOCAL_VLLM_BATCH_TIMEOUT_MS:-50}"
LOCAL_VLLM_TENSOR_PARALLEL_SIZE="${LOCAL_VLLM_TENSOR_PARALLEL_SIZE:-${SLURM_GPUS_ON_NODE:-4}}"
LOCAL_VLLM_GPU_MEMORY_UTILIZATION="${LOCAL_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
LOCAL_VLLM_MAX_MODEL_LEN="${LOCAL_VLLM_MAX_MODEL_LEN:-32768}"
LOCAL_VLLM_MAX_NUM_BATCHED_TOKENS="${LOCAL_VLLM_MAX_NUM_BATCHED_TOKENS:-24576}"
LOCAL_VLLM_MAX_NUM_SEQS="${LOCAL_VLLM_MAX_NUM_SEQS:-16}"
LOCAL_VLLM_DTYPE="${LOCAL_VLLM_DTYPE:-auto}"
VLLM_SERVER_STARTUP_TIMEOUT_SECONDS="${VLLM_SERVER_STARTUP_TIMEOUT_SECONDS:-1800}"
VLLM_SERVER_LOG="${VLLM_SERVER_LOG:-${AUG_WORK_DIR}/vllm_server.log}"
VLLM_SERVER_EXTRA_ARGS="${VLLM_SERVER_EXTRA_ARGS:---disable-custom-all-reduce}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

YOSYS_BIN="${YOSYS_BIN:-yosys}"
OPENROAD_BIN="${OPENROAD_BIN:-openroad}"
CLOCK_PERIOD_NS="${CLOCK_PERIOD_NS:-0.01}"
SYNTH_TIMEOUT_SECONDS="${SYNTH_TIMEOUT_SECONDS:-300}"
SYNTHESIZE_MISSING_BASELINE="${SYNTHESIZE_MISSING_BASELINE:-0}"
SKIP_MISSING_SYNTHESIS_REPORT="${SKIP_MISSING_SYNTHESIS_REPORT:-1}"
MAX_SUMMARY_ERRORS="${MAX_SUMMARY_ERRORS:-2000}"
FILTER_SYNTHESIZABLE="${FILTER_SYNTHESIZABLE:-1}"
SYNTHESIZABLE_OUTPUT_DIR="${SYNTHESIZABLE_OUTPUT_DIR:-${OUTPUT_DIR}_synthesizable}"
SYNTHESIZABLE_SUMMARY="${SYNTHESIZABLE_SUMMARY:-${SYNTHESIZABLE_OUTPUT_DIR}/filter_summary.json}"

MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-47.5}"
SHUTDOWN_MARGIN_MINUTES="${SHUTDOWN_MARGIN_MINUTES:-15}"

DESC_MAX_TOKENS="${DESC_MAX_TOKENS:-1024}"
DESC_TEMPERATURE="${DESC_TEMPERATURE:-0.2}"
DESC_CONCURRENCY="${DESC_CONCURRENCY:-16}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME
export HF_HUB_CACHE
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export XDG_CACHE_HOME
export TMPDIR
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Submit this script with sbatch." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python environment not found: ${PYTHON_BIN}" >&2
  exit 3
fi
if ! command -v ninja >/dev/null 2>&1; then
  echo "ERROR: ninja is required by vLLM/FlashInfer JIT but was not found in PATH." >&2
  echo "Expected it in the Python environment bin directory: ${PYTHON_ENV_BIN}" >&2
  exit 4
fi
if [[ "${RUN_DESCRIPTION_PREP}" == "1" || "${RUN_DESCRIPTION_PREP}" == "true" ]]; then
  if [[ ! -e "${DESC_INPUT}" ]]; then
    echo "ERROR: Description input dataset does not exist: ${DESC_INPUT}" >&2
    exit 5
  fi
elif [[ ! -e "${INPUT_DATASET}" ]]; then
  echo "ERROR: Input dataset does not exist: ${INPUT_DATASET}" >&2
  exit 5
fi
if [[ "${API_BACKEND}" != "local-vllm" && "${API_BACKEND}" != "vllm" ]]; then
  echo "ERROR: Slurm augmentation script expects API_BACKEND=local-vllm or vllm." >&2
  exit 6
fi
if [[ "${API_BACKEND}" == "vllm" && -z "${API_BASE_URL}" ]]; then
  echo "ERROR: API_BASE_URL is required when API_BACKEND=vllm." >&2
  exit 7
fi
if [[ "${API_BACKEND}" != "vllm" && "${START_VLLM_SERVER}" == "1" ]]; then
  echo "ERROR: START_VLLM_SERVER=1 requires API_BACKEND=vllm." >&2
  exit 8
fi

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}" \
  "$(dirname "${OUTPUT_DIR}")" \
  "$(dirname "${DESC_OUTPUT}")" \
  "${SYNTHESIZABLE_OUTPUT_DIR}" \
  "${AUG_WORK_DIR}" \
  slurm_log

count_rows() {
  local dataset_path="$1"
  if [[ -f "${dataset_path}" ]]; then
    wc -l < "${dataset_path}"
  elif [[ -f "${dataset_path}/data.jsonl" ]]; then
    wc -l < "${dataset_path}/data.jsonl"
  elif [[ -f "${dataset_path}/train.jsonl" ]]; then
    wc -l < "${dataset_path}/train.jsonl"
  else
    echo "unknown"
  fi
}

echo "===== Qwen augmentation job ====="
echo "date=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "project_dir=${PROJECT_DIR}"
echo "python=${PYTHON_BIN}"
echo "python_env_bin=${PYTHON_ENV_BIN}"
echo "conda_prefix=${CONDA_PREFIX}"
echo "ninja=$(command -v ninja)"
echo "work_dir=${WORK_DIR}"
echo "run_dir=${RUN_DIR}"
echo "run_name=${RUN_NAME}"
echo "source_dataset=${SOURCE_DATASET}"
echo "input_dataset=${INPUT_DATASET}"
echo "input_rows=$(count_rows "${INPUT_DATASET}")"
echo "output_dir=${OUTPUT_DIR}"
echo "augmentation_work_dir=${AUG_WORK_DIR}"
echo "model=${MODEL_NAME}"
echo "api_backend=${API_BACKEND}"
echo "api_base_url=${API_BASE_URL:-unset}"
echo "start_vllm_server=${START_VLLM_SERVER}"
echo "vllm_host=${VLLM_HOST}"
echo "vllm_port=${VLLM_PORT}"
echo "vllm_server_log=${VLLM_SERVER_LOG}"
echo "vllm_engine_ready_timeout_s=${VLLM_ENGINE_READY_TIMEOUT_S}"
echo "vllm_server_startup_timeout_seconds=${VLLM_SERVER_STARTUP_TIMEOUT_SECONDS}"
echo "targets=${TARGETS}"
echo "target_selection_mode=${TARGET_SELECTION_MODE}"
echo "num_generations=${NUM_GENERATIONS}"
echo "candidates_per_generation=${CANDIDATES_PER_GENERATION}"
echo "start=${START}"
echo "limit=${LIMIT}"
echo "concurrency=${CONCURRENCY}"
echo "llm_concurrency=${LLM_CONCURRENCY}"
echo "synthesis_concurrency=${SYNTHESIS_CONCURRENCY}"
echo "local_vllm_batch_size=${LOCAL_VLLM_BATCH_SIZE}"
echo "local_vllm_tensor_parallel_size=${LOCAL_VLLM_TENSOR_PARALLEL_SIZE}"
echo "local_vllm_gpu_memory_utilization=${LOCAL_VLLM_GPU_MEMORY_UTILIZATION}"
echo "local_vllm_max_model_len=${LOCAL_VLLM_MAX_MODEL_LEN}"
echo "local_vllm_max_num_batched_tokens=${LOCAL_VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "local_vllm_max_num_seqs=${LOCAL_VLLM_MAX_NUM_SEQS}"
echo "max_runtime_hours=${MAX_RUNTIME_HOURS}"
echo "shutdown_margin_minutes=${SHUTDOWN_MARGIN_MINUTES}"
echo "skip_missing_synthesis_report=${SKIP_MISSING_SYNTHESIS_REPORT}"
echo "max_summary_errors=${MAX_SUMMARY_ERRORS}"
echo "filter_synthesizable=${FILTER_SYNTHESIZABLE}"
echo "synthesizable_output_dir=${SYNTHESIZABLE_OUTPUT_DIR}"
echo "synthesizable_summary=${SYNTHESIZABLE_SUMMARY}"
echo "omp_num_threads=${OMP_NUM_THREADS}"
echo "hf_home=${HF_HOME}"
echo "tmpdir=${TMPDIR}"
echo "run_description_prep=${RUN_DESCRIPTION_PREP}"
echo "description_input=${DESC_INPUT}"
echo "description_output=${DESC_OUTPUT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

VLLM_SERVER_PID=""
cleanup() {
  if [[ -n "${VLLM_SERVER_PID}" ]] && kill -0 "${VLLM_SERVER_PID}" 2>/dev/null; then
    echo "Stopping vLLM server pid=${VLLM_SERVER_PID}"
    kill "${VLLM_SERVER_PID}" 2>/dev/null || true
    wait "${VLLM_SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

vllm_is_ready() {
  local models_url="${API_BASE_URL%/}/models"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "${models_url}" >/dev/null 2>&1
    return $?
  fi
  "${PYTHON_BIN}" -c '
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
        raise SystemExit(0 if response.status < 500 else 1)
except Exception:
    raise SystemExit(1)
' "${models_url}"
}

start_vllm_server() {
  local -a server_args=(
    -m vllm.entrypoints.openai.api_server
    --model "${MODEL_NAME}"
    --served-model-name "${MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --tensor-parallel-size "${LOCAL_VLLM_TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${LOCAL_VLLM_GPU_MEMORY_UTILIZATION}"
    --max-num-seqs "${LOCAL_VLLM_MAX_NUM_SEQS}"
    --dtype "${LOCAL_VLLM_DTYPE}"
  )
  if [[ -n "${LOCAL_VLLM_MAX_MODEL_LEN}" ]]; then
    server_args+=(--max-model-len "${LOCAL_VLLM_MAX_MODEL_LEN}")
  fi
  if [[ -n "${LOCAL_VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
    server_args+=(--max-num-batched-tokens "${LOCAL_VLLM_MAX_NUM_BATCHED_TOKENS}")
  fi
  if [[ "${TRUST_REMOTE_CODE}" == "1" || "${TRUST_REMOTE_CODE}" == "true" ]]; then
    server_args+=(--trust-remote-code)
  fi
  if [[ "${ENABLE_PREFIX_CACHING}" == "1" || "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
    server_args+=(--enable-prefix-caching)
  fi
  if [[ -n "${VLLM_SERVER_EXTRA_ARGS}" ]]; then
    local -a extra_args=()
    read -r -a extra_args <<< "${VLLM_SERVER_EXTRA_ARGS}"
    server_args+=("${extra_args[@]}")
  fi

  echo "===== Starting vLLM server ====="
  echo "vllm_command=${PYTHON_BIN} ${server_args[*]}"
  "${PYTHON_BIN}" "${server_args[@]}" >"${VLLM_SERVER_LOG}" 2>&1 &
  VLLM_SERVER_PID="$!"
  echo "vllm_server_pid=${VLLM_SERVER_PID}"

  local deadline=$((SECONDS + VLLM_SERVER_STARTUP_TIMEOUT_SECONDS))
  until vllm_is_ready; do
    if ! kill -0 "${VLLM_SERVER_PID}" 2>/dev/null; then
      echo "ERROR: vLLM server exited before becoming ready." >&2
      tail -n 80 "${VLLM_SERVER_LOG}" >&2 || true
      exit 9
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: vLLM server did not become ready within ${VLLM_SERVER_STARTUP_TIMEOUT_SECONDS}s." >&2
      tail -n 80 "${VLLM_SERVER_LOG}" >&2 || true
      exit 10
    fi
    sleep 5
  done
  echo "vllm_server_ready=${API_BASE_URL%/}/models"
}

if [[ "${API_BACKEND}" == "vllm" && ("${START_VLLM_SERVER}" == "1" || "${START_VLLM_SERVER}" == "true") ]]; then
  start_vllm_server
fi

common_llm_args=(
  --api-backend "${API_BACKEND}"
  --model-name "${MODEL_NAME}"
  --request-timeout-seconds "${REQUEST_TIMEOUT_SECONDS}"
  --local-vllm-batch-size "${LOCAL_VLLM_BATCH_SIZE}"
  --local-vllm-batch-timeout-ms "${LOCAL_VLLM_BATCH_TIMEOUT_MS}"
  --local-vllm-tensor-parallel-size "${LOCAL_VLLM_TENSOR_PARALLEL_SIZE}"
  --local-vllm-gpu-memory-utilization "${LOCAL_VLLM_GPU_MEMORY_UTILIZATION}"
  --local-vllm-max-num-seqs "${LOCAL_VLLM_MAX_NUM_SEQS}"
  --local-vllm-dtype "${LOCAL_VLLM_DTYPE}"
)
if [[ -n "${API_BASE_URL}" ]]; then
  common_llm_args+=(--api-base-url "${API_BASE_URL}")
fi
if [[ -n "${LOCAL_VLLM_MAX_MODEL_LEN}" ]]; then
  common_llm_args+=(--local-vllm-max-model-len "${LOCAL_VLLM_MAX_MODEL_LEN}")
fi
if [[ -n "${LOCAL_VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
  common_llm_args+=(
    --local-vllm-max-num-batched-tokens
    "${LOCAL_VLLM_MAX_NUM_BATCHED_TOKENS}"
  )
fi
if [[ "${TRUST_REMOTE_CODE}" != "1" && "${TRUST_REMOTE_CODE}" != "true" ]]; then
  common_llm_args+=(--no-trust-remote-code)
fi
if [[ "${ENABLE_PREFIX_CACHING}" != "1" && "${ENABLE_PREFIX_CACHING}" != "true" ]]; then
  common_llm_args+=(--disable-prefix-caching)
fi

if [[ "${RUN_DESCRIPTION_PREP}" == "1" || "${RUN_DESCRIPTION_PREP}" == "true" ]]; then
  desc_args=(
    --input "${DESC_INPUT}"
    --output "${DESC_OUTPUT}"
    --temperature "${DESC_TEMPERATURE}"
    --top-p "${TOP_P}"
    --max-tokens "${DESC_MAX_TOKENS}"
    --concurrency "${DESC_CONCURRENCY}"
    "${common_llm_args[@]}"
  )
  if [[ "${DESC_OVERWRITE_EXISTING}" == "1" || "${DESC_OVERWRITE_EXISTING}" == "true" ]]; then
    desc_args+=(--overwrite-existing)
  fi
  echo "===== Preparing descriptions with Qwen ====="
  "${PYTHON_BIN}" -m augmentation.prepare_dataset_description "${desc_args[@]}"
  INPUT_DATASET="${DESC_OUTPUT}"
  echo "description_output=${INPUT_DATASET}"
  echo "description_rows=$(count_rows "${INPUT_DATASET}")"
fi

augmentation_args=(
  --input "${INPUT_DATASET}"
  --output "${OUTPUT_DIR}"
  --work-dir "${AUG_WORK_DIR}"
  --targets ${TARGETS}
  --target-selection-mode "${TARGET_SELECTION_MODE}"
  --random-seed "${RANDOM_SEED}"
  --num-generations "${NUM_GENERATIONS}"
  --candidates-per-generation "${CANDIDATES_PER_GENERATION}"
  --concurrency "${CONCURRENCY}"
  --llm-concurrency "${LLM_CONCURRENCY}"
  --synthesis-concurrency "${SYNTHESIS_CONCURRENCY}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-tokens "${MAX_TOKENS}"
  --yosys-bin "${YOSYS_BIN}"
  --openroad-bin "${OPENROAD_BIN}"
  --clock-period-ns "${CLOCK_PERIOD_NS}"
  --synth-timeout-seconds "${SYNTH_TIMEOUT_SECONDS}"
  --max-runtime-hours "${MAX_RUNTIME_HOURS}"
  --shutdown-margin-minutes "${SHUTDOWN_MARGIN_MINUTES}"
  --max-summary-errors "${MAX_SUMMARY_ERRORS}"
  "${common_llm_args[@]}"
)
if [[ -n "${START}" && "${START}" != "none" ]]; then
  augmentation_args+=(--start "${START}")
fi
if [[ -n "${LIMIT}" && "${LIMIT}" != "all" && "${LIMIT}" != "none" ]]; then
  augmentation_args+=(--limit "${LIMIT}")
fi
if [[ "${RESUME}" != "1" && "${RESUME}" != "true" ]]; then
  augmentation_args+=(--no-resume)
fi
if [[ "${SYNTHESIZE_MISSING_BASELINE}" == "1" || "${SYNTHESIZE_MISSING_BASELINE}" == "true" ]]; then
  augmentation_args+=(--synthesize-missing-baseline)
fi
if [[ "${SKIP_MISSING_SYNTHESIS_REPORT}" == "1" || "${SKIP_MISSING_SYNTHESIS_REPORT}" == "true" ]]; then
  augmentation_args+=(--skip-missing-synthesis-report)
fi

echo "===== Running augmentation ====="
"${PYTHON_BIN}" -m augmentation.run_data_augmentation "${augmentation_args[@]}"

if [[ "${FILTER_SYNTHESIZABLE}" == "1" || "${FILTER_SYNTHESIZABLE}" == "true" ]]; then
  echo "===== Filtering synthesizable augmentation rows ====="
  "${PYTHON_BIN}" - "${OUTPUT_DIR}/data.jsonl" "${SYNTHESIZABLE_OUTPUT_DIR}" "${SYNTHESIZABLE_SUMMARY}" <<'PY'
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


source = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
output_path = output_dir / "data.jsonl"

if not source.exists():
    raise SystemExit(f"ERROR: Raw augmentation JSONL does not exist: {source}")
if source.parent.resolve() == output_dir.resolve():
    raise SystemExit("ERROR: Synthesizable output directory must differ from raw output directory.")

output_dir.mkdir(parents=True, exist_ok=True)
summary_path.parent.mkdir(parents=True, exist_ok=True)
tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")

total = 0
kept = 0
dropped = 0
by_target: dict[str, Counter[str]] = defaultdict(Counter)
by_generation: dict[str, Counter[str]] = defaultdict(Counter)
dropped_errors: Counter[str] = Counter()

with source.open("r", encoding="utf-8") as fin, tmp_output.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        target = str(row.get("target"))
        generation = str(row.get("generation"))
        if row.get("synthesis_success") is True:
            kept += 1
            by_target[target]["kept"] += 1
            by_generation[generation]["kept"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            dropped += 1
            by_target[target]["dropped"] += 1
            by_generation[generation]["dropped"] += 1
            dropped_errors[str(row.get("error") or "NO_ERROR_FIELD")] += 1

os.replace(tmp_output, output_path)

manifest_path = output_dir / "dataset_info.json"
manifest_path.write_text(
    json.dumps(
        {
            "format": "jsonl",
            "data_file": output_path.name,
            "source": str(source),
            "filter": "synthesis_success == true",
            "num_records": kept,
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

summary = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "input": str(source),
    "output": str(output_path),
    "manifest": str(manifest_path),
    "filter": "synthesis_success == true",
    "total_input_rows": total,
    "kept_rows": kept,
    "dropped_rows": dropped,
    "kept_rate": kept / total if total else 0.0,
    "by_target": {key: dict(value) for key, value in sorted(by_target.items())},
    "by_generation": {
        key: dict(value)
        for key, value in sorted(
            by_generation.items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else 999999,
        )
    },
    "dropped_error_counts": dict(dropped_errors.most_common()),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

print(f"synthesizable_input={source}")
print(f"synthesizable_output={output_path}")
print(f"synthesizable_summary={summary_path}")
print(f"synthesizable_total={total}")
print(f"synthesizable_kept={kept}")
print(f"synthesizable_dropped={dropped}")
PY
fi
