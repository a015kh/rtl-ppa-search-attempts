#!/usr/bin/env bash
# Qwen3-Coder-Next original REvolution experiment using run_evolution.py.
#
# This script does not submit itself. From the REvolution/src directory:
#   mkdir -p slurm_log
#   sbatch slurm_run_qwen_original_evolution.sh
#
# It starts an OpenAI-compatible vLLM server inside the Slurm allocation and
# points run_evolution.py at that server. The default vLLM port is 8001, separate
# from the augmentation wrapper's default port 8000.
#
# Useful overrides:
#   sbatch --export=ALL,VLLM_PORT=8011 slurm_run_qwen_original_evolution.sh
#   sbatch --export=ALL,NUM_WORKERS=4,PROBLEMS_FILE=problems_count_50.txt slurm_run_qwen_original_evolution.sh
#   sbatch --export=ALL,START_VLLM_SERVER=0,API_BASE_URL=http://127.0.0.1:8001/v1 slurm_run_qwen_original_evolution.sh

#SBATCH --job-name=qwen3-coder-eoh
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
if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd -- "${PROJECT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_evolution.py" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SCRIPT_DIR}/run_evolution.py" ]]; then
  PROJECT_DIR="${SCRIPT_DIR}"
else
  echo "ERROR: Could not locate the REvolution src directory. Submit from src or set PROJECT_DIR." >&2
  exit 1
fi
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
RUN_DIR="${RUN_DIR:-${WORK_DIR}/revolution-clean/original_evolution}"
RUN_NAME="${RUN_NAME:-qwen3_coder_next_original_evolution_verilogeval_spec_to_rtl}"
SAVE_PATH="${SAVE_PATH:-${RUN_DIR}/exp/${RUN_NAME}}"

HF_HOME="${HF_HOME:-${WORK_DIR}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_DIR}/.cache}"
TMPDIR="${TMPDIR:-${WORK_DIR}/tmp/${SLURM_JOB_ID:-manual}}"

BENCHMARK_ROOT="${PROJECT_DIR}/../bench"
BENCHMARKS="${BENCHMARKS:-VerilogEval-Spec-to-RTL}"
PROBLEMS_FILE="${PROBLEMS_FILE:-problems.txt}"
PROBLEMS="${PROBLEMS:-}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Coder-Next}"
API_BACKEND="${API_BACKEND:-vllm}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8001}"
START_VLLM_SERVER="${START_VLLM_SERVER:-1}"
API_BASE_URL="${API_BASE_URL:-}"
if [[ "${API_BACKEND}" == "vllm" && -z "${API_BASE_URL}" ]]; then
  API_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
fi

POPULATION_SIZE="${POPULATION_SIZE:-10}"
NUM_GENERATIONS="${NUM_GENERATIONS:-20}"
NUM_WORKERS="${NUM_WORKERS:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
STRATEGY_SELECTION="${STRATEGY_SELECTION:-random}"
EPSILON="${EPSILON:-0.1}"
UCB_C="${UCB_C:-2.0}"
GENERATION_MODE="${GENERATION_MODE:-whole}"
START_FROM_REFERENCE="${START_FROM_REFERENCE:-0}"

SYNTHESIS_EVALUATOR="${SYNTHESIS_EVALUATOR:-synthesis}"
PROXY_CHECKPOINT="${PROXY_CHECKPOINT:-}"
PROXY_ENCODER="${PROXY_ENCODER:-google/t5gemma-s-s-prefixlm}"
PROXY_DEVICE="${PROXY_DEVICE:-}"
PROXY_INPUT_PREPROCESS="${PROXY_INPUT_PREPROCESS:-none}"
PROXY_YOSYS_BIN="${PROXY_YOSYS_BIN:-yosys}"
PROXY_YOSYS_TIMEOUT_SECONDS="${PROXY_YOSYS_TIMEOUT_SECONDS:-120}"
PROXY_YOSYS_PDK="${PROXY_YOSYS_PDK:-nangate45}"
METRICS="${METRICS:-area delay static_power switching_power}"
DEBUG_BYPASS_PATH="${DEBUG_BYPASS_PATH:-}"
EXTRA_RUN_EVOLUTION_ARGS="${EXTRA_RUN_EVOLUTION_ARGS:-}"

LOCAL_VLLM_TENSOR_PARALLEL_SIZE="${LOCAL_VLLM_TENSOR_PARALLEL_SIZE:-${SLURM_GPUS_ON_NODE:-4}}"
LOCAL_VLLM_GPU_MEMORY_UTILIZATION="${LOCAL_VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
LOCAL_VLLM_MAX_MODEL_LEN="${LOCAL_VLLM_MAX_MODEL_LEN:-32768}"
LOCAL_VLLM_MAX_NUM_BATCHED_TOKENS="${LOCAL_VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
LOCAL_VLLM_MAX_NUM_SEQS="${LOCAL_VLLM_MAX_NUM_SEQS:-64}"
LOCAL_VLLM_DTYPE="${LOCAL_VLLM_DTYPE:-auto}"
VLLM_SERVER_STARTUP_TIMEOUT_SECONDS="${VLLM_SERVER_STARTUP_TIMEOUT_SECONDS:-1800}"
VLLM_SERVER_LOG="${VLLM_SERVER_LOG:-${SAVE_PATH}/vllm_server.log}"
VLLM_SERVER_EXTRA_ARGS="${VLLM_SERVER_EXTRA_ARGS:---disable-custom-all-reduce}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-600}"

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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export REVOLUTION_VLLM_BASE_URL="${API_BASE_URL}"
export VLLM_BASE_URL="${API_BASE_URL}"
export REVOLUTION_LLM_TIMEOUT_SECONDS="${REVOLUTION_LLM_TIMEOUT_SECONDS:-${REQUEST_TIMEOUT_SECONDS}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

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
if [[ "${API_BACKEND}" != "vllm" ]]; then
  echo "ERROR: This script expects API_BACKEND=vllm." >&2
  exit 5
fi
if [[ -z "${API_BASE_URL}" ]]; then
  echo "ERROR: API_BASE_URL is required when API_BACKEND=vllm." >&2
  exit 6
fi
if [[ "${START_VLLM_SERVER}" != "1" && "${START_VLLM_SERVER}" != "true" && "${START_VLLM_SERVER}" != "0" && "${START_VLLM_SERVER}" != "false" ]]; then
  echo "ERROR: START_VLLM_SERVER must be 1/true or 0/false." >&2
  exit 7
fi
if [[ "${SYNTHESIS_EVALUATOR}" == "proxy" && -z "${PROXY_CHECKPOINT}" ]]; then
  echo "ERROR: PROXY_CHECKPOINT is required when SYNTHESIS_EVALUATOR=proxy." >&2
  exit 8
fi

read -r -a benchmark_args <<< "${BENCHMARKS}"
for benchmark in "${benchmark_args[@]}"; do
  benchmark_dir="${BENCHMARK_ROOT}/${benchmark}"
  if [[ ! -d "${benchmark_dir}" ]]; then
    echo "ERROR: Benchmark directory does not exist: ${benchmark_dir}" >&2
    exit 9
  fi
  if [[ ! -f "${benchmark_dir}/${PROBLEMS_FILE}" ]]; then
    echo "ERROR: Problems file does not exist: ${benchmark_dir}/${PROBLEMS_FILE}" >&2
    exit 10
  fi
done

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}" \
  "${SAVE_PATH}" \
  "$(dirname "${VLLM_SERVER_LOG}")" \
  slurm_log

count_problems() {
  local total=0
  local benchmark
  for benchmark in "${benchmark_args[@]}"; do
    local problems_path="${BENCHMARK_ROOT}/${benchmark}/${PROBLEMS_FILE}"
    local count
    count="$(wc -l < "${problems_path}")"
    total=$((total + count))
  done
  echo "${total}"
}

echo "===== Qwen original evolution job ====="
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
echo "save_path=${SAVE_PATH}"
echo "benchmark_root=${BENCHMARK_ROOT}"
echo "benchmarks=${BENCHMARKS}"
echo "problems_file=${PROBLEMS_FILE}"
echo "problems_override=${PROBLEMS:-unset}"
echo "problems_in_file=$(count_problems)"
echo "model=${MODEL_NAME}"
echo "api_backend=${API_BACKEND}"
echo "api_base_url=${API_BASE_URL}"
echo "start_vllm_server=${START_VLLM_SERVER}"
echo "vllm_host=${VLLM_HOST}"
echo "vllm_port=${VLLM_PORT}"
echo "vllm_server_log=${VLLM_SERVER_LOG}"
echo "vllm_engine_ready_timeout_s=${VLLM_ENGINE_READY_TIMEOUT_S}"
echo "vllm_server_startup_timeout_seconds=${VLLM_SERVER_STARTUP_TIMEOUT_SECONDS}"
echo "local_vllm_tensor_parallel_size=${LOCAL_VLLM_TENSOR_PARALLEL_SIZE}"
echo "local_vllm_max_model_len=${LOCAL_VLLM_MAX_MODEL_LEN}"
echo "local_vllm_max_num_batched_tokens=${LOCAL_VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "local_vllm_max_num_seqs=${LOCAL_VLLM_MAX_NUM_SEQS}"
echo "population_size=${POPULATION_SIZE}"
echo "num_generations=${NUM_GENERATIONS}"
echo "num_workers=${NUM_WORKERS}"
echo "temperature=${TEMPERATURE}"
echo "top_p=${TOP_P}"
echo "max_tokens=${MAX_TOKENS}"
echo "strategy_selection=${STRATEGY_SELECTION}"
echo "generation_mode=${GENERATION_MODE}"
echo "synthesis_evaluator=${SYNTHESIS_EVALUATOR}"
echo "proxy_input_preprocess=${PROXY_INPUT_PREPROCESS}"
echo "proxy_yosys_bin=${PROXY_YOSYS_BIN}"
echo "proxy_yosys_timeout_seconds=${PROXY_YOSYS_TIMEOUT_SECONDS}"
echo "proxy_yosys_pdk=${PROXY_YOSYS_PDK}"
echo "request_timeout_seconds=${REVOLUTION_LLM_TIMEOUT_SECONDS}"
echo "hf_home=${HF_HOME}"
echo "tmpdir=${TMPDIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

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
      exit 11
    fi
    if (( SECONDS >= deadline )); then
      echo "ERROR: vLLM server did not become ready within ${VLLM_SERVER_STARTUP_TIMEOUT_SECONDS}s." >&2
      tail -n 80 "${VLLM_SERVER_LOG}" >&2 || true
      exit 12
    fi
    sleep 5
  done
  echo "vllm_server_ready=${API_BASE_URL%/}/models"
}

if [[ "${START_VLLM_SERVER}" == "1" || "${START_VLLM_SERVER}" == "true" ]]; then
  start_vllm_server
elif ! vllm_is_ready; then
  echo "ERROR: START_VLLM_SERVER=0 but no vLLM server is ready at ${API_BASE_URL%/}/models." >&2
  exit 13
fi

evolution_args=(
  --benchmarks "${benchmark_args[@]}"
  --api_backend "${API_BACKEND}"
  --model_name "${MODEL_NAME}"
  --population_size "${POPULATION_SIZE}"
  --num_generations "${NUM_GENERATIONS}"
  --save_path "${SAVE_PATH}"
  --num_workers "${NUM_WORKERS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max_tokens "${MAX_TOKENS}"
  --strategy_selection "${STRATEGY_SELECTION}"
  --epsilon "${EPSILON}"
  --ucb_c "${UCB_C}"
  --generation_mode "${GENERATION_MODE}"
  --synthesis_evaluator "${SYNTHESIS_EVALUATOR}"
  --problems_file "${PROBLEMS_FILE}"
)

if [[ -n "${PROBLEMS}" ]]; then
  read -r -a selected_problems <<< "${PROBLEMS}"
  evolution_args+=(--problems "${selected_problems[@]}")
fi
if [[ "${START_FROM_REFERENCE}" == "1" || "${START_FROM_REFERENCE}" == "true" ]]; then
  evolution_args+=(--start_from_reference)
fi
if [[ "${SYNTHESIS_EVALUATOR}" == "proxy" ]]; then
  evolution_args+=(--proxy_checkpoint "${PROXY_CHECKPOINT}")
  evolution_args+=(--proxy_encoder "${PROXY_ENCODER}")
  if [[ -n "${PROXY_DEVICE}" ]]; then
    evolution_args+=(--proxy_device "${PROXY_DEVICE}")
  fi
  evolution_args+=(--proxy_input_preprocess "${PROXY_INPUT_PREPROCESS}")
  evolution_args+=(--proxy_yosys_bin "${PROXY_YOSYS_BIN}")
  evolution_args+=(--proxy_yosys_timeout_seconds "${PROXY_YOSYS_TIMEOUT_SECONDS}")
  evolution_args+=(--proxy_yosys_pdk "${PROXY_YOSYS_PDK}")
  read -r -a metric_args <<< "${METRICS}"
  evolution_args+=(--metrics "${metric_args[@]}")
fi
if [[ -n "${DEBUG_BYPASS_PATH}" ]]; then
  evolution_args+=(--debug_bypass_path "${DEBUG_BYPASS_PATH}")
fi
if [[ -n "${EXTRA_RUN_EVOLUTION_ARGS}" ]]; then
  read -r -a extra_run_args <<< "${EXTRA_RUN_EVOLUTION_ARGS}"
  evolution_args+=("${extra_run_args[@]}")
fi

echo "===== Running original evolution ====="
echo "run_evolution_command=${PYTHON_BIN} run_evolution.py ${evolution_args[*]}"
"${PYTHON_BIN}" run_evolution.py "${evolution_args[@]}"
