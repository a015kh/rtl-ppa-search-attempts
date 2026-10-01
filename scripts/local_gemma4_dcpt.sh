#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-run}"

VLLM_ENV="${VLLM_ENV:-${REPO_ROOT}/.venv-vllm}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM_ENV}/bin/python}"
PIPELINE_PYTHON="${PIPELINE_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

VLLM_VERSION="${VLLM_VERSION:-0.20.0}"
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v0.20.0/vllm-0.20.0-cp38-abi3-manylinux_2_35_x86_64.whl}"
VLLM_WHEEL_SHA256="${VLLM_WHEEL_SHA256:-24d28892e210200f6e1bd13f699c42a74cd2bb7364c11248e2348f677c7f6dfb}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.5.3}"

MODEL_SPEC="${MODEL_SPEC:-hf:LilaRest/gemma-4-31B-it-NVFP4-turbo}"
MODEL_NAME="${MODEL_SPEC#hf:}"
MODEL_REVISION="${MODEL_REVISION:-782d58ea7f9cf3a148be86e153f5689a2d73238b}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
API_BASE_URL="${API_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
START_VLLM_SERVER="${START_VLLM_SERVER:-1}"
KEEP_VLLM_SERVER="${KEEP_VLLM_SERVER:-0}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-3600}"

BENCHMARK="${BENCHMARK:-VerilogEval-Spec-to-RTL-DC-opt}"
PROBLEMS="${PROBLEMS:-all}"
PROBLEMS_FILE="${PROBLEMS_FILE:-problem_large_but_solvable.txt}"
POPULATION_SIZE="${POPULATION_SIZE:-10}"
NUM_GENERATIONS="${NUM_GENERATIONS:-20}"
NUM_WORKERS="${NUM_WORKERS:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
STRATEGY_SELECTION="${STRATEGY_SELECTION:-ucb}"
EPSILON="${EPSILON:-0.1}"
UCB_C="${UCB_C:-2.0}"
GENERATION_MODE="${GENERATION_MODE:-whole}"
START_FROM_REFERENCE="${START_FROM_REFERENCE:-0}"
BEST_RECORD_REQUIRE_CONSTRAINTS="${BEST_RECORD_REQUIRE_CONSTRAINTS:-0}"
DCPT_CLOCKS="${DCPT_CLOCKS:-}"
DCPT_CONSTRAINT_TOLERANCE_NS="${DCPT_CONSTRAINT_TOLERANCE_NS:-0.000001}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/output/local_gemma4_dcpt/${RUN_STAMP}}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

prepare_python_headers() {
  if [[ -f /usr/include/python3.12/Python.h ]]; then
    return
  fi
  local root="${REPO_ROOT}/.cache/python3.12-dev"
  local header="${root}/usr/include/python3.12/Python.h"
  if [[ ! -f "${header}" ]]; then
    command -v apt-get >/dev/null 2>&1 || die "apt-get is required to download Python headers."
    command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb is required to extract Python headers."
    mkdir -p "${root}/packages"
    (cd "${root}/packages" && apt-get download libpython3.12-dev)
    local package
    package="$(find "${root}/packages" -maxdepth 1 -name 'libpython3.12-dev_*.deb' -print -quit)"
    [[ -n "${package}" ]] || die "Failed to download libpython3.12-dev."
    dpkg-deb -x "${package}" "${root}"
  fi
  export C_INCLUDE_PATH="${root}/usr/include/python3.12:${root}/usr/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
}

setup_runtime() {
  prepare_python_headers
  [[ -n "${UV_BIN}" ]] || die "uv is required to create the isolated vLLM environment."
  command -v curl >/dev/null 2>&1 || die "curl is required to download the pinned vLLM wheel."
  local cache_dir="${REPO_ROOT}/.cache/vllm-wheels"
  local wheel="${cache_dir}/vllm-${VLLM_VERSION}-cp38-abi3-manylinux_2_35_x86_64.whl"
  mkdir -p "${cache_dir}"
  if [[ ! -f "${wheel}" ]] || ! echo "${VLLM_WHEEL_SHA256}  ${wheel}" | sha256sum -c - >/dev/null 2>&1; then
    curl -fL --retry 3 --output "${wheel}.partial" "${VLLM_WHEEL_URL}"
    echo "${VLLM_WHEEL_SHA256}  ${wheel}.partial" | sha256sum -c -
    mv "${wheel}.partial" "${wheel}"
  fi
  "${UV_BIN}" venv --python /usr/bin/python3.12 --clear "${VLLM_ENV}"
  "${UV_BIN}" pip install --python "${VLLM_PYTHON}" \
    "${wheel}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    nvidia-cuda-nvcc==13.0.88 \
    nvidia-cuda-crt==13.0.88 \
    nvidia-cuda-runtime==13.0.96 \
    nvidia-nvvm==13.0.88 \
    nvidia-cuda-cccl==13.0.85
  local cuda_home
  cuda_home="$("${VLLM_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')/nvidia/cu13"
  [[ -x "${VLLM_ENV}/bin/ninja" ]] || die "The vLLM environment is missing ninja."
  [[ -x "${cuda_home}/bin/nvcc" ]] || die "The vLLM environment is missing nvcc."
  [[ -f "${cuda_home}/include/cccl/cuda/std/utility" ]] || die "The vLLM environment is missing CUDA CCCL headers."
  [[ -e "${cuda_home}/lib64" ]] || ln -s lib "${cuda_home}/lib64"
  [[ -f "${cuda_home}/lib64/libcudart.so.13" ]] || die "The vLLM environment is missing the CUDA runtime library."
  [[ -e "${cuda_home}/lib/libcudart.so" ]] || ln -s libcudart.so.13 "${cuda_home}/lib/libcudart.so"
  "${VLLM_PYTHON}" - <<'PY'
import torch
import transformers
import vllm

assert vllm.__version__ == "0.20.0", vllm.__version__
assert transformers.__version__ == "5.5.3", transformers.__version__
assert torch.cuda.is_available(), "CUDA is not available"
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) == (12, 0), (major, minor)
print(f"vllm={vllm.__version__} transformers={transformers.__version__} torch={torch.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)} capability={major}.{minor} cuda={torch.version.cuda}")
PY
}

if [[ "${ACTION}" == "setup" ]]; then
  setup_runtime
  exit 0
fi
[[ "${ACTION}" == "run" ]] || die "Usage: $0 [setup|run]"

[[ -x "${VLLM_PYTHON}" ]] || die "Missing vLLM environment. Run '$0 setup' first."
prepare_python_headers
CUDA_HOME="${CUDA_HOME:-$("${VLLM_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')/nvidia/cu13}"
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || die "Missing CUDA compiler: ${CUDA_HOME}/bin/nvcc"
[[ -e "${CUDA_HOME}/lib64" ]] || ln -s lib "${CUDA_HOME}/lib64"
[[ -f "${CUDA_HOME}/lib64/libcudart.so.13" ]] || die "Missing CUDA runtime library under ${CUDA_HOME}/lib64."
[[ -e "${CUDA_HOME}/lib/libcudart.so" ]] || ln -s libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so"
export CUDA_HOME
export PATH="${VLLM_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
[[ -x "${PIPELINE_PYTHON}" ]] || die "Missing project Python: ${PIPELINE_PYTHON}"
command -v curl >/dev/null 2>&1 || die "curl is required."
command -v jq >/dev/null 2>&1 || die "jq is required."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required."

mkdir -p "${SAVE_PATH}"
VLLM_SERVER_LOG="${VLLM_SERVER_LOG:-${SAVE_PATH}/vllm_server.log}"
PIPELINE_LOG="${PIPELINE_LOG:-${SAVE_PATH}/pipeline.log}"
RUN_CONFIG="${SAVE_PATH}/run_config.txt"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export REVOLUTION_VLLM_BASE_URL="${API_BASE_URL}"
export REVOLUTION_LLM_TIMEOUT_SECONDS="${REVOLUTION_LLM_TIMEOUT_SECONDS:-600}"

{
  echo "date=$(date --iso-8601=seconds)"
  echo "repo_root=${REPO_ROOT}"
  echo "vllm_python=${VLLM_PYTHON}"
  echo "vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}"
  echo "pipeline_python=${PIPELINE_PYTHON}"
  echo "model=${MODEL_NAME}"
  echo "model_revision=${MODEL_REVISION}"
  echo "api_base_url=${API_BASE_URL}"
  echo "benchmark=${BENCHMARK}"
  echo "problems=${PROBLEMS}"
  echo "population_size=${POPULATION_SIZE}"
  echo "num_generations=${NUM_GENERATIONS}"
  echo "num_workers=${NUM_WORKERS}"
  echo "temperature=${TEMPERATURE}"
  echo "top_p=${TOP_P}"
  echo "max_tokens=${MAX_TOKENS}"
  echo "strategy_selection=${STRATEGY_SELECTION}"
  echo "epsilon=${EPSILON}"
  echo "ucb_c=${UCB_C}"
  echo "generation_mode=${GENERATION_MODE}"
  echo "start_from_reference=${START_FROM_REFERENCE}"
  echo "best_record_require_constraints=${BEST_RECORD_REQUIRE_CONSTRAINTS}"
  echo "dcpt_clocks=${DCPT_CLOCKS:-optimization clock only}"
  echo "dcpt_constraint_tolerance_ns=${DCPT_CONSTRAINT_TOLERANCE_NS}"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader
  "${VLLM_PYTHON}" -c 'import torch, transformers, vllm; print(f"vllm={vllm.__version__} transformers={transformers.__version__} torch={torch.__version__} cuda={torch.version.cuda}")'
} | tee "${RUN_CONFIG}"

VLLM_SERVER_PID=""
cleanup() {
  if [[ -n "${VLLM_SERVER_PID}" ]] && kill -0 "${VLLM_SERVER_PID}" 2>/dev/null; then
    if [[ "${KEEP_VLLM_SERVER}" == "1" || "${KEEP_VLLM_SERVER}" == "true" ]]; then
      echo "Keeping vLLM server pid=${VLLM_SERVER_PID}"
    else
      kill "${VLLM_SERVER_PID}" 2>/dev/null || true
      wait "${VLLM_SERVER_PID}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

models_ready() {
  curl -fsS --max-time 5 "${API_BASE_URL%/}/models" >/dev/null 2>&1
}

if [[ "${START_VLLM_SERVER}" == "1" || "${START_VLLM_SERVER}" == "true" ]]; then
  models_ready && die "A service is already listening at ${API_BASE_URL}; set START_VLLM_SERVER=0 to reuse it."
  server_args=(
    -m vllm.entrypoints.openai.api_server
    --model "${MODEL_NAME}"
    --revision "${MODEL_REVISION}"
    --served-model-name "${MODEL_NAME}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --tensor-parallel-size 1
    --quantization modelopt
    --dtype auto
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --max-model-len 16384
    --max-num-seqs 16
    --max-num-batched-tokens 8192
    --kv-cache-dtype fp8
    --enable-prefix-caching
    --trust-remote-code
  )
  echo "Starting vLLM; log=${VLLM_SERVER_LOG}"
  "${VLLM_PYTHON}" "${server_args[@]}" >"${VLLM_SERVER_LOG}" 2>&1 &
  VLLM_SERVER_PID="$!"
  deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  until models_ready; do
    if ! kill -0 "${VLLM_SERVER_PID}" 2>/dev/null; then
      tail -n 120 "${VLLM_SERVER_LOG}" >&2 || true
      die "vLLM exited before becoming ready."
    fi
    if (( SECONDS >= deadline )); then
      tail -n 120 "${VLLM_SERVER_LOG}" >&2 || true
      die "vLLM did not become ready within ${STARTUP_TIMEOUT_SECONDS}s."
    fi
    sleep 5
  done
elif ! models_ready; then
  die "No vLLM server is ready at ${API_BASE_URL}."
fi

served_model="$(curl -fsS "${API_BASE_URL%/}/models" | jq -r '.data[0].id // empty')"
[[ "${served_model}" == "${MODEL_NAME}" ]] || die "Expected served model ${MODEL_NAME}, got ${served_model:-none}."

health_payload="$(jq -n --arg model "${MODEL_NAME}" '{model:$model,messages:[{role:"user",content:"Reply with exactly: READY"}],max_tokens:8,temperature:0}')"
health_content="$(curl -fsS "${API_BASE_URL%/}/chat/completions" -H 'Content-Type: application/json' --data "${health_payload}" | jq -r '.choices[0].message.content // empty')"
[[ -n "${health_content}" ]] || die "vLLM chat completion health check returned empty content."
echo "vLLM health response: ${health_content}"

evolution_args=(
  --benchmark_root "${REPO_ROOT}/bench"
  --benchmarks "${BENCHMARK}"
  --problems_file "${PROBLEMS_FILE}"
  --api_backend vllm
  --model_name "${MODEL_NAME}"
  --population_size "${POPULATION_SIZE}"
  --num_generations "${NUM_GENERATIONS}"
  --num_workers "${NUM_WORKERS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max_tokens "${MAX_TOKENS}"
  --generation_mode "${GENERATION_MODE}"
  --strategy_selection "${STRATEGY_SELECTION}"
  --epsilon "${EPSILON}"
  --ucb_c "${UCB_C}"
  --synthesis_evaluator dcpt
  --dcpt_constraint_tolerance_ns "${DCPT_CONSTRAINT_TOLERANCE_NS}"
  --save_path "${SAVE_PATH}/experiment"
)
if [[ "${START_FROM_REFERENCE}" == "1" || "${START_FROM_REFERENCE}" == "true" ]]; then
  evolution_args+=(--start_from_reference)
elif [[ "${START_FROM_REFERENCE}" != "0" && "${START_FROM_REFERENCE}" != "false" ]]; then
  die "START_FROM_REFERENCE must be 1/true or 0/false."
fi
if [[ "${BEST_RECORD_REQUIRE_CONSTRAINTS}" == "1" || "${BEST_RECORD_REQUIRE_CONSTRAINTS}" == "true" ]]; then
  evolution_args+=(--best_record_require_constraints)
elif [[ "${BEST_RECORD_REQUIRE_CONSTRAINTS}" != "0" && "${BEST_RECORD_REQUIRE_CONSTRAINTS}" != "false" ]]; then
  die "BEST_RECORD_REQUIRE_CONSTRAINTS must be 1/true or 0/false."
fi
if [[ -n "${DCPT_CLOCKS}" ]]; then
  read -r -a dcpt_clocks <<< "${DCPT_CLOCKS}"
  evolution_args+=(--dcpt_clocks "${dcpt_clocks[@]}")
fi
if [[ "${PROBLEMS}" == "all" ]]; then
  expected_problem_count="$(awk 'NF && $1 !~ /^#/ { count++ } END { print count + 0 }' "${REPO_ROOT}/bench/${BENCHMARK}/${PROBLEMS_FILE}")"
else
  read -r -a selected_problems <<< "${PROBLEMS}"
  expected_problem_count="${#selected_problems[@]}"
  evolution_args+=(--problems "${selected_problems[@]}")
fi

echo "Running REvolution benchmark; log=${PIPELINE_LOG}"
cd "${REPO_ROOT}"
"${PIPELINE_PYTHON}" src/run_evolution.py "${evolution_args[@]}" 2>&1 | tee "${PIPELINE_LOG}"

mapfile -t problem_summaries < <(find "${SAVE_PATH}/experiment" -type f -name '*_summary.json' -print)
if (( ${#problem_summaries[@]} != expected_problem_count )); then
  die "Expected ${expected_problem_count} problem summaries, found ${#problem_summaries[@]}; see ${PIPELINE_LOG}."
fi
mapfile -t dcpt_reports < <(find "${SAVE_PATH}/experiment" -type f -name '*.dcpt.json' -print)
if (( ${#dcpt_reports[@]} == 0 )); then
  die "The run completed without a model-generated DC/PT PPA report; see ${PIPELINE_LOG}."
fi
printf 'Completed problem summaries: %d\n' "${#problem_summaries[@]}"
printf 'DC/PT reports: %d\n' "${#dcpt_reports[@]}"
echo "Benchmark experiment completed: ${SAVE_PATH}"
