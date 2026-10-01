#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-gpt-5.6-luna}"
POPULATION_SIZE="${POPULATION_SIZE:-10}"
NUM_GENERATIONS="${NUM_GENERATIONS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROBLEMS="${PROBLEMS:-Q4_HC}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
STRATEGY_SELECTION="${STRATEGY_SELECTION:-ucb}"
EPSILON="${EPSILON:-0.1}"
UCB_C="${UCB_C:-2.0}"
GENERATION_MODE="${GENERATION_MODE:-whole}"
DCPT_CLOCKS="${DCPT_CLOCKS:-1 3 5 8 10 15}"
DCPT_CONSTRAINT_TOLERANCE_NS="${DCPT_CONSTRAINT_TOLERANCE_NS:-0.000001}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/output/openai_gpt56_luna_icrtl/${RUN_STAMP}}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON}" ]] || die "Python environment not found: ${PYTHON}"

if [[ "${1:-run}" == "prepare" ]]; then
  . /etc/profile.d/eda_tools.sh
  tool 4 6
  read -r -a selected_problems <<< "${PROBLEMS}"
  exec "${PYTHON}" src/build_icrtl_dc_benchmark.py all \
    --workers "${NUM_WORKERS}" \
    --overwrite \
    --problems "${selected_problems[@]}"
fi
[[ "${1:-run}" == "run" ]] || die "Usage: $0 [prepare|run]"
if [[ -z "${RTL_OAI_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export RTL_OAI_KEY="${OPENAI_API_KEY}"
fi
[[ -n "${RTL_OAI_KEY:-}" ]] || die "Set RTL_OAI_KEY to your OpenAI API key."
[[ -f bench/ICRTL/benchmark.json ]] ||
  die "ICRTL DC/PT metadata is missing. Run: $0 prepare"
. /etc/profile.d/eda_tools.sh
tool 4 6

mkdir -p "${SAVE_PATH}"
{
  echo "date=$(date --iso-8601=seconds)"
  echo "model=${MODEL_NAME}"
  echo "benchmark=ICRTL"
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
  echo "start_from_reference=true"
  echo "best_record_require_constraints=true"
  echo "dcpt_clocks=${DCPT_CLOCKS}"
} | tee "${SAVE_PATH}/run_config.txt"

read -r -a clocks <<< "${DCPT_CLOCKS}"
read -r -a selected_problems <<< "${PROBLEMS}"
"${PYTHON}" src/run_evolution.py \
  --benchmark_root "${REPO_ROOT}/bench" \
  --benchmarks ICRTL \
  --problems "${selected_problems[@]}" \
  --problems_file problems.txt \
  --api_backend openai \
  --model_name "${MODEL_NAME}" \
  --population_size "${POPULATION_SIZE}" \
  --num_generations "${NUM_GENERATIONS}" \
  --num_workers "${NUM_WORKERS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --strategy_selection "${STRATEGY_SELECTION}" \
  --epsilon "${EPSILON}" \
  --ucb_c "${UCB_C}" \
  --generation_mode "${GENERATION_MODE}" \
  --start_from_reference \
  --synthesis_evaluator dcpt \
  --dcpt_clocks "${clocks[@]}" \
  --dcpt_constraint_tolerance_ns "${DCPT_CONSTRAINT_TOLERANCE_NS}" \
  --best_record_require_constraints \
  --save_path "${SAVE_PATH}/experiment" \
  2>&1 | tee "${SAVE_PATH}/pipeline.log"
