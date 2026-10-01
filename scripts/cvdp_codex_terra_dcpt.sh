#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-run}"
shift || true

CVDP_ENV="${CVDP_ENV:-${REPO_ROOT}/.venv-cvdp}"
CVDP_PYTHON="${CVDP_PYTHON:-${CVDP_ENV}/bin/python}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
DATA_ROOT="${DATA_ROOT:-${HOME}/data/cvdp}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-${DATA_ROOT}/upstream}"
HEAVY_ROOT="${HEAVY_ROOT:-${UPSTREAM_ROOT}/heavy}"
HOST_TOOLS_ROOT="${HOST_TOOLS_ROOT:-${UPSTREAM_ROOT}/tools}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${REPO_ROOT}/bench/CVDP-DC-opt}"
RUN_STAMP="${RUN_STAMP:-default}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/cvdp_codex_dcpt/${RUN_STAMP}}"

MODEL="${MODEL:-gpt-5.6-terra}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
POPULATION_SIZE="${POPULATION_SIZE:-10}"
NUM_GENERATIONS="${NUM_GENERATIONS:-20}"
TASKS="${TASKS:-all}"
SELECTION="${SELECTION:-all}"
CLOCKS="${CLOCKS:-1 3 5 8 10 15}"
CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-3600}"
HARNESS_TIMEOUT_SECONDS="${HARNESS_TIMEOUT_SECONDS:-900}"
CONSTRAINT_TOLERANCE_NS="${CONSTRAINT_TOLERANCE_NS:-0.000001}"
SEED="${SEED:-0}"
NO_SANDBOX="${NO_SANDBOX:-0}"

CVDP_REPO_COMMIT="8e894cf74414ab1eaea1e2b4e80a02f123df07b6"
CVDP_DATASET_REVISION="5b807d945f6a99aa645f7e43a64a2115e281b4bf"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

setup_runtime() {
  [[ -n "${UV_BIN}" ]] || die "uv is required for setup."
  "${UV_BIN}" venv --python python3 --clear "${CVDP_ENV}"
  "${UV_BIN}" pip install --python "${CVDP_PYTHON}" \
    'cocotb==2.0.1' \
    'pytest==8.3.2' \
    'pyyaml==6.0.2' \
    'huggingface-hub>=0.34,<2'

  if [[ ! -x "${HOST_TOOLS_ROOT}/bin/iverilog" ]]; then
    command -v apt-get >/dev/null 2>&1 || die "apt-get is required to download host-compatible Icarus."
    command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb is required to extract Icarus."
    mkdir -p "${HOST_TOOLS_ROOT}/packages"
    (
      cd "${HOST_TOOLS_ROOT}/packages"
      apt-get download iverilog
      package="$(find . -maxdepth 1 -name 'iverilog_*.deb' -print -quit)"
      [[ -n "${package}" ]] || die "Failed to download Icarus."
      dpkg-deb -x "${package}" "${HOST_TOOLS_ROOT}"
    )
    mkdir -p "${HOST_TOOLS_ROOT}/bin"
    cat >"${HOST_TOOLS_ROOT}/bin/iverilog" <<EOF
#!/usr/bin/env bash
exec "${HOST_TOOLS_ROOT}/usr/bin/iverilog" -B"${HOST_TOOLS_ROOT}/usr/lib/x86_64-linux-gnu/ivl" "\$@"
EOF
    cat >"${HOST_TOOLS_ROOT}/bin/vvp" <<EOF
#!/usr/bin/env bash
exec "${HOST_TOOLS_ROOT}/usr/bin/vvp" "\$@"
EOF
    chmod +x "${HOST_TOOLS_ROOT}/bin/iverilog" "${HOST_TOOLS_ROOT}/bin/vvp"
  fi

  mkdir -p "${UPSTREAM_ROOT}"
  local checkout="${UPSTREAM_ROOT}/cvdp_benchmark"
  if [[ ! -d "${checkout}/.git" ]]; then
    git clone https://github.com/NVlabs/cvdp_benchmark.git "${checkout}"
  fi
  git -C "${checkout}" fetch --quiet origin "${CVDP_REPO_COMMIT}"
  git -C "${checkout}" checkout --quiet --detach "${CVDP_REPO_COMMIT}"

  mkdir -p "${HEAVY_ROOT}"
  "${CVDP_PYTHON}" - "${HEAVY_ROOT}" "${CVDP_DATASET_REVISION}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/cvdp-benchmark-dataset",
    repo_type="dataset",
    revision=sys.argv[2],
    allow_patterns=["cvdp_v1.1.0_agentic_heavy_code_generation_public/**"],
    local_dir=sys.argv[1],
)
PY
}

if [[ "${ACTION}" == "setup" ]]; then
  setup_runtime
  exit 0
fi

[[ "${ACTION}" =~ ^(prepare|validate|baseline|run|report)$ ]] || \
  die "Usage: $0 [setup|prepare|validate|baseline|run|report] [extra arguments]"
[[ -x "${CVDP_PYTHON}" ]] || die "Missing ${CVDP_ENV}; run '$0 setup' first."
command -v codex >/dev/null 2>&1 || die "codex CLI is not installed."

export PATH="${CVDP_ENV}/bin:${HOST_TOOLS_ROOT}/bin:${PATH}"
export PYTHONHASHSEED=0

if [[ "${ACTION}" == "baseline" || "${ACTION}" == "run" ]]; then
  if [[ -f /etc/profile.d/eda_tools.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/eda_tools.sh
  fi
  if type tool >/dev/null 2>&1; then
    tool 4 6
  fi
fi

mkdir -p "${OUTPUT_ROOT}"
cat >"${OUTPUT_ROOT}/run_config.json" <<EOF
{
  "model": "${MODEL}",
  "reasoning_effort": "${REASONING_EFFORT}",
  "population_size": ${POPULATION_SIZE},
  "num_generations": ${NUM_GENERATIONS},
  "tasks": "${TASKS}",
  "selection": "${SELECTION}",
  "clocks": "${CLOCKS}",
  "require_constraints": true,
  "seed": ${SEED}
}
EOF

read -r -a task_args <<<"${TASKS}"
read -r -a clock_args <<<"${CLOCKS}"
args=(
  "${REPO_ROOT}/src/run_cvdp_evolution.py"
  "${ACTION}"
  --data-root "${DATA_ROOT}"
  --benchmark-dir "${BENCHMARK_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --heavy-root "${HEAVY_ROOT}"
  --model "${MODEL}"
  --reasoning-effort "${REASONING_EFFORT}"
  --population-size "${POPULATION_SIZE}"
  --num-generations "${NUM_GENERATIONS}"
  --codex-timeout-seconds "${CODEX_TIMEOUT_SECONDS}"
  --harness-timeout-seconds "${HARNESS_TIMEOUT_SECONDS}"
  --constraint-tolerance-ns "${CONSTRAINT_TOLERANCE_NS}"
  --seed "${SEED}"
  --tasks "${task_args[@]}"
  --selection "${SELECTION}"
  --clocks "${clock_args[@]}"
)
if [[ "${NO_SANDBOX}" == "1" || "${NO_SANDBOX}" == "true" ]]; then
  args+=(--no-sandbox)
fi
args+=("$@")

exec "${CVDP_PYTHON}" "${args[@]}"
