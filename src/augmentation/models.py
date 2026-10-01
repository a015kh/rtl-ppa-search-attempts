from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

OptimizationTarget = Literal["area", "power", "delay"]
TargetSelectionMode = Literal["all", "random"]
PPA_KEY_ORDER = ("tns", "wns", "eff_clk_period", "power", "area")


@dataclass
class PpaMetrics:
    tns: float | None = None
    wns: float | None = None
    eff_clk_period: float | None = None
    power: float | None = None
    area: float | None = None
    report_path: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "PpaMetrics":
        if not data:
            return cls()
        return cls(
            tns=_to_float(data.get("tns")),
            wns=_to_float(data.get("wns")),
            eff_clk_period=_to_float(
                data.get("eff_clk_period", data.get("delay"))
            ),
            power=_to_float(data.get("power")),
            area=_to_float(data.get("area")),
            report_path=str(data["report_path"]) if data.get("report_path") else None,
        )

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "tns": self.tns,
            "wns": self.wns,
            "eff_clk_period": self.eff_clk_period,
            "power": self.power,
            "area": self.area,
            "report_path": self.report_path,
        }

    def get_target_value(self, target: OptimizationTarget) -> float | None:
        if target == "delay":
            return self.eff_clk_period
        return getattr(self, target)

    def is_complete_for_targets(self, targets: list[OptimizationTarget]) -> bool:
        return all(self.get_target_value(target) is not None for target in targets)


@dataclass
class AugmentationConfig:
    input_dataset: Path = Path("data/pyra_desc_yosys")
    output_dir: Path = Path("data/pyra_desc_augmented")
    work_dir: Path = Path("outputs/augumention")
    targets: list[OptimizationTarget] = field(
        default_factory=lambda: ["area", "power", "delay"]
    )
    target_selection_mode: TargetSelectionMode = "all"
    random_seed: int | None = None
    num_generations: int = 1
    candidates_per_generation: int = 1
    concurrency: int = 4
    llm_concurrency: int | None = None
    synthesis_concurrency: int = 8
    start: int | None = None
    limit: int | None = None
    resume: bool = True
    model_name: str = "Qwen/Qwen3-Coder-Next"
    api_backend: str = "local-vllm"
    api_base_url: str | None = None
    request_timeout_seconds: float = 600.0
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    local_vllm_batch_size: int = 64
    local_vllm_batch_timeout_ms: int = 50
    local_vllm_tensor_parallel_size: int = 1
    local_vllm_gpu_memory_utilization: float = 0.92
    local_vllm_max_model_len: int | None = None
    local_vllm_max_num_batched_tokens: int | None = None
    local_vllm_max_num_seqs: int | None = 64
    local_vllm_dtype: str = "auto"
    trust_remote_code: bool = True
    enable_prefix_caching: bool = True
    yosys_bin: str = "yosys"
    openroad_bin: str = "openroad"
    clock_period_ns: float = 0.01
    synth_timeout_seconds: int = 300
    synthesize_missing_baseline: bool = False
    skip_missing_synthesis_report: bool = False
    max_summary_errors: int = 2000
    max_runtime_seconds: float | None = 47.5 * 3600
    shutdown_margin_seconds: float = 900.0


@dataclass
class AugmentationResult:
    sample_id: str
    design_name: str | None
    target: OptimizationTarget
    generation: int
    candidate_index: int
    original_verilog: str
    generated_verilog: str
    description: str
    top_module: str
    thought: str
    feedback: dict[str, Any]
    original_ppa: PpaMetrics
    generated_ppa: PpaMetrics
    synthesis_success: bool
    synthesis_log_path: str | None
    verilog_path: str
    thought_path: str | None
    feedback_path: str | None
    improvement: float | None
    improved: bool | None
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "design_name": self.design_name,
            "target": self.target,
            "generation": self.generation,
            "candidate_index": self.candidate_index,
            "description": self.description,
            "top_module": self.top_module,
            "thought": self.thought,
            "feedback": self.feedback,
            "original_verilog": self.original_verilog,
            "generated_verilog": self.generated_verilog,
            "original_ppa": self.original_ppa.to_dict(),
            "generated_ppa": self.generated_ppa.to_dict(),
            "synthesis_success": self.synthesis_success,
            "synthesis_log_path": self.synthesis_log_path,
            "verilog_path": self.verilog_path,
            "thought_path": self.thought_path,
            "feedback_path": self.feedback_path,
            "improvement": self.improvement,
            "improved": self.improved,
            "error": self.error,
        }


def target_improvement(
    original: PpaMetrics, candidate: PpaMetrics, target: OptimizationTarget
) -> tuple[float | None, bool | None]:
    original_value = original.get_target_value(target)
    candidate_value = candidate.get_target_value(target)
    if original_value is None or candidate_value is None:
        return None, None
    if original_value == 0:
        return None, candidate_value < original_value
    improvement = (original_value - candidate_value) / abs(original_value)
    return improvement, candidate_value < original_value


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
