from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .dataset_io import (
    DatasetRecord,
    append_record,
    load_dataset_records,
    load_existing_keys,
    sanitize_id,
)
from .llm_client import AugmentationLLM
from .models import (
    AugmentationConfig,
    AugmentationResult,
    OptimizationTarget,
    PpaMetrics,
    target_improvement,
)
from .synthesis import PpaSynthesizer, SynthesisResult


class DataAugmentationEngine:
    def __init__(
        self,
        *,
        config: AugmentationConfig,
        llm: AugmentationLLM,
        synthesizer: PpaSynthesizer,
        started_monotonic: float | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.synthesizer = synthesizer
        self.output_jsonl = self.config.output_dir / "data.jsonl"
        self.started_monotonic = started_monotonic or time.monotonic()
        self.stop_scheduling_at = self._stop_scheduling_at()

    def run(self) -> Path:
        records = load_dataset_records(self.config.input_dataset)
        original_record_count = len(records)
        records = _deduplicate_records_by_verilog(records)
        rng = random.Random(self.config.random_seed)
        rng.shuffle(records)
        if self.config.limit is not None:
            if self.config.start is not None:
                records = records[
                    self.config.start : self.config.start + self.config.limit
                ]
            else:
                records = records[: self.config.limit]

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        _write_design_name_map(records, self.config.work_dir)
        existing = (
            load_existing_keys(self.output_jsonl) if self.config.resume else set()
        )

        summary = _new_summary(
            self.config,
            len(records),
            self.output_jsonl,
            input_record_count=original_record_count,
        )
        asyncio.run(self._run_records(records, existing, rng, summary))

        _write_summary(summary, self.config.work_dir)
        return self.output_jsonl

    async def _run_records(
        self,
        records: list[DatasetRecord],
        existing: set[tuple[str, str, int, int]],
        rng: random.Random,
        summary: dict[str, Any],
    ) -> None:
        candidate_semaphore = asyncio.Semaphore(max(1, self.config.concurrency))
        llm_semaphore = asyncio.Semaphore(
            max(1, self.config.llm_concurrency or self.config.concurrency)
        )
        synthesis_semaphore = asyncio.Semaphore(
            max(1, self.config.synthesis_concurrency)
        )
        append_lock = asyncio.Lock()
        evolution_tasks = []
        total_candidate_slots = 0
        for record in records:
            selected_targets = self._select_targets(rng)
            try:
                baseline = self._ensure_baseline_ppa(record, selected_targets)
            except (RuntimeError, ValueError) as exc:
                if not self.config.skip_missing_synthesis_report:
                    raise
                summary["skipped_missing_synthesis_report"] += 1
                summary["skipped_records"].append(
                    {
                        "sample_id": record.sample_id,
                        "design_name": record.design_name,
                        "targets": selected_targets,
                        "reason": str(exc),
                    }
                )
                continue
            summary["target_selections"].append(
                {
                    "sample_id": record.sample_id,
                    "design_name": record.design_name,
                    "targets": selected_targets,
                }
            )
            for target in selected_targets:
                total_candidate_slots += (
                    self.config.num_generations
                    * self.config.candidates_per_generation
                )
                evolution_tasks.append(
                    partial(
                        self._run_target_evolution,
                        record=record,
                        baseline=baseline,
                        target=target,
                        existing=existing,
                        candidate_semaphore=candidate_semaphore,
                        llm_semaphore=llm_semaphore,
                        synthesis_semaphore=synthesis_semaphore,
                        append_lock=append_lock,
                        summary=summary,
                    )
                )

        if not evolution_tasks:
            return

        with tqdm(
            total=total_candidate_slots,
            desc=f"Running augmentation ({self.config.concurrency} concurrent)",
        ) as progress:
            await asyncio.gather(
                *(task(progress=progress) for task in evolution_tasks)
            )

    async def _run_target_evolution(
        self,
        *,
        record: DatasetRecord,
        baseline: PpaMetrics,
        target: OptimizationTarget,
        existing: set[tuple[str, str, int, int]],
        candidate_semaphore: asyncio.Semaphore,
        llm_semaphore: asyncio.Semaphore,
        synthesis_semaphore: asyncio.Semaphore,
        append_lock: asyncio.Lock,
        summary: dict[str, Any],
        progress: tqdm,
    ) -> None:
        feedback: str | None = None
        parent_verilog = record.verilog
        best_result: AugmentationResult | None = None

        for generation in range(self.config.num_generations):
            if best_result and best_result.improved:
                parent_verilog = best_result.generated_verilog
                feedback = _feedback_text(best_result.feedback)

            generation_tasks = []
            for candidate_index in range(self.config.candidates_per_generation):
                key = (
                    record.sample_id,
                    target,
                    generation,
                    candidate_index,
                )
                if key in existing:
                    summary["skipped_existing"] += 1
                    progress.update(1)
                    continue
                if not self._can_start_candidate():
                    summary["skipped_deadline"] += 1
                    summary["stopped_by_deadline"] = True
                    progress.update(1)
                    continue
                generation_tasks.append(
                    self._run_candidate_with_limit(
                        candidate_semaphore=candidate_semaphore,
                        llm_semaphore=llm_semaphore,
                        synthesis_semaphore=synthesis_semaphore,
                        record=record,
                        baseline=baseline,
                        target=target,
                        generation=generation,
                        candidate_index=candidate_index,
                        parent_verilog=parent_verilog,
                        feedback=feedback,
                    )
                )

            if not generation_tasks:
                continue

            for task in asyncio.as_completed(generation_tasks):
                result = await task
                if result is None:
                    async with append_lock:
                        summary["skipped_deadline"] += 1
                        summary["stopped_by_deadline"] = True
                        progress.update(1)
                    continue
                async with append_lock:
                    if result.generated_verilog.strip():
                        append_record(result.to_record(), self.output_jsonl)
                    else:
                        summary["not_written_empty_generation"] += 1
                    _update_summary(summary, result)
                    progress.update(1)
                if _is_better(result, best_result):
                    best_result = result

    async def _run_candidate_with_limit(
        self,
        *,
        candidate_semaphore: asyncio.Semaphore,
        llm_semaphore: asyncio.Semaphore,
        synthesis_semaphore: asyncio.Semaphore,
        record: DatasetRecord,
        baseline: PpaMetrics,
        target: OptimizationTarget,
        generation: int,
        candidate_index: int,
        parent_verilog: str,
        feedback: str | None,
    ) -> AugmentationResult | None:
        async with candidate_semaphore:
            if not self._can_start_candidate():
                return None
            return await self._run_candidate(
                llm_semaphore=llm_semaphore,
                synthesis_semaphore=synthesis_semaphore,
                record=record,
                baseline=baseline,
                target=target,
                generation=generation,
                candidate_index=candidate_index,
                parent_verilog=parent_verilog,
                feedback=feedback,
            )

    def _select_targets(self, rng: random.Random) -> list[OptimizationTarget]:
        if self.config.target_selection_mode == "all":
            return self.config.targets
        if not self.config.targets:
            raise ValueError("At least one target must be provided.")
        return [rng.choice(self.config.targets)]

    def _ensure_baseline_ppa(
        self, record: DatasetRecord, targets: list[OptimizationTarget]
    ) -> PpaMetrics:
        if record.ppa.is_complete_for_targets(targets):
            return record.ppa
        if not self.config.synthesize_missing_baseline:
            raise ValueError(
                f"Record {record.sample_id} is missing baseline PPA metrics. "
                "Run prepare_ppa_dataset.py first or pass --synthesize-missing-baseline."
            )
        result = self.synthesizer.synthesize_text(
            verilog=record.verilog,
            sample_id=record.sample_id,
            top_module=record.top_module,
            variant="baseline",
        )
        if not result.success:
            raise RuntimeError(
                f"Baseline synthesis failed for {record.sample_id}: {result.error}"
            )
        return result.metrics

    async def _run_candidate(
        self,
        *,
        llm_semaphore: asyncio.Semaphore,
        synthesis_semaphore: asyncio.Semaphore,
        record: DatasetRecord,
        baseline: PpaMetrics,
        target: OptimizationTarget,
        generation: int,
        candidate_index: int,
        parent_verilog: str,
        feedback: str | None,
    ) -> AugmentationResult:
        variant = f"{target}_gen{generation}_cand{candidate_index}"
        verilog_path = (
            self.config.work_dir
            / sanitize_id(record.sample_id)
            / variant
            / f"{sanitize_id(record.sample_id)}_{variant}.sv"
        )
        thought_path = verilog_path.with_name(f"{verilog_path.stem}_thought.txt")
        feedback_path = verilog_path.with_name(f"{verilog_path.stem}_feedback.txt")
        verilog_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with llm_semaphore:
                thought, generated = await self.llm.optimize_module(
                    description=record.description,
                    verilog=parent_verilog,
                    top_module=record.top_module,
                    target=target,
                    original_ppa=baseline,
                    feedback=feedback,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                )
            await asyncio.to_thread(
                verilog_path.write_text,
                generated,
                encoding="utf-8",
            )
            await asyncio.to_thread(
                thought_path.write_text,
                str(thought),
                encoding="utf-8",
            )
            async with synthesis_semaphore:
                synthesis = await asyncio.to_thread(
                    self.synthesizer.synthesize_file,
                    verilog_file=verilog_path,
                    sample_id=record.sample_id,
                    top_module=record.top_module,
                    variant=variant,
                )
            log_excerpt = _read_log_excerpt(synthesis)
            async with llm_semaphore:
                llm_feedback = await self.llm.feedback(
                    description=record.description,
                    original_verilog=record.verilog,
                    generated_verilog=generated,
                    target=target,
                    original_ppa=baseline,
                    generated_ppa=synthesis.metrics,
                    synthesis_log_excerpt=log_excerpt,
                    thought=thought,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                )
            await asyncio.to_thread(
                feedback_path.write_text,
                _format_feedback(llm_feedback),
                encoding="utf-8",
            )
            improvement, improved = target_improvement(
                baseline, synthesis.metrics, target
            )
            return AugmentationResult(
                sample_id=record.sample_id,
                design_name=record.design_name,
                target=target,
                generation=generation,
                candidate_index=candidate_index,
                original_verilog=record.verilog,
                generated_verilog=generated,
                description=record.description,
                top_module=record.top_module,
                thought=thought,
                feedback=llm_feedback,
                original_ppa=baseline,
                generated_ppa=synthesis.metrics,
                synthesis_success=synthesis.success,
                synthesis_log_path=synthesis.report_path,
                verilog_path=str(verilog_path),
                thought_path=str(thought_path),
                feedback_path=str(feedback_path),
                improvement=improvement,
                improved=improved,
                error=synthesis.error,
            )
        except Exception as exc:
            return AugmentationResult(
                sample_id=record.sample_id,
                design_name=record.design_name,
                target=target,
                generation=generation,
                candidate_index=candidate_index,
                original_verilog=record.verilog,
                generated_verilog="",
                description=record.description,
                top_module=record.top_module,
                thought="",
                feedback={"raw": ""},
                original_ppa=baseline,
                generated_ppa=PpaMetrics(),
                synthesis_success=False,
                synthesis_log_path=None,
                verilog_path=str(verilog_path),
                thought_path=str(thought_path) if thought_path.exists() else None,
                feedback_path=str(feedback_path) if feedback_path.exists() else None,
                improvement=None,
                improved=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _stop_scheduling_at(self) -> float | None:
        if self.config.max_runtime_seconds is None:
            return None
        runtime = max(0.0, self.config.max_runtime_seconds)
        margin = max(0.0, self.config.shutdown_margin_seconds)
        return self.started_monotonic + max(0.0, runtime - margin)

    def _can_start_candidate(self) -> bool:
        if self.stop_scheduling_at is None:
            return True
        return time.monotonic() < self.stop_scheduling_at


def _is_better(
    result: AugmentationResult, current_best: AugmentationResult | None
) -> bool:
    if result.improvement is None:
        return False
    if current_best is None or current_best.improvement is None:
        return True
    return result.improvement > current_best.improvement


def _feedback_text(feedback: dict) -> str:
    parts = [
        feedback.get("equivalence_reasoning"),
        feedback.get("ppa_verification"),
        feedback.get("feedback_thoughts"),
    ]
    return "\n\n".join(part for part in parts if part) or str(
        feedback.get("raw", "")
    )


def _format_feedback(feedback: dict[str, Any]) -> str:
    return (
        "EQUIVALENCE_REASONING:\n"
        f"{feedback.get('equivalence_reasoning') or ''}\n\n"
        "PPA_VERIFICATION:\n"
        f"{feedback.get('ppa_verification') or ''}\n\n"
        "FEEDBACK_THOUGHTS:\n"
        f"{feedback.get('feedback_thoughts') or ''}\n\n"
        "RAW:\n"
        f"{feedback.get('raw') or ''}"
    )


def _read_log_excerpt(synthesis: SynthesisResult, max_chars: int = 8000) -> str:
    for path in (
        synthesis.report_path,
        synthesis.openroad_log_path,
        synthesis.yosys_log_path,
    ):
        if path and Path(path).exists():
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            return text[-max_chars:]
    return synthesis.error or ""


def _deduplicate_records_by_verilog(
    records: list[DatasetRecord],
) -> list[DatasetRecord]:
    deduplicated: dict[str, DatasetRecord] = {}
    for record in records:
        key = _verilog_key(record)
        if key not in deduplicated:
            deduplicated[key] = record
    return list(deduplicated.values())


def _verilog_key(record: DatasetRecord) -> str:
    text = record.verilog.replace("\r\n", "\n").replace("\r", "\n").strip()
    payload = f"{record.top_module}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _new_summary(
    config: AugmentationConfig,
    total_records: int,
    output_jsonl: Path,
    *,
    input_record_count: int | None = None,
) -> dict[str, Any]:
    if input_record_count is None:
        input_record_count = total_records
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input_dataset": str(config.input_dataset),
        "output_jsonl": str(output_jsonl),
        "work_dir": str(config.work_dir),
        "configured_targets": list(config.targets),
        "target_selection_mode": config.target_selection_mode,
        "random_seed": config.random_seed,
        "num_generations": config.num_generations,
        "candidates_per_generation": config.candidates_per_generation,
        "concurrency": config.concurrency,
        "llm_concurrency": config.llm_concurrency or config.concurrency,
        "synthesis_concurrency": config.synthesis_concurrency,
        "skip_missing_synthesis_report": config.skip_missing_synthesis_report,
        "max_summary_errors": config.max_summary_errors,
        "max_runtime_seconds": config.max_runtime_seconds,
        "shutdown_margin_seconds": config.shutdown_margin_seconds,
        "input_records": input_record_count,
        "total_records": total_records,
        "deduplicated_records": input_record_count - total_records,
        "target_selections": [],
        "attempted_candidates": 0,
        "skipped_existing": 0,
        "skipped_deadline": 0,
        "skipped_missing_synthesis_report": 0,
        "not_written_empty_generation": 0,
        "stopped_by_deadline": False,
        "synthesis_successes": 0,
        "synthesis_failures": 0,
        "improved_candidates": 0,
        "best_by_sample_target": {},
        "skipped_records": [],
        "errors": [],
        "suppressed_errors": 0,
    }


def _update_summary(summary: dict[str, Any], result: AugmentationResult) -> None:
    summary["attempted_candidates"] += 1
    if result.synthesis_success:
        summary["synthesis_successes"] += 1
    else:
        summary["synthesis_failures"] += 1
    if result.improved:
        summary["improved_candidates"] += 1
    if result.error:
        _append_summary_error(summary, result)

    key = f"{result.sample_id}:{result.target}"
    current = summary["best_by_sample_target"].get(key)
    if result.improvement is None:
        return
    if current is None or result.improvement > current["improvement"]:
        summary["best_by_sample_target"][key] = {
            "sample_id": result.sample_id,
            "design_name": result.design_name,
            "target": result.target,
            "generation": result.generation,
            "candidate_index": result.candidate_index,
            "improvement": result.improvement,
            "improved": result.improved,
            "synthesis_success": result.synthesis_success,
            "verilog_path": result.verilog_path,
        }


def _append_summary_error(
    summary: dict[str, Any], result: AugmentationResult
) -> None:
    max_errors = summary.get("max_summary_errors", 2000)
    if max_errors is not None and max_errors >= 0:
        if len(summary["errors"]) >= max_errors:
            summary["suppressed_errors"] = summary.get("suppressed_errors", 0) + 1
            return

    summary["errors"].append(
        {
            "sample_id": result.sample_id,
            "design_name": result.design_name,
            "target": result.target,
            "generation": result.generation,
            "candidate_index": result.candidate_index,
            "error": result.error,
        }
    )


def _write_summary(summary: dict[str, Any], work_dir: Path) -> None:
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    target_counts = Counter(
        target
        for selection in summary["target_selections"]
        for target in selection["targets"]
    )
    summary["selected_target_counts"] = dict(sorted(target_counts.items()))
    summary_path = work_dir / "evolution_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_design_name_map(records: list[DatasetRecord], work_dir: Path) -> Path:
    mapping = {
        record.sample_id: record.design_name
        for record in records
        if record.design_name is not None
    }
    path = work_dir / "sample_design_names.json"
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    return path
