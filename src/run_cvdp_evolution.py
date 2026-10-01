#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from revolution.codex_cli import CodexCLIError, CodexCLIRunner
from revolution.cvdp import (
    CVDP_CLOCKS_NS,
    CVDPError,
    CVDPTask,
    NativeCocotbRunner,
    prepare_catalog,
    project_digest,
)
from revolution.cvdp_dcpt import CVDPProjectDCPTEvaluator, pareto_front, save_result
from revolution.dcpt import DEFAULT_CELL_LIBRARY, DEFAULT_FLOW_ROOT, DCPTFlow


QOR_SLOW_TASKS = {"cvdp_copilot_gaussian_rounding_div_0022"}


EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["thought", "summary"],
    "additionalProperties": False,
}
FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "analysis": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": ["score", "analysis", "next_step"],
    "additionalProperties": False,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CVDPError(f"Expected JSON object: {path}")
    return value


def git(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *command], cwd=cwd, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise CVDPError(completed.stderr.strip() or f"git {' '.join(command)} failed")
    return completed.stdout.strip()


def initialize_candidate_workspace(task: CVDPTask, workspace: Path) -> None:
    code = workspace / "code"
    if not (code / ".git").is_dir():
        git(["init", "-q"], code)
    git(["config", "user.name", "CVDP Revolution"], code)
    git(["config", "user.email", "revolution@localhost"], code)
    (code / "SPEC.md").write_text(task.prompt + "\n", encoding="utf-8")
    (code / "AGENTS.md").write_text(
        "# CVDP candidate rules\n\n"
        f"Edit only these target files: {', '.join(task.targets)}.\n"
        "Preserve all module interfaces and functional behavior. Do not modify SPEC.md, "
        "AGENTS.md, verification files, or generated artifacts. The external pipeline runs "
        "simulation and synthesis after the turn.\n",
        encoding="utf-8",
    )
    git(["add", "-A"], code)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=code).returncode:
        git(["commit", "-q", "-m", "candidate baseline"], code)


def clone_workspace(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["cp", "-a", "--reflink=auto", str(source), str(destination)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        shutil.copytree(source, destination, symlinks=True)


def changed_paths(code: Path) -> list[str]:
    tracked = git(["diff", "--name-only", "HEAD"], code).splitlines()
    untracked = git(["ls-files", "--others", "--exclude-standard"], code).splitlines()
    return sorted(set(item for item in tracked + untracked if item))


def save_events(path: Path, events: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(event), sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def feasible_points(result: Mapping[str, Any]) -> dict[float, Mapping[str, Any]]:
    return {
        float(item["clock_ns"]): item
        for item in result.get("clock_results", [])
        if item.get("success") and item.get("gate_passed") and item.get("constraint_met")
    }


def mean_improvement(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    candidate_points = feasible_points(candidate)
    baseline_points = feasible_points(baseline)
    values = []
    for clock in sorted(candidate_points.keys() & baseline_points.keys()):
        current = candidate_points[clock]["metrics"]
        reference = baseline_points[clock]["metrics"]
        if float(reference["area"]) <= 0 or float(reference["power"]) <= 0:
            continue
        area = (float(reference["area"]) - float(current["area"])) / float(reference["area"])
        power = (float(reference["power"]) - float(current["power"])) / float(reference["power"])
        values.append((area + power) / 2.0)
    return sum(values) / len(values) if values else -math.inf


@dataclass
class Candidate:
    candidate_id: str
    workspace: Path
    result: dict[str, Any]
    thought: str
    feedback: str
    score: float


class CVDPSearch:
    def __init__(self, args: argparse.Namespace, task: CVDPTask, task_root: Path) -> None:
        self.args = args
        self.task = task
        self.task_root = task_root
        self.codex = CodexCLIRunner(
            args.model,
            args.reasoning_effort,
            args.codex_timeout_seconds,
        )
        self.evaluator = CVDPProjectDCPTEvaluator(
            task_root / ".dcpt_cache",
            args.dcpt_flow_root,
            args.dcpt_config,
            args.cell_library,
            args.clocks,
            args.constraint_tolerance_ns,
            args.harness_timeout_seconds,
            not args.no_sandbox,
        )
        self.random = random.Random(args.seed)

    def _generation_prompt(self, parent: Candidate, generation: int, index: int) -> str:
        strategies = (
            "simplify redundant logic and reduce width or duplicated state",
            "refactor the microarchitecture for lower area and switching activity",
            "explore a materially different synthesizable architecture",
            "improve the critical combinational paths while preserving latency",
            "apply resource sharing and constant-strength reduction where behavior permits",
        )
        strategy = strategies[(generation * self.args.population_size + index) % len(strategies)]
        feedback = parent.feedback or "This is the generated baseline; preserve correctness and improve PPA."
        return (
            f"Optimize CVDP task {self.task.id}. Read SPEC.md and the current target RTL. "
            f"Your assigned strategy is to {strategy}.\n\n"
            f"Measured parent feedback:\n{feedback}\n\n"
            f"Only edit these targets: {', '.join(self.task.targets)}. Run no simulation or "
            "synthesis yourself; the external evaluator will do that. Make one coherent change "
            "and return the requested structured summary."
        )

    def _evaluate_workspace(self, workspace: Path, result_path: Path) -> dict[str, Any]:
        digest = project_digest(workspace, self.task.targets)
        if result_path.is_file():
            cached = read_json(result_path)
            if cached.get("source_digest") == digest:
                return cached
        result = self.evaluator.evaluate(self.task.id, workspace, digest)
        save_result(result_path, result)
        return read_json(result_path)

    def run(self, baseline_workspace: Path, baseline: dict[str, Any]) -> dict[str, Any]:
        summary_path = self.task_root / "search.summary.json"
        if summary_path.is_file():
            return read_json(summary_path)
        baseline_candidate = Candidate(
            "baseline",
            baseline_workspace,
            baseline,
            "Generated CVDP seed.",
            "",
            0.0,
        )
        population = [baseline_candidate]
        records: list[dict[str, Any]] = []
        start_generation = 1
        checkpoint_path = self.task_root / "search.checkpoint.json"
        if checkpoint_path.is_file():
            checkpoint = read_json(checkpoint_path)
            records = list(checkpoint.get("records") or [])
            restored: list[Candidate] = []
            for candidate_id in checkpoint.get("population") or []:
                if candidate_id == "baseline":
                    restored.append(baseline_candidate)
                    continue
                candidate_dir = self.task_root / "candidates" / candidate_id
                result_path = candidate_dir / "dcpt.json"
                feedback_path = candidate_dir / "feedback.json"
                generation_path = candidate_dir / "generation.json"
                if not result_path.is_file() or not feedback_path.is_file():
                    continue
                result = read_json(result_path)
                feedback_data = read_json(feedback_path)
                generation_data = read_json(generation_path)
                restored.append(
                    Candidate(
                        candidate_id,
                        candidate_dir / "workspace",
                        result,
                        str(generation_data.get("output", {}).get("thought", "")),
                        str(feedback_data.get("output", {}).get("analysis", "")),
                        mean_improvement(result, baseline),
                    )
                )
            if restored:
                population = sorted(restored, key=lambda item: item.score, reverse=True)
                start_generation = int(checkpoint.get("completed_generation", 0)) + 1
        for generation in range(start_generation, self.args.num_generations + 1):
            offspring: list[Candidate] = []
            for index in range(1, self.args.population_size + 1):
                parent = self.random.choice(population[: min(5, len(population))])
                candidate_id = f"gen_{generation:03d}_{index:03d}"
                candidate_dir = self.task_root / "candidates" / candidate_id
                workspace = candidate_dir / "workspace"
                clone_workspace(parent.workspace, workspace)
                code = workspace / "code"
                git(["reset", "--hard", "-q", "HEAD"], code)
                prompt = self._generation_prompt(parent, generation, index)
                try:
                    edit = self.codex.run(
                        prompt,
                        EDIT_SCHEMA,
                        cwd=code,
                        sandbox="workspace-write",
                    )
                    save_events(candidate_dir / "generation.events.jsonl", edit.events)
                    write_json(
                        candidate_dir / "generation.json",
                        {
                            "thread_id": edit.thread_id,
                            "output": edit.output,
                            "usage": {
                                "input_tokens": edit.prompt_tokens,
                                "cached_input_tokens": edit.cached_prompt_tokens,
                                "output_tokens": edit.completion_tokens,
                                "reasoning_tokens": edit.reasoning_tokens,
                            },
                        },
                    )
                    changes = changed_paths(code)
                    invalid = sorted(set(changes) - set(self.task.targets))
                    if invalid:
                        raise CVDPError(f"Codex modified files outside the target set: {invalid}")
                    if not changes:
                        raise CVDPError("Codex did not modify a target file.")
                    result = self._evaluate_workspace(
                        workspace, candidate_dir / "dcpt.json"
                    )
                    feedback_prompt = (
                        "The external CVDP evaluator measured your preceding RTL edit. Analyze "
                        "the result, explain the dominant correctness or PPA issue, and propose one "
                        "next optimization. Do not edit files in this turn.\n\n"
                        + json.dumps(result, indent=2)
                    )
                    feedback = self.codex.run(
                        feedback_prompt,
                        FEEDBACK_SCHEMA,
                        thread_id=edit.thread_id,
                        cwd=code,
                    )
                    save_events(candidate_dir / "feedback.events.jsonl", feedback.events)
                    write_json(
                        candidate_dir / "feedback.json",
                        {
                            "thread_id": feedback.thread_id,
                            "output": feedback.output,
                            "usage": {
                                "input_tokens": feedback.prompt_tokens,
                                "cached_input_tokens": feedback.cached_prompt_tokens,
                                "output_tokens": feedback.completion_tokens,
                                "reasoning_tokens": feedback.reasoning_tokens,
                            },
                        },
                    )
                    score = mean_improvement(result, baseline)
                    git(["add", "--", *self.task.targets], code)
                    git(["commit", "-q", "-m", candidate_id], code)
                    candidate = Candidate(
                        candidate_id,
                        workspace,
                        result,
                        str(edit.output["thought"]),
                        str(feedback.output["analysis"])
                        + "\nNext step: "
                        + str(feedback.output["next_step"]),
                        score,
                    )
                    offspring.append(candidate)
                    records.append(
                        {
                            "candidate_id": candidate_id,
                            "parent_id": parent.candidate_id,
                            "generation": generation,
                            "score": None if not math.isfinite(score) else score,
                            "eligible": bool(result.get("eligible")),
                            "source_digest": result.get("source_digest"),
                        }
                    )
                except (CVDPError, CodexCLIError, OSError, ValueError) as exc:
                    write_json(
                        candidate_dir / "failure.json",
                        {
                            "candidate_id": candidate_id,
                            "parent_id": parent.candidate_id,
                            "error": str(exc),
                        },
                    )
                    records.append(
                        {
                            "candidate_id": candidate_id,
                            "parent_id": parent.candidate_id,
                            "generation": generation,
                            "score": None,
                            "eligible": False,
                            "error": str(exc),
                        }
                    )
            population = sorted(
                population + offspring,
                key=lambda item: item.score,
                reverse=True,
            )[: self.args.population_size]
            write_json(
                self.task_root / "search.checkpoint.json",
                {
                    "completed_generation": generation,
                    "population": [item.candidate_id for item in population],
                    "records": records,
                },
            )
        all_points = [
            {**point, "candidate_id": "baseline"}
            for point in baseline.get("clock_results", [])
        ]
        for result_path in sorted((self.task_root / "candidates").glob("*/dcpt.json")):
            candidate_id = result_path.parent.name
            result = read_json(result_path)
            for point in result.get("clock_results", []):
                all_points.append({**point, "candidate_id": candidate_id})
        summary = {
            "task_id": self.task.id,
            "model": self.args.model,
            "reasoning_effort": self.args.reasoning_effort,
            "require_constraints": True,
            "population_size": self.args.population_size,
            "num_generations": self.args.num_generations,
            "baseline_digest": baseline.get("source_digest"),
            "best_mean_improvement": population[0].score,
            "best_candidate_id": population[0].candidate_id,
            "pareto_front": pareto_front(all_points),
            "records": records,
        }
        write_json(summary_path, summary)
        return summary


def catalog_entries(benchmark_dir: Path) -> list[dict[str, Any]]:
    path = benchmark_dir / "catalog.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def selected_tasks(args: argparse.Namespace) -> list[str]:
    entries = catalog_entries(args.benchmark_dir)
    eligible_entries = [entry for entry in entries if entry["cohort"] == "optimization"]
    eligible = [entry["id"] for entry in eligible_entries]
    if not args.tasks or args.tasks == ["all"]:
        if args.selection in {"qor", "qor-fast"}:
            eligible_entries = [
                entry
                for entry in eligible_entries
                if entry.get("conversion_strategy") == "native_qor_to_optimization"
            ]
            if args.selection == "qor-fast":
                eligible_entries = [
                    entry for entry in eligible_entries if entry["id"] not in QOR_SLOW_TASKS
                ]
        elif args.selection == "native-synthesis":
            eligible_entries = [
                entry for entry in eligible_entries if entry.get("native_synthesis_harness")
            ]
        return [entry["id"] for entry in eligible_entries]
    unknown = sorted(set(args.tasks) - {entry["id"] for entry in entries})
    if unknown:
        raise CVDPError(f"Unknown CVDP tasks: {unknown}")
    ineligible = [
        task_id for task_id in args.tasks if task_id not in eligible
    ]
    if ineligible:
        raise CVDPError(f"Tasks are not in the optimization cohort: {ineligible}")
    return args.tasks


def ensure_baseline_workspace(args: argparse.Namespace, task: CVDPTask, task_root: Path) -> Path:
    workspace = task_root / "baseline" / "workspace"
    if not (workspace / "code").is_dir():
        task.materialize(workspace, args.heavy_root)
        initialize_candidate_workspace(task, workspace)
    return workspace


def action_validate(args: argparse.Namespace) -> None:
    runner = NativeCocotbRunner(args.harness_timeout_seconds, not args.no_sandbox)
    for task_id in selected_tasks(args):
        task = CVDPTask.load(args.benchmark_dir, task_id)
        task_root = args.output_root / task_id
        workspace = ensure_baseline_workspace(args, task, task_root)
        result = runner.run(workspace)
        write_json(
            task_root / "baseline" / "validation.json",
            {
                "task_id": task_id,
                "passed": result.passed,
                "returncode": result.returncode,
                "command": result.command,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        print(f"{task_id}: {'passed' if result.passed else 'failed'}")


def evaluator_from_args(args: argparse.Namespace, task_root: Path) -> CVDPProjectDCPTEvaluator:
    return CVDPProjectDCPTEvaluator(
        task_root / ".dcpt_cache",
        args.dcpt_flow_root,
        args.dcpt_config,
        args.cell_library,
        args.clocks,
        args.constraint_tolerance_ns,
        args.harness_timeout_seconds,
        not args.no_sandbox,
    )


def ensure_baseline_result(
    args: argparse.Namespace, task: CVDPTask, task_root: Path, workspace: Path
) -> dict[str, Any]:
    path = task_root / "baseline" / "dcpt.json"
    digest = project_digest(workspace, task.targets)
    if path.is_file():
        result = read_json(path)
        if result.get("source_digest") == digest:
            return result
    evaluator = evaluator_from_args(args, task_root)
    evaluated = evaluator.evaluate(task.id, workspace, digest)
    save_result(path, evaluated)
    return read_json(path)


def action_baseline(args: argparse.Namespace) -> None:
    for task_id in selected_tasks(args):
        task = CVDPTask.load(args.benchmark_dir, task_id)
        task_root = args.output_root / task_id
        workspace = ensure_baseline_workspace(args, task, task_root)
        result = ensure_baseline_result(args, task, task_root, workspace)
        print(f"{task_id}: eligible={result.get('eligible', False)}")


def action_run(args: argparse.Namespace) -> None:
    for task_id in selected_tasks(args):
        task = CVDPTask.load(args.benchmark_dir, task_id)
        task_root = args.output_root / task_id
        workspace = ensure_baseline_workspace(args, task, task_root)
        baseline = ensure_baseline_result(args, task, task_root, workspace)
        if not baseline.get("eligible"):
            write_json(
                task_root / "search.summary.json",
                {
                    "task_id": task_id,
                    "status": "baseline_ineligible",
                    "require_constraints": True,
                },
            )
            print(f"{task_id}: skipped because the local baseline is ineligible")
            continue
        summary = CVDPSearch(args, task, task_root).run(workspace, baseline)
        print(
            f"{task_id}: best_mean_improvement={summary['best_mean_improvement']:.6f} "
            f"pareto_points={len(summary['pareto_front'])}"
        )


def action_report(args: argparse.Namespace) -> None:
    entries = catalog_entries(args.benchmark_dir)
    cohort_counts: dict[str, int] = {}
    for entry in entries:
        cohort_counts[entry["cohort"]] = cohort_counts.get(entry["cohort"], 0) + 1
    summaries = []
    for path in sorted(args.output_root.glob("*/search.summary.json")):
        summaries.append(read_json(path))
    completed = [item for item in summaries if "best_mean_improvement" in item]
    values = [float(item["best_mean_improvement"]) for item in completed]
    report = {
        "benchmark": "CVDP-DC-opt",
        "catalog_cohorts": cohort_counts,
        "search_summaries": len(summaries),
        "completed_searches": len(completed),
        "mean_best_improvement": sum(values) / len(values) if values else None,
        "tasks": summaries,
    }
    write_json(args.output_root / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Docker-free CVDP Codex/DCPT optimization")
    parser.add_argument("action", choices=["prepare", "validate", "baseline", "run", "report"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--heavy-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument(
        "--selection",
        choices=["all", "qor", "qor-fast", "native-synthesis"],
        default="all",
        help=(
            "Select all valid seeds, native QoR tasks, the fast QoR subset, or all "
            "native-synthesis tasks."
        ),
    )
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--codex-timeout-seconds", type=int, default=3600)
    parser.add_argument("--harness-timeout-seconds", type=int, default=900)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--num-generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clocks", type=float, nargs="+", default=list(CVDP_CLOCKS_NS))
    parser.add_argument("--constraint-tolerance-ns", type=float, default=1e-6)
    parser.add_argument("--dcpt-flow-root", default=None)
    parser.add_argument("--dcpt-config", default=None)
    parser.add_argument("--cell-library", default=None)
    parser.add_argument("--no-sandbox", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.action in {"baseline", "run"} and not all([args.dcpt_flow_root, args.dcpt_config, args.cell_library]):
        raise SystemExit("DC/PT requires --dcpt-flow-root, --dcpt-config and --cell-library.")
    args.data_root = args.data_root.expanduser().resolve()
    args.benchmark_dir = args.benchmark_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.heavy_root = args.heavy_root.expanduser().resolve()
    try:
        if args.action == "prepare":
            benchmark = prepare_catalog(args.data_root, args.benchmark_dir)
            print(json.dumps(benchmark, indent=2))
            return 0
        if not (args.benchmark_dir / "catalog.jsonl").is_file():
            raise CVDPError("Benchmark is not prepared; run the prepare action first.")
        args.output_root.mkdir(parents=True, exist_ok=True)
        if args.action in {"baseline", "run"}:
            DCPTFlow.load(args.dcpt_flow_root).preflight(
                args.output_root / ".dcpt_preflight", 1, args.dcpt_config
            )
        {
            "validate": action_validate,
            "baseline": action_baseline,
            "run": action_run,
            "report": action_report,
        }[args.action](args)
        return 0
    except (CVDPError, CodexCLIError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
