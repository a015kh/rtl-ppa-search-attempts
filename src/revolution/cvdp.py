from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from functools import lru_cache

import yaml


CVDP_GITHUB_COMMIT = "8e894cf74414ab1eaea1e2b4e80a02f123df07b6"
CVDP_DATASET_REVISION = "5b807d945f6a99aa645f7e43a64a2115e281b4bf"
CVDP_CLOCKS_NS = (1.0, 3.0, 5.0, 8.0, 10.0, 15.0)
NATIVE_GLOBS = (
    "native/cvdp_v1.1.0_nonagentic_code_generation_no_commercial_generated_with_solutions.jsonl",
    "native/cvdp_v1.1.0_agentic_code_generation_no_commercial_generated_with_solutions.jsonl",
    "native/cvdp_v1.1.0_agentic_heavy_code_generation_generated_with_solutions.jsonl",
)


class CVDPError(RuntimeError):
    """CVDP data, harness, or project operation failed."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_relative(raw: str) -> Path:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise CVDPError(f"Unsafe CVDP path: {raw!r}")
    return Path(*path.parts)


def _mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise CVDPError(f"{label} must be an object.")
    result: dict[str, str] = {}
    for raw_path, content in value.items():
        _safe_relative(str(raw_path))
        if content is None:
            continue
        if not isinstance(content, str):
            raise CVDPError(f"{label}[{raw_path!r}] must be text.")
        result[str(raw_path)] = content
    return result


def normalize_native_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize nonagentic and agentic CVDP rows to one project representation."""
    task_id = str(row.get("id", "")).strip()
    if not task_id:
        raise CVDPError("CVDP row has no id.")
    if isinstance(row.get("input"), dict):
        input_data = row["input"]
        output_data = row.get("output") or {}
        context = _mapping(input_data.get("context") or {}, "input.context")
        seed = _mapping(output_data.get("context") or {}, "output.context")
        prompt = str(input_data.get("prompt") or "")
        raw_harness = row.get("harness") or {}
        harness = _mapping(raw_harness.get("files") or raw_harness, "harness")
        heavy = False
        repository = None
    else:
        raw_context = row.get("context") or {}
        heavy = set(raw_context) == {"commit", "repo"}
        context = {} if heavy else _mapping(raw_context, "context")
        seed = _mapping(row.get("patch") or {}, "patch")
        prompt = str(row.get("prompt") or "")
        harness = _mapping(row.get("harness") or {}, "harness")
        repository = dict(raw_context) if heavy else None
    return {
        "id": task_id,
        "categories": list(row.get("categories") or []),
        "prompt": prompt,
        "context": context,
        "seed": seed,
        "targets": sorted(seed),
        "harness": harness,
        "heavy": heavy,
        "repository": repository,
    }


def load_native_index(data_root: Path) -> dict[str, tuple[str, int, dict[str, Any]]]:
    index: dict[str, tuple[str, int, dict[str, Any]]] = {}
    for relative in NATIVE_GLOBS:
        path = data_root / relative
        if not path.is_file():
            raise CVDPError(f"Missing generated CVDP data: {path}")
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                try:
                    row = json.loads(line)
                    normalized = normalize_native_row(row)
                except (json.JSONDecodeError, CVDPError) as exc:
                    raise CVDPError(f"Invalid row {row_index} in {path}: {exc}") from exc
                task_id = normalized["id"]
                if task_id in index:
                    raise CVDPError(f"Duplicate native CVDP id: {task_id}")
                index[task_id] = (relative, row_index, normalized)
    return index


def prepare_catalog(data_root: Path, benchmark_dir: Path) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve()
    manifest_path = data_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise CVDPError(f"Missing generated CVDP manifest: {manifest_path}")
    native = load_native_index(data_root)
    catalog: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CVDPError(f"Invalid manifest row {line_number}: {exc}") from exc
            task_id = str(source.get("original_id") or "")
            if task_id not in native:
                raise CVDPError(f"Manifest task has no native row: {task_id}")
            native_file, native_row, normalized = native[task_id]
            generation_status = str((source.get("generation") or {}).get("status", "missing"))
            validation_status = str((source.get("validation") or {}).get("status", "not_validated"))
            if generation_status != "completed":
                cohort = "generation_missing"
            elif validation_status == "passed":
                cohort = "optimization"
            else:
                cohort = "validation_failed"
            catalog.append(
                {
                    "id": task_id,
                    "cohort": cohort,
                    "generation_status": generation_status,
                    "validation_status": validation_status,
                    "native_file": native_file,
                    "native_row": native_row,
                    "mode": str(source.get("mode") or "unknown"),
                    "heavy": bool(source.get("heavy", normalized["heavy"])),
                    "conversion_strategy": str(
                        source.get("conversion_strategy") or "unknown"
                    ),
                    "native_synthesis_harness": bool(
                        source.get("native_synthesis_harness", False)
                    ),
                    "targets": normalized["targets"],
                    "categories": normalized["categories"],
                    "source_row_sha256": source.get("row_sha256"),
                }
            )
    catalog.sort(key=lambda item: item["id"])
    counts: dict[str, int] = {}
    for item in catalog:
        counts[item["cohort"]] = counts.get(item["cohort"], 0) + 1
    expected = {
        "optimization": 167,
        "validation_failed": 130,
        "generation_missing": 45,
    }
    if len(catalog) != 342 or counts != expected:
        raise CVDPError(
            f"Unexpected CVDP cohorts: total={len(catalog)} counts={counts}; expected={expected}"
        )
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "catalog.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in catalog),
        encoding="utf-8",
    )
    eligible = [item["id"] for item in catalog if item["cohort"] == "optimization"]
    (benchmark_dir / "problems.txt").write_text("\n".join(eligible) + "\n", encoding="utf-8")
    benchmark = {
        "schema_version": 1,
        "name": "CVDP-DC-opt",
        "data_root": str(data_root),
        "github_commit": CVDP_GITHUB_COMMIT,
        "dataset_revision": CVDP_DATASET_REVISION,
        "clocks_ns": list(CVDP_CLOCKS_NS),
        "require_constraints": True,
        "counts": counts,
        "total": len(catalog),
    }
    (benchmark_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return benchmark


def read_jsonl_row(path: Path, row_index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == row_index:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CVDPError(f"Expected JSON object in {path}:{row_index + 1}")
                return value
    raise CVDPError(f"Missing row {row_index} in {path}")


@dataclass(frozen=True)
class CVDPTask:
    id: str
    prompt: str
    context: dict[str, str]
    seed: dict[str, str]
    targets: tuple[str, ...]
    harness: dict[str, str]
    heavy: bool
    repository: dict[str, str] | None
    catalog: dict[str, Any]

    @classmethod
    def load(cls, benchmark_dir: Path, task_id: str) -> "CVDPTask":
        config = json.loads((benchmark_dir / "benchmark.json").read_text(encoding="utf-8"))
        entry = None
        for line in (benchmark_dir / "catalog.jsonl").read_text(encoding="utf-8").splitlines():
            candidate = json.loads(line)
            if candidate["id"] == task_id:
                entry = candidate
                break
        if entry is None:
            raise CVDPError(f"Unknown CVDP task: {task_id}")
        data_root = Path(config["data_root"])
        row = read_jsonl_row(data_root / entry["native_file"], int(entry["native_row"]))
        normalized = normalize_native_row(row)
        return cls(
            id=task_id,
            prompt=normalized["prompt"],
            context=normalized["context"],
            seed=normalized["seed"],
            targets=tuple(normalized["targets"]),
            harness=normalized["harness"],
            heavy=normalized["heavy"],
            repository=normalized["repository"],
            catalog=entry,
        )

    def _checkout_heavy(self, destination: Path, heavy_root: Path) -> None:
        assert self.repository is not None
        repository = self.repository["repo"]
        commit = self.repository["commit"]
        candidates = sorted(heavy_root.rglob(f"{repository}*.bundle"))
        candidates += sorted(heavy_root.rglob(f"{repository}*.gitbundle"))
        unpacked = heavy_root / repository
        if unpacked.is_dir():
            command = ["git", "clone", "--quiet", str(unpacked), str(destination)]
        elif candidates:
            command = ["git", "clone", "--quiet", str(candidates[0]), str(destination)]
        else:
            raise CVDPError(
                f"Heavy repository {repository} is not available under {heavy_root}; run setup first."
            )
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise CVDPError(completed.stderr.strip() or f"Failed to clone {repository}")
        completed = subprocess.run(
            ["git", "checkout", "--quiet", commit],
            cwd=destination,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise CVDPError(completed.stderr.strip() or f"Cannot checkout {commit}")

    def materialize(self, workspace: Path, heavy_root: Path | None = None) -> Path:
        workspace = workspace.resolve()
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        code = workspace / "code"
        if self.heavy:
            if heavy_root is None:
                raise CVDPError(f"Task {self.id} requires --heavy-root")
            self._checkout_heavy(code, heavy_root.expanduser().resolve())
        else:
            code.mkdir()
            for relative, content in self.context.items():
                target = code / _safe_relative(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
        for relative, content in self.seed.items():
            target = code / _safe_relative(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        for relative, content in self.harness.items():
            target = workspace / _safe_relative(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        (workspace / "rundir").mkdir(exist_ok=True)
        return workspace


def parse_dotenv(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip("\"'")
    return result


def translate_container_paths(value: str, workspace: Path) -> str:
    replacements = {
        "/code": str(workspace / "code"),
        "/src": str(workspace / "src"),
        "/rundir": str(workspace / "rundir"),
    }
    result = value
    for old, new in replacements.items():
        result = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?=/|\b)", new, result)
    return result


def harness_environment(workspace: Path, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), env.get("PATH", "")]
    )
    dotenv_path = workspace / "src" / ".env"
    if dotenv_path.is_file():
        for key, value in parse_dotenv(dotenv_path.read_text(encoding="utf-8")).items():
            env[key] = translate_container_paths(value, workspace)
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "CVDP_RANDOM_SEED": "0",
            "COCOTB_RANDOM_SEED": "0",
            "RANDOM_SEED": "0",
        }
    )
    python_path = [str(workspace / "runtime"), str(workspace / "src")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    if overrides:
        env.update({key: translate_container_paths(value, workspace) for key, value in overrides.items()})
    return env


def harness_command(workspace: Path) -> str:
    compose_path = workspace / "docker-compose.yml"
    try:
        compose = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        services = compose["services"]
        service = next(iter(services.values()))
        command = service["command"]
    except (OSError, KeyError, StopIteration, TypeError, yaml.YAMLError) as exc:
        raise CVDPError(f"Cannot parse CVDP harness command: {compose_path}: {exc}") from exc
    if isinstance(command, list):
        command_text = shlex.join(str(item) for item in command)
    else:
        command_text = str(command).strip()
    shell_prefix = re.match(r"^/bin/(?:ba)?sh\s+-c\s+(.+)$", command_text, re.S)
    if shell_prefix:
        try:
            parsed = shlex.split(command_text)
            command_text = parsed[2]
        except (ValueError, IndexError):
            pass
    return translate_container_paths(command_text, workspace)


@dataclass(frozen=True)
class HarnessResult:
    passed: bool
    returncode: int
    stdout: str
    stderr: str
    command: str


@lru_cache(maxsize=1)
def bwrap_usable() -> bool:
    executable = shutil.which("bwrap")
    if not executable:
        return False
    completed = subprocess.run(
        [executable, "--die-with-parent", "--ro-bind", "/", "/", "/bin/true"],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


class NativeCocotbRunner:
    def __init__(self, timeout_seconds: int = 900, sandbox: bool = True) -> None:
        self.timeout_seconds = int(timeout_seconds)
        self.sandbox = sandbox

    def run(
        self, workspace: Path, overrides: Mapping[str, str] | None = None
    ) -> HarnessResult:
        workspace = workspace.resolve()
        runtime = workspace / "runtime"
        runtime.mkdir(exist_ok=True)
        (runtime / "sitecustomize.py").write_text(
            "import os, random\n"
            "random.seed(int(os.environ.get('CVDP_RANDOM_SEED', '0')))\n"
            "if os.environ.get('CVDP_FIXED_GATE_NETLIST') == '1':\n"
            "    import cocotb_tools.runner as _runner\n"
            "    _original_build = _runner.Runner.build\n"
            "    def _fixed_gate_build(self, *args, **kwargs):\n"
            "        kwargs['parameters'] = {}\n"
            "        return _original_build(self, *args, **kwargs)\n"
            "    _runner.Runner.build = _fixed_gate_build\n",
            encoding="utf-8",
        )
        command = harness_command(workspace)
        env = harness_environment(workspace, overrides)
        invocation = ["/bin/sh", "-c", command]
        if self.sandbox and bwrap_usable():
            invocation = [
                "bwrap",
                "--die-with-parent",
                "--unshare-net",
                "--ro-bind",
                "/",
                "/",
                "--bind",
                str(workspace),
                str(workspace),
                "--chdir",
                str(workspace / "rundir"),
                *invocation,
            ]
        process = subprocess.Popen(
            invocation,
            cwd=workspace / "rundir",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
            return HarnessResult(
                passed=process.returncode == 0,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
                command=command,
            )
        except subprocess.TimeoutExpired as exc:
            # The shell/pytest process can spawn a simulator such as vvp. Killing
            # only the immediate child leaves that simulator consuming a CPU
            # indefinitely, so terminate the session created above as a unit.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
            return HarnessResult(
                passed=False,
                returncode=124,
                stdout=stdout or str(exc.stdout or ""),
                stderr=(
                    f"Harness timed out after {self.timeout_seconds}s\n"
                    f"{stderr or exc.stderr or ''}"
                ),
                command=command,
            )


def project_sources(workspace: Path, env: Mapping[str, str]) -> list[Path]:
    raw_sources = env.get("VERILOG_SOURCES") or env.get("VHDL_SOURCES") or ""
    sources = [Path(item) for item in shlex.split(raw_sources)]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise CVDPError(f"Harness source files do not exist: {missing}")
    if not sources:
        sources = sorted((workspace / "code").rglob("*.sv")) + sorted(
            (workspace / "code").rglob("*.v")
        )
    if not sources:
        raise CVDPError("CVDP task has no synthesizable Verilog sources.")
    return list(dict.fromkeys(path.resolve() for path in sources))


def project_digest(workspace: Path, targets: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(targets):
        path = workspace / "code" / _safe_relative(relative)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
