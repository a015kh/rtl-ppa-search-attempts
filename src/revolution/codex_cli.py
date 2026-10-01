from __future__ import annotations

import json
import os
import subprocess
import tempfile
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CodexCLIError(RuntimeError):
    """A non-interactive Codex invocation failed or returned invalid output."""


@lru_cache(maxsize=1)
def _network_sandbox_usable() -> bool:
    """Codex's Linux sandbox requires a usable private network namespace."""
    try:
        completed = subprocess.run(
            [
                "bwrap",
                "--die-with-parent",
                "--unshare-net",
                "--ro-bind",
                "/",
                "/",
                "/bin/true",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


@dataclass(frozen=True)
class CodexCLIResult:
    output: dict[str, Any]
    thread_id: str
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    events: tuple[dict[str, Any], ...]


class CodexCLIRunner:
    """Run schema-constrained Codex turns and resume their persisted sessions."""

    def __init__(
        self,
        model: str,
        reasoning_effort: str = "medium",
        timeout_seconds: int = 3600,
        executable: str = "codex",
        cwd: str | Path | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = int(timeout_seconds)
        self.executable = executable
        self.cwd = Path(cwd or os.getcwd()).expanduser().resolve()

    @staticmethod
    def _effective_sandbox(sandbox: str) -> str:
        if sandbox != "danger-full-access" and not _network_sandbox_usable():
            return "danger-full-access"
        return sandbox

    def _common(self, sandbox: str) -> list[str]:
        return [
            "--ignore-user-config",
            "--strict-config",
            "-m",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "-c",
            'approval_policy="never"',
            "-c",
            f'sandbox_mode="{sandbox}"',
            "-c",
            'web_search="disabled"',
        ]

    @staticmethod
    def _parse_events(text: str) -> tuple[tuple[dict[str, Any], ...], str, dict[str, int]]:
        events: list[dict[str, Any]] = []
        thread_id: str | None = None
        usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
        }
        for line in text.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)
            if isinstance(event.get("thread_id"), str):
                thread_id = event["thread_id"]
            event_usage = event.get("usage")
            if event.get("type") == "turn.completed" and isinstance(event_usage, dict):
                for key in usage:
                    value = event_usage.get(key, 0)
                    if isinstance(value, int):
                        usage[key] += value
        if thread_id is None:
            raise CodexCLIError("Codex JSONL did not contain a thread_id.")
        return tuple(events), thread_id, usage

    def run(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        thread_id: str | None = None,
        cwd: str | Path | None = None,
        sandbox: str = "read-only",
    ) -> CodexCLIResult:
        run_cwd = Path(cwd or self.cwd).expanduser().resolve()
        sandbox = self._effective_sandbox(sandbox)
        if not (run_cwd / ".git").exists():
            raise CodexCLIError(f"Codex workspace must be a Git repository: {run_cwd}")

        with tempfile.TemporaryDirectory(prefix="revolution-codex-") as raw_tmp:
            tmp = Path(raw_tmp)
            schema_path = tmp / "output.schema.json"
            output_path = tmp / "output.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")

            if thread_id:
                command = [
                    self.executable,
                    "exec",
                    "resume",
                    *self._common(sandbox),
                    "--json",
                    "--output-schema",
                    str(schema_path),
                    "-o",
                    str(output_path),
                    thread_id,
                    prompt,
                ]
            else:
                command = [
                    self.executable,
                    "exec",
                    *self._common(sandbox),
                    "-C",
                    str(run_cwd),
                    "-s",
                    sandbox,
                    "--json",
                    "--output-schema",
                    str(schema_path),
                    "-o",
                    str(output_path),
                    prompt,
                ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=run_cwd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise CodexCLIError(f"Codex invocation failed: {exc}") from exc
            if completed.returncode:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise CodexCLIError(
                    f"Codex exited with status {completed.returncode}: {detail[-4000:]}"
                )
            events, returned_thread_id, usage = self._parse_events(completed.stdout)
            try:
                output = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CodexCLIError("Codex did not produce valid structured output.") from exc
            if not isinstance(output, dict):
                raise CodexCLIError("Codex structured output must be a JSON object.")
            return CodexCLIResult(
                output=output,
                thread_id=returned_thread_id,
                prompt_tokens=usage["input_tokens"],
                cached_prompt_tokens=usage["cached_input_tokens"],
                completion_tokens=usage["output_tokens"],
                reasoning_tokens=usage["reasoning_output_tokens"],
                events=events,
            )
