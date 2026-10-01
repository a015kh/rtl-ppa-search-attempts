import json

import pytest

from revolution.codex_cli import CodexCLIError, CodexCLIRunner


def test_parse_codex_events_collects_thread_and_usage():
    events = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 7,
                        "output_tokens": 4,
                        "reasoning_output_tokens": 2,
                    },
                }
            ),
        ]
    )
    parsed, thread_id, usage = CodexCLIRunner._parse_events(events)
    assert thread_id == "thread-1"
    assert len(parsed) == 2
    assert usage == {
        "input_tokens": 10,
        "cached_input_tokens": 7,
        "output_tokens": 4,
        "reasoning_output_tokens": 2,
    }


def test_parse_codex_events_requires_thread_id():
    with pytest.raises(CodexCLIError, match="thread_id"):
        CodexCLIRunner._parse_events('{"type":"turn.completed","usage":{}}')


def test_codex_uses_full_access_when_network_sandbox_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "revolution.codex_cli._network_sandbox_usable", lambda: False
    )
    runner = CodexCLIRunner("test-model")
    effective = runner._effective_sandbox("workspace-write")
    common = runner._common(effective)
    assert effective == "danger-full-access"
    assert 'sandbox_mode="danger-full-access"' in common
