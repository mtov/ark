from __future__ import annotations

from pathlib import Path

from ark.finish_handler import INVALID_FINISH_MESSAGE, apply_finish
from ark.inputs import AgentConfig
from ark.models import ModelConfig
from ark.protocol import ToolRequest


def build_context(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        model_config=ModelConfig("openai-compatible", 30, None, "model", "OPENAI_API_KEY"),
        system_prompt="system",
        user_prompt="prompt",
        source_workspace_path=tmp_path,
        workspace_path=tmp_path,
    )


def test_finish_runs_final_tests(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("ark.finish_handler.run_tests_with_status", lambda _path: (True, "3 passed"))
    events: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr("ark.finish_handler.trace_finish_event", lambda *args: events.append(args))

    result = apply_finish(build_context(tmp_path), ToolRequest("done", "finish", ""))

    assert result.status == "completed"
    assert events == [("completed", "finish")]


def test_finish_returns_failed_tests_without_reverting_edits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("ark.finish_handler.run_tests_with_status", lambda _path: (False, "1 failed"))

    result = apply_finish(build_context(tmp_path), ToolRequest("done", "finish", ""))

    assert result.status == "post_apply_tests_failed"
    assert result.test_output == "1 failed"


def test_finish_rejects_nonempty_input(tmp_path: Path) -> None:
    result = apply_finish(build_context(tmp_path), ToolRequest("done", "finish", "patch"))

    assert result.status == "invalid_finish"
    assert INVALID_FINISH_MESSAGE == "Finish action must have an empty Action Input."
