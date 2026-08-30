from __future__ import annotations

from pathlib import Path

from ark.agentic_loop import FINISH_SUCCESS_MESSAGE, MAX_ITERATIONS_REACHED_MESSAGE, agentic_loop
from ark.finish_handler import ApplyFinishResult
from ark.inputs import AgentConfig
from ark.memory import Memory
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


def test_finish_retries_after_failed_tests_without_resetting_workspace(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    responses = iter([ToolRequest("first", "finish", ""), ToolRequest("done", "finish", "")])
    seen_histories: list[str] = []
    attempts = 0

    def next_request(_config: AgentConfig, memory: Memory) -> ToolRequest:
        nonlocal attempts
        attempts += 1
        seen_histories.append(memory.to_text())
        return next(responses)

    def finish(_config: AgentConfig, _request: ToolRequest) -> ApplyFinishResult:
        if attempts == 1:
            return ApplyFinishResult("post_apply_tests_failed", "1 failed")
        return ApplyFinishResult("completed")

    monkeypatch.setattr("ark.agentic_loop.get_next_tool_request", next_request)
    monkeypatch.setattr("ark.agentic_loop.apply_finish", finish)

    result = agentic_loop(context)

    assert result.status == "success"
    assert result.output == FINISH_SUCCESS_MESSAGE
    assert result.tools_called == ["finish", "run_tests", "finish", "run_tests"]
    assert "approved edits remain in the workspace" in seen_histories[1]


def test_approved_edit_is_kept_after_successful_finish(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    file_path = tmp_path / "example.py"
    file_path.write_text("value = old\n", encoding="utf-8")
    responses = iter([
        ToolRequest(
            "edit",
            "edit_file",
            "path: example.py\nold:\n```\nvalue = old\n```\nnew:\n```\nvalue = new\n```",
        ),
        ToolRequest("done", "finish", ""),
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    monkeypatch.setattr("ark.agentic_loop.get_next_tool_request", lambda _config, _memory: next(responses))
    monkeypatch.setattr("ark.agentic_loop.apply_finish", lambda *_args: ApplyFinishResult("completed"))

    result = agentic_loop(context)

    assert result.status == "success"
    assert result.tools_called == ["edit_file", "finish", "run_tests"]
    assert file_path.read_text(encoding="utf-8") == "value = new\n"
    assert context.snapshot_path is None


def test_max_iterations_rolls_back_transaction(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    file_path = tmp_path / "example.py"
    file_path.write_text("old\n", encoding="utf-8")
    snapshot = tmp_path.parent / "snapshot" / "workspace"
    snapshot.parent.mkdir()
    snapshot.mkdir()
    (snapshot / "example.py").write_text("old\n", encoding="utf-8")
    context.snapshot_path = snapshot
    file_path.write_text("new\n", encoding="utf-8")

    monkeypatch.setattr(
        "ark.agentic_loop.get_next_tool_request",
        lambda _config, _memory: ToolRequest("explore", "list_files", "."),
    )
    monkeypatch.setattr("ark.agentic_loop.MAX_ITERATIONS", 1)
    monkeypatch.setattr("ark.agentic_loop.run_tool_with_status", lambda *_args: ("files", None))

    result = agentic_loop(context)

    assert result.status == "max_iterations_reached"
    assert result.error == MAX_ITERATIONS_REACHED_MESSAGE
    assert file_path.read_text(encoding="utf-8") == "old\n"
    assert context.snapshot_path is None


def test_redundant_consecutive_read_is_still_skipped(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    responses = iter([
        ToolRequest("read", "read_file", "example.py"),
        ToolRequest("read again", "read_file", "example.py"),
        ToolRequest("done", "finish", ""),
    ])
    monkeypatch.setattr("ark.agentic_loop.get_next_tool_request", lambda _config, _memory: next(responses))
    monkeypatch.setattr("ark.agentic_loop.apply_finish", lambda *_args: ApplyFinishResult("completed"))

    result = agentic_loop(context)

    assert result.status == "success"
