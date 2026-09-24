from __future__ import annotations

from pathlib import Path

from ark.agentic_loop import (
    FINISH_SUCCESS_MESSAGE,
    FINISH_WITHOUT_EDIT_MESSAGE,
    LoopResult,
    MAX_ITERATIONS_REACHED_MESSAGE,
    Memory,
    agentic_loop,
    handle_finish,
    invoke_model,
)
from ark.finish_handler import ApplyFinishResult
from ark.inputs import AgentConfig
from ark.models import ModelConfig, build_model_response
from ark.protocol import ToolCall
from ark.tools import ToolResult


def build_context(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        model_config=ModelConfig("openai-compatible", 30, None, "model", "OPENAI_API_KEY"),
        system_prompt="system",
        user_prompt="prompt",
        source_workspace_path=tmp_path,
        workspace_path=tmp_path,
    )


def tool_call_sequence(*tool_calls: ToolCall):
    responses = iter(tool_calls)

    def next_tool_call(_config: AgentConfig, memory: Memory) -> ToolCall:
        tool_call = next(responses)
        memory.record_tool_call(tool_call.name)
        return tool_call

    return next_tool_call


def test_loop_result_copies_tool_calls_from_memory() -> None:
    memory = Memory(tools_called=["read_file"])

    result = LoopResult.success("done", memory)
    memory.record_tool_call("finish")

    assert result.tools_called == ["read_file"]


def test_invoke_model_records_tool_call(monkeypatch, tmp_path: Path) -> None:
    memory = Memory()
    monkeypatch.setattr(
        "ark.agentic_loop.call_model_api",
        lambda *_args: build_model_response(
            "Thought: inspect\nAction: read_file\nAction Input: example.py"
        ),
    )
    monkeypatch.setattr("ark.agentic_loop.trace_action", lambda _request: None)

    tool_call = invoke_model(build_context(tmp_path), memory)

    assert tool_call.name == "read_file"
    assert memory.tools_called == ["read_file"]


def test_invoke_model_records_repaired_tool_call(monkeypatch, tmp_path: Path) -> None:
    memory = Memory()
    repaired_call = ToolCall("recover", "list_files", ".")
    monkeypatch.setattr(
        "ark.agentic_loop.call_model_api",
        lambda *_args: build_model_response("Thought: invalid"),
    )
    monkeypatch.setattr(
        "ark.agentic_loop.repair_response",
        lambda *_args: repaired_call,
    )
    monkeypatch.setattr("ark.agentic_loop.trace_validation_error", lambda *_args: None)
    monkeypatch.setattr("ark.agentic_loop.trace_action", lambda _request: None)

    tool_call = invoke_model(build_context(tmp_path), memory)

    assert tool_call is repaired_call
    assert memory.tools_called == ["list_files"]


def test_finish_retries_after_failed_tests_without_resetting_workspace(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    responses = iter([
        ToolCall("first finish", "finish", ""),
        ToolCall("second finish", "finish", ""),
    ])
    seen_histories: list[str] = []
    attempts = 0

    def next_tool_call(_config: AgentConfig, memory: Memory) -> ToolCall:
        nonlocal attempts
        attempts += 1
        seen_histories.append(memory.to_text())
        tool_call = next(responses)
        memory.record_tool_call(tool_call.name)
        return tool_call

    def finish(_config: AgentConfig, _tool_call: ToolCall) -> ApplyFinishResult:
        if attempts == 1:
            return ApplyFinishResult("post_apply_tests_failed", "1 failed")
        return ApplyFinishResult("completed")

    monkeypatch.setattr("ark.agentic_loop.invoke_model", next_tool_call)
    monkeypatch.setattr("ark.agentic_loop.apply_finish", finish)
    monkeypatch.setattr(Memory, "has_successful_edit", lambda _memory: True)

    result = agentic_loop(context)

    assert result.status == "success"
    assert result.output == FINISH_SUCCESS_MESSAGE
    assert result.tools_called == ["finish", "run_tests", "finish", "run_tests"]
    assert "approved edits remain in the workspace" in seen_histories[1]


def test_finish_requires_an_approved_edit(monkeypatch, tmp_path: Path) -> None:
    memory = Memory()
    finish_call = ToolCall("done", "finish", "")
    apply_calls: list[ToolCall] = []
    monkeypatch.setattr(
        "ark.agentic_loop.apply_finish",
        lambda _config, tool_call: apply_calls.append(tool_call),
    )

    result = handle_finish(build_context(tmp_path), memory, 1, finish_call)

    assert result is None
    assert apply_calls == []
    assert memory.entries[-1].result == FINISH_WITHOUT_EDIT_MESSAGE


def test_approved_edit_is_kept_after_successful_finish(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    file_path = tmp_path / "example.py"
    file_path.write_text("value = old\n", encoding="utf-8")
    responses = iter([
        ToolCall(
            "edit",
            "edit_file",
            "path: example.py\nold:\n```\nvalue = old\n```\nnew:\n```\nvalue = new\n```",
        ),
        ToolCall("done", "finish", ""),
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    monkeypatch.setattr(
        "ark.agentic_loop.invoke_model",
        tool_call_sequence(*responses),
    )
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
        "ark.agentic_loop.invoke_model",
        tool_call_sequence(ToolCall("explore", "list_files", ".")),
    )
    monkeypatch.setattr("ark.agentic_loop.MAX_ITERATIONS", 1)
    monkeypatch.setattr(
        "ark.agentic_loop.run_tool",
        lambda *_args: ToolResult("files"),
    )

    result = agentic_loop(context)

    assert result.status == "max_iterations_reached"
    assert result.error == MAX_ITERATIONS_REACHED_MESSAGE
    assert file_path.read_text(encoding="utf-8") == "old\n"
    assert context.snapshot_path is None


def test_agentic_loop_returns_failure_with_tool_history(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    monkeypatch.setattr(
        "ark.agentic_loop.invoke_model",
        lambda _config, _memory: (_ for _ in ()).throw(ValueError("missing content")),
    )

    result = agentic_loop(context)

    assert result.status == "failed"
    assert result.error == "missing content"
    assert result.error_type == "ValueError"
    assert result.tools_called == []


def test_redundant_consecutive_read_is_still_skipped(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    responses = iter([
        ToolCall("read", "read_file", "example.py"),
        ToolCall("read again", "read_file", "example.py"),
        ToolCall("done", "finish", ""),
    ])
    monkeypatch.setattr(
        "ark.agentic_loop.invoke_model",
        tool_call_sequence(*responses),
    )
    monkeypatch.setattr("ark.agentic_loop.apply_finish", lambda *_args: ApplyFinishResult("completed"))
    monkeypatch.setattr(Memory, "has_successful_edit", lambda _memory: True)

    result = agentic_loop(context)

    assert result.status == "success"
