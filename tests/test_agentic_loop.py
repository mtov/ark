from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ark.agentic_loop import (
    FINISH_SUCCESS_MESSAGE,
    FINISH_WITHOUT_EDIT_MESSAGE,
    INVALID_FINISH_MESSAGE,
    MAX_ITERATIONS_REACHED_MESSAGE,
    LoopResult,
    agentic_loop,
    invoke_model,
    main,
    try_finish,
)
from ark.inputs import Context
from ark.memory import Memory
from ark.models import Model, build_model_response
from ark.protocol import REPAIR_PROMPT, ToolCall
from ark.tools import ToolResult


def build_context(tmp_path: Path) -> Context:
    return Context(
        model=Model(
            name="model",
            system_prompt="system",
            timeout_seconds=30,
            api_key_env="OPENAI_API_KEY",
        ),
        user_prompt="prompt",
        workspace_path=tmp_path,
    )


def tool_call_sequence(*tool_calls: ToolCall):
    responses = iter(tool_calls)

    def next_tool_call(context: Context) -> ToolCall:
        tool_call = next(responses)
        context.memory.record_tool_call(tool_call.name)
        return tool_call

    return next_tool_call


@pytest.mark.parametrize(
    ("loop_result", "exit_code", "message"),
    [
        (LoopResult(status="success"), 0, FINISH_SUCCESS_MESSAGE),
        (LoopResult(status="max_iterations_reached"), 1, MAX_ITERATIONS_REACHED_MESSAGE),
    ],
)
def test_main_reports_tool_calls_from_context(
    monkeypatch,
    capsys,
    tmp_path: Path,
    loop_result: LoopResult,
    exit_code: int,
    message: str,
) -> None:
    context = build_context(tmp_path)
    context.memory.record_tool_call("read_file")
    context.memory.record_tool_call("finish")
    summaries: list[list[str]] = []
    monkeypatch.setattr(
        "ark.agentic_loop.parse_args",
        lambda: SimpleNamespace(workspace_path=str(tmp_path), test_runner="maven", test_timeout=600),
    )
    prepared = []

    def prepare_run(path: str, **options) -> Context:
        prepared.append((path, options))
        return context

    monkeypatch.setattr("ark.agentic_loop.prepare_run", prepare_run)
    monkeypatch.setattr("ark.agentic_loop.agentic_loop", lambda _context: loop_result)
    monkeypatch.setattr(
        "ark.agentic_loop.trace_run_summary",
        lambda _elapsed, tools_called: summaries.append(tools_called.copy()),
    )

    assert main() == exit_code
    assert prepared == [(str(tmp_path), {"test_runner": "maven", "test_timeout": 600})]
    assert summaries == [["read_file", "finish"]]
    assert message in capsys.readouterr().out


def test_invoke_model_records_tool_call(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    monkeypatch.setattr(
        Model,
        "call",
        lambda _model, _prompt, _history: build_model_response(
            "Thought: inspect\nAction: read_file\nAction Input: example.py"
        ),
    )
    monkeypatch.setattr("ark.agentic_loop.trace_action", lambda _request: None)

    tool_call = invoke_model(context)

    assert tool_call.name == "read_file"
    assert context.memory.tools_called == ["read_file"]


def test_invoke_model_records_repaired_tool_call(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    responses = iter(["Thought: invalid", "Thought: recover\nAction: list_files\nAction Input: ."])
    requests: list[tuple[str, str]] = []

    def call(_model: Model, user_prompt: str, history: str):
        requests.append((user_prompt, history))
        return build_model_response(next(responses))

    monkeypatch.setattr(Model, "call", call)
    monkeypatch.setattr("ark.agentic_loop.trace_validation_error", lambda *_args: None)
    monkeypatch.setattr("ark.agentic_loop.trace_action", lambda _request: None)

    tool_call = invoke_model(context)

    assert tool_call == ToolCall("recover", "list_files", ".")
    assert context.memory.tools_called == ["list_files"]
    assert requests == [
        (context.user_prompt, "No previous steps."),
        (context.user_prompt, f"No previous steps.\n\n{REPAIR_PROMPT}"),
    ]


def test_finish_retries_after_failed_tests_without_resetting_workspace(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    responses = iter([
        ToolCall("first finish", "finish", ""),
        ToolCall("second finish", "finish", ""),
    ])
    seen_histories: list[str] = []

    def next_tool_call(context: Context) -> ToolCall:
        seen_histories.append(context.memory.to_text())
        tool_call = next(responses)
        context.memory.record_tool_call(tool_call.name)
        return tool_call

    test_results = iter([(False, "1 failed"), (True, "3 passed")])
    monkeypatch.setattr("ark.agentic_loop.invoke_model", next_tool_call)
    monkeypatch.setattr("ark.agentic_loop.run_tests_with_status", lambda _context: next(test_results))
    monkeypatch.setattr(Memory, "has_successful_edit", lambda _memory: True)

    result = agentic_loop(context)

    assert result.status == "success"
    assert context.memory.tools_called == ["finish", "run_tests", "finish", "run_tests"]
    assert "1 failed" in seen_histories[1]


def test_finish_requires_an_approved_edit(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    finish_call = ToolCall("done", "finish", "")
    test_calls: list[Context] = []
    monkeypatch.setattr(
        "ark.agentic_loop.run_tests_with_status",
        lambda context: test_calls.append(context),
    )

    result = try_finish(context, 1, finish_call)

    assert result is False
    assert test_calls == []
    assert context.memory.entries[-1].result == FINISH_WITHOUT_EDIT_MESSAGE


@pytest.mark.parametrize(("runner", "timeout"), [("pytest", 30), ("maven", 600)])
def test_finish_runs_final_tests(monkeypatch, tmp_path: Path, runner, timeout) -> None:
    context = build_context(tmp_path)
    context.test_runner = runner
    context.test_timeout_seconds = timeout
    test_calls: list[Context] = []
    events: list[tuple] = []

    def run_tests(context: Context) -> tuple[bool, str]:
        test_calls.append(context)
        return True, "3 passed"

    monkeypatch.setattr(Memory, "has_successful_edit", lambda _memory: True)
    monkeypatch.setattr("ark.agentic_loop.run_tests_with_status", run_tests)
    monkeypatch.setattr("ark.agentic_loop.trace_finish_event", lambda *args: events.append(args))

    assert try_finish(context, 1, ToolCall("done", "finish", "")) is True
    assert test_calls == [context]
    assert context.memory.tools_called == ["run_tests"]
    assert events == [("completed", "finish")]


@pytest.mark.parametrize("test_output", ["1 failed", "Tests timed out after 300 seconds using maven."])
def test_finish_returns_failed_tests_without_reverting_edits(monkeypatch, tmp_path: Path, test_output) -> None:
    context = build_context(tmp_path)
    file_path = tmp_path / "example.py"
    file_path.write_text("approved edit\n", encoding="utf-8")
    events: list[tuple] = []
    monkeypatch.setattr(Memory, "has_successful_edit", lambda _memory: True)
    monkeypatch.setattr("ark.agentic_loop.run_tests_with_status", lambda _context: (False, test_output))
    monkeypatch.setattr("ark.agentic_loop.trace_finish_event", lambda *args: events.append(args))

    assert try_finish(context, 1, ToolCall("done", "finish", "")) is False
    assert context.memory.entries[-1].result == test_output
    assert context.memory.tools_called == ["run_tests"]
    assert file_path.read_text(encoding="utf-8") == "approved edit\n"
    assert events == [("failed", "post_apply_tests", test_output)]


def test_finish_rejects_nonempty_input(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    test_calls: list[Context] = []
    events: list[tuple] = []
    monkeypatch.setattr(Memory, "has_successful_edit", lambda _memory: True)
    monkeypatch.setattr("ark.agentic_loop.run_tests_with_status", lambda context: test_calls.append(context))
    monkeypatch.setattr("ark.agentic_loop.trace_finish_event", lambda *args: events.append(args))

    assert try_finish(context, 1, ToolCall("done", "finish", "patch")) is False
    assert test_calls == []
    assert context.memory.tools_called == []
    assert context.memory.entries[-1].result == INVALID_FINISH_MESSAGE
    assert events == [("failed", "finish_validation", INVALID_FINISH_MESSAGE)]


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
    monkeypatch.setattr("ark.agentic_loop.invoke_model", tool_call_sequence(*responses))
    monkeypatch.setattr("ark.agentic_loop.run_tests_with_status", lambda _context: (True, "3 passed"))

    result = agentic_loop(context)

    assert result.status == "success"
    assert context.memory.tools_called == ["edit_file", "finish", "run_tests"]
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
    monkeypatch.setattr("ark.agentic_loop.run_tool", lambda *_args: ToolResult("files"))

    result = agentic_loop(context)

    assert result.status == "max_iterations_reached"
    assert result.error is None
    assert file_path.read_text(encoding="utf-8") == "old\n"
    assert context.snapshot_path is None


def test_agentic_loop_returns_failure(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    monkeypatch.setattr(
        "ark.agentic_loop.invoke_model",
        lambda _context: (_ for _ in ()).throw(ValueError("missing content")),
    )

    result = agentic_loop(context)

    assert result.status == "failed"
    assert result.error == "missing content"
    assert result.error_type == "ValueError"
    assert context.memory.tools_called == []
