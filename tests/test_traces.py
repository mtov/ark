from __future__ import annotations

from pathlib import Path

from ark import traces
from ark.models import TokenUsage
from ark.protocol import ToolRequest


def test_record_response_usage_accumulates_total_tokens(
    monkeypatch,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "agent_trace.log"
    monkeypatch.setattr(traces, "LOG_PATH", log_path)

    traces.clear_trace()
    traces.record_response_usage(
        token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    traces.record_response_usage(
        token_usage=TokenUsage(input_tokens=4, output_tokens=3, total_tokens=7),
    )

    assert traces.get_total_tokens() == 22


def test_get_total_tokens_returns_none_when_usage_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "agent_trace.log"
    monkeypatch.setattr(traces, "LOG_PATH", log_path)

    traces.clear_trace()
    traces.record_response_usage(token_usage=TokenUsage())

    assert traces.get_total_tokens() is None


def test_trace_finish_event_records_stage_and_detail(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "agent_trace.log"
    monkeypatch.setattr(traces, "LOG_PATH", log_path)

    traces.clear_trace()
    traces.trace_finish_event("failed", "finish_validation", "unexpected input")

    content = log_path.read_text(encoding="utf-8")

    assert "[finish]" in content
    assert "status: failed" in content
    assert "stage: finish_validation" in content
    assert "detail: unexpected input" in content


def test_trace_action_uses_one_line_for_action_and_arguments(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "agent_trace.log"
    monkeypatch.setattr(traces, "LOG_PATH", log_path)

    traces.clear_trace()
    traces.record_response_usage()
    traces.trace_action(
        ToolRequest(
            thought="Inspect the implementation.",
            name="read_file",
            args="src/products.py",
        )
    )

    content = log_path.read_text(encoding="utf-8")

    assert "[response 1]" in content
    assert "thought: Inspect the implementation." in content
    assert "action: read_file src/products.py" in content
    assert "action_input" not in content


def test_trace_action_records_edit_details_separately(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "agent_trace.log"
    monkeypatch.setattr(traces, "LOG_PATH", log_path)

    traces.clear_trace()
    traces.record_response_usage()
    traces.trace_action(
        ToolRequest(
            thought="Edit the file.",
            name="edit_file",
            args="path: file.py\nold:\n```\nold\n```\nnew:\n```\nnew\n```",
        )
    )

    content = log_path.read_text(encoding="utf-8")

    assert "action: edit_file path: file.py" in content
    assert "edit:\npath: file.py" in content


def test_trace_run_summary_records_total_tokens_and_elapsed_time(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "agent_trace.log"
    monkeypatch.setattr(traces, "LOG_PATH", log_path)

    traces.clear_trace()
    traces.record_response_usage(
        token_usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
    )
    traces.trace_run_summary(12.345, ["list_files", "read_file", "edit_file", "finish", "run_tests"])

    content = log_path.read_text(encoding="utf-8")

    assert "[run_summary]" in content
    assert "total_tokens: 5" in content
    assert "elapsed_seconds: 12.35" in content
    assert "tools_called: list_files, read_file, edit_file, finish, run_tests" in content
