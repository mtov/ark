from __future__ import annotations

from pathlib import Path

from ark import traces
from ark.models import TokenUsage


def test_trace_response_records_responses_and_cumulative_total(
    monkeypatch,
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "agent_trace.log"
    monkeypatch.setattr(traces, "LOG_PATH", log_path)

    traces.clear_trace()
    traces.trace_response(
        "first response",
        token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    traces.trace_response(
        "second response",
        token_usage=TokenUsage(input_tokens=4, output_tokens=3, total_tokens=7),
    )

    content = log_path.read_text(encoding="utf-8")

    assert "[response 1]" in content
    assert "first response" in content
    assert "[response 2]" in content
    assert "second response" in content
    assert traces.get_total_tokens() == 22


def test_get_total_tokens_returns_none_when_usage_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "agent_trace.log"
    monkeypatch.setattr(traces, "LOG_PATH", log_path)

    traces.clear_trace()
    traces.trace_response("response without usage", token_usage=TokenUsage())

    assert traces.get_total_tokens() is None


def test_trace_finish_event_records_stage_and_detail(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "agent_trace.log"
    monkeypatch.setattr(traces, "LOG_PATH", log_path)

    traces.clear_trace()
    traces.trace_finish_event("failed", "patch_validation", "corrupt patch")

    content = log_path.read_text(encoding="utf-8")

    assert "[finish]" in content
    assert "status: failed" in content
    assert "stage: patch_validation" in content
    assert "detail: corrupt patch" in content


def test_trace_command_event_records_result_details(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "agent_trace.log"
    monkeypatch.setattr(traces, "LOG_PATH", log_path)

    traces.clear_trace()
    traces.trace_command_event(
        "failed",
        "git apply patch.txt",
        tmp_path,
        exit_code=1,
        detail="patch does not apply",
    )

    content = log_path.read_text(encoding="utf-8")

    assert "[command]" in content
    assert "status: failed" in content
    assert "exit_code: 1" in content
    assert "detail: patch does not apply" in content


def test_trace_run_summary_records_total_tokens_and_elapsed_time(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "agent_trace.log"
    monkeypatch.setattr(traces, "LOG_PATH", log_path)

    traces.clear_trace()
    traces.trace_response(
        "response",
        token_usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
    )
    traces.trace_run_summary(12.345, ["list_files", "read_file", "finish", "apply_patch", "run_tests"])

    content = log_path.read_text(encoding="utf-8")

    assert "[run_summary]" in content
    assert "total_tokens: 5" in content
    assert "elapsed_seconds: 12.35" in content
    assert "tools_called: list_files, read_file, finish, apply_patch, run_tests" in content
