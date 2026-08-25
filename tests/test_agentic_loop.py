from __future__ import annotations

from pathlib import Path

from ark.agentic_loop import (
    FULL_FILE_REQUIRED_MESSAGE,
    INVALID_FINISH_MESSAGE,
    MAX_ITERATIONS_REACHED_MESSAGE,
    REDUNDANT_READ_FILE_MESSAGE,
    LoopResult,
    Memory,
    agentic_loop,
    summarize_patch_failure,
)
from ark.cli_output import (
    format_elapsed_time,
    format_failure_message,
    format_success_message,
    print_elapsed_time,
    print_final_result,
)
from ark.finish_handler import FinishResult
from ark.inputs import AgentConfig
from ark.models import ModelConfig
from ark.protocol import ToolRequest
from ark.test_failures import summarize_test_failure_output


def build_context() -> AgentConfig:
    return AgentConfig(
        model_config=ModelConfig(
            model="openai-compatible",
            timeout_seconds=30,
            openai_base_url=None,
            openai_model="gpt-5.4-mini",
            openai_api_key_env="OPENAI_API_KEY",
        ),
        system_prompt="system",
        user_prompt="prompt",
        source_workspace_path=Path("/tmp/source"),
        workspace_path=Path("/tmp/runtime"),
    )


def test_format_success_message_for_unified_diff_patch() -> None:
    result = format_success_message(
        "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n",
        LoopResult.success("--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n"),
    )

    assert result == "Ark result: success. Patch applied successfully."


def test_format_success_message_mentions_post_apply_tests_when_available() -> None:
    result = format_success_message(
        "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n",
        LoopResult.success(
            "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n",
            post_apply_tests_passed=True,
        ),
    )

    assert result == "Ark result: success. Patch applied successfully. Post-apply tests passed."


def test_format_success_message_for_non_patch_results() -> None:
    result = format_success_message(
        "Task completed successfully.",
        LoopResult.success("Task completed successfully."),
    )

    assert result == "Ark result: success. Task completed successfully."


def test_format_success_message_for_empty_results() -> None:
    result = format_success_message("  ", LoopResult.success("  "))

    assert result == "Ark result: success."


def test_format_failure_message() -> None:
    result = format_failure_message(ValueError("Post-apply tests failed"))

    assert result == "Ark result: failed. Post-apply tests failed"


def test_format_elapsed_time_under_one_minute() -> None:
    assert format_elapsed_time(12.345) == "Elapsed time: 12.35s"


def test_format_elapsed_time_over_one_minute() -> None:
    assert format_elapsed_time(75.4321) == "Elapsed time: 1m 15.43s"


def test_print_final_result_prints_summary_for_unified_diff_patches(capsys) -> None:
    print_final_result(
        "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n",
        LoopResult.success("--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n"),
    )

    captured = capsys.readouterr()

    assert captured.out == "Ark result: success. Patch applied successfully.\n"


def test_print_final_result_mentions_post_apply_tests_when_available(capsys) -> None:
    print_final_result(
        "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n",
        LoopResult.success(
            "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n",
            post_apply_tests_passed=True,
        ),
    )

    captured = capsys.readouterr()

    assert captured.out == "Ark result: success. Patch applied successfully. Post-apply tests passed.\n"


def test_print_final_result_prints_summary_for_non_patch_results(capsys) -> None:
    print_final_result(
        "Task completed successfully.",
        LoopResult.success("Task completed successfully."),
    )

    captured = capsys.readouterr()

    assert captured.out == "Ark result: success. Task completed successfully.\n"


def test_print_elapsed_time(capsys) -> None:
    print_elapsed_time(3.5)

    captured = capsys.readouterr()

    assert captured.out == "Elapsed time: 3.50s\n"


def test_agentic_loop_retries_after_invalid_finish(monkeypatch, capsys) -> None:
    context = build_context()
    responses = iter(
        [
            ToolRequest(thought="need more context", name="finish", args="I am done."),
            ToolRequest(
                thought="done",
                name="finish",
                args="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n",
            ),
        ]
    )
    seen_histories: list[str] = []

    def fake_get_next_tool_request(_context: AgentConfig, history: Memory) -> ToolRequest:
        seen_histories.append(history.to_text())
        return next(responses)

    monkeypatch.setattr("ark.agentic_loop.get_next_tool_request", fake_get_next_tool_request)
    monkeypatch.setattr(
        "ark.agentic_loop.apply_finish",
        lambda _context, tool_request: FinishResult(
            status="applied",
            request=tool_request,
            tools_called=["apply_patch", "run_tests"],
        ),
    )

    result = agentic_loop(context)
    captured = capsys.readouterr()

    assert result.status == "success"
    assert result.output is not None
    assert result.output.startswith("--- a/file.py")
    assert "[1] finish" in captured.out
    assert "[2] finish" in captured.out
    assert seen_histories[0] == "No previous steps."
    assert INVALID_FINISH_MESSAGE in seen_histories[1]


def test_agentic_loop_retries_after_post_apply_test_failure(monkeypatch, capsys) -> None:
    context = build_context()
    responses = iter(
        [
            ToolRequest(
                thought="first try",
                name="finish",
                args="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+attempt1\n",
            ),
            ToolRequest(
                thought="second try",
                name="finish",
                args="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+attempt2\n",
            ),
        ]
    )
    seen_histories: list[str] = []
    reset_calls: list[Path] = []
    finish_attempts = 0

    def fake_get_next_tool_request(_context: AgentConfig, history: Memory) -> ToolRequest:
        seen_histories.append(history.to_text())
        return next(responses)

    def fake_apply_finish(_context: AgentConfig, tool_request: ToolRequest) -> FinishResult:
        nonlocal finish_attempts
        finish_attempts += 1
        if finish_attempts == 1:
            return FinishResult(
                status="post_apply_tests_failed",
                request=tool_request,
                test_output="..F\nassert 1 == 2",
                tools_called=["apply_patch", "run_tests"],
            )
        return FinishResult(
            status="applied",
            request=tool_request,
            tools_called=["apply_patch", "run_tests"],
        )

    def fake_reset_runtime_workspace(runtime_context: AgentConfig) -> None:
        reset_calls.append(runtime_context.source_workspace_path)
        runtime_context.workspace_path = Path("/tmp/reset-runtime")

    monkeypatch.setattr("ark.agentic_loop.get_next_tool_request", fake_get_next_tool_request)
    monkeypatch.setattr("ark.agentic_loop.apply_finish", fake_apply_finish)
    monkeypatch.setattr("ark.agentic_loop.reset_runtime_workspace", fake_reset_runtime_workspace)
    monkeypatch.setattr("ark.agentic_loop.run_tool_with_status", lambda _tool_request, _context, _previous_request=None: ("file contents", None))

    result = agentic_loop(context)
    captured = capsys.readouterr()

    assert result.status == "success"
    assert result.output is not None
    assert result.output.endswith("+attempt2\n")
    assert "[1] finish" in captured.out
    assert "[2] finish" in captured.out
    assert reset_calls == [Path("/tmp/source")]
    assert context.workspace_path == Path("/tmp/reset-runtime")
    assert "Post-apply tests failed. The runtime workspace has been reset to the original source state." in seen_histories[1]
    assert "Use the failed test details below to produce a different patch." in seen_histories[1]
    assert "..F\nassert 1 == 2" in seen_histories[1]


def test_agentic_loop_retries_after_patch_validation_failure(monkeypatch, capsys) -> None:
    context = build_context()
    responses = iter(
        [
            ToolRequest(
                thought="first try",
                name="finish",
                args="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+attempt1\n",
            ),
            ToolRequest(
                thought="second try",
                name="finish",
                args="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+attempt2\n",
            ),
        ]
    )
    seen_histories: list[str] = []
    finish_attempts = 0

    def fake_get_next_tool_request(_context: AgentConfig, history: Memory) -> ToolRequest:
        seen_histories.append(history.to_text())
        return next(responses)

    def fake_apply_finish(_context: AgentConfig, tool_request: ToolRequest) -> FinishResult:
        nonlocal finish_attempts
        finish_attempts += 1
        if finish_attempts == 1:
            raise ValueError("Patch validation failed for src/pricing.py")
        return FinishResult(
            status="applied",
            request=tool_request,
            tools_called=["apply_patch", "run_tests"],
        )

    monkeypatch.setattr("ark.agentic_loop.get_next_tool_request", fake_get_next_tool_request)
    monkeypatch.setattr("ark.agentic_loop.apply_finish", fake_apply_finish)

    result = agentic_loop(context)
    captured = capsys.readouterr()

    assert result.status == "success"
    assert result.output is not None
    assert result.output.endswith("+attempt2\n")
    assert "[1] finish" in captured.out
    assert "[2] finish" in captured.out
    assert summarize_patch_failure(ValueError("Patch validation failed for src/pricing.py")) in seen_histories[1]


def test_agentic_loop_requires_full_file_after_two_patch_validation_failures(monkeypatch, capsys) -> None:
    context = build_context()
    responses = iter(
        [
            ToolRequest(
                thought="first try",
                name="finish",
                args="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+attempt1\n",
            ),
            ToolRequest(
                thought="second try",
                name="finish",
                args="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+attempt2\n",
            ),
            ToolRequest(
                thought="third try",
                name="finish",
                args="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+attempt3\n",
            ),
            ToolRequest(
                thought="fallback",
                name="finish",
                args="FILE: file.py\n```python\nnew contents\n```\n",
            ),
        ]
    )
    seen_histories: list[str] = []
    finish_attempts = 0

    def fake_get_next_tool_request(_context: AgentConfig, history: Memory) -> ToolRequest:
        seen_histories.append(history.to_text())
        return next(responses)

    def fake_apply_finish(_context: AgentConfig, tool_request: ToolRequest) -> FinishResult:
        nonlocal finish_attempts
        finish_attempts += 1
        if tool_request.args.startswith("FILE:"):
            return FinishResult(
                status="applied",
                request=tool_request,
                tools_called=["apply_patch", "run_tests"],
            )
        raise ValueError(f"Patch validation failed for attempt {finish_attempts}")

    monkeypatch.setattr("ark.agentic_loop.get_next_tool_request", fake_get_next_tool_request)
    monkeypatch.setattr("ark.agentic_loop.apply_finish", fake_apply_finish)

    result = agentic_loop(context)
    captured = capsys.readouterr()

    assert result.status == "success"
    assert result.output == "FILE: file.py\n```python\nnew contents\n```\n"
    assert "[1] finish" in captured.out
    assert "[2] finish" in captured.out
    assert "[3] finish" in captured.out
    assert FULL_FILE_REQUIRED_MESSAGE in seen_histories[3]


def test_summarize_test_failure_output_extracts_failed_cases() -> None:
    output = """.....FF                                                                  [100%]
=================================== FAILURES ===================================
_________________ test_multiple_groups_discount_multiple_units _________________

>       assert calculate_order_total(items, "BUY2GET50") == 75.0
E       AssertionError: assert 77.5 == 75.0

tests/test_checkout.py:57: AssertionError
_______________________ test_rounds_only_the_final_total _______________________

>       assert calculate_order_total(items, "BUY2GET50") == 44.97
E       AssertionError: assert 44.98 == 44.97

tests/test_checkout.py:66: AssertionError
=========================== short test summary info ============================
FAILED tests/test_checkout.py::test_multiple_groups_discount_multiple_units - AssertionError: assert 77.5 == 75.0
FAILED tests/test_checkout.py::test_rounds_only_the_final_total - AssertionError: assert 44.98 == 44.97
"""

    summary = summarize_test_failure_output(output)

    assert "Use the failed test details below to produce a different patch." in summary
    assert "- tests/test_checkout.py::test_multiple_groups_discount_multiple_units: AssertionError: assert 77.5 == 75.0" in summary
    assert "- test_multiple_groups_discount_multiple_units: expected 75.0, got 77.5" in summary
    assert "- tests/test_checkout.py::test_rounds_only_the_final_total: AssertionError: assert 44.98 == 44.97" in summary
    assert "- test_rounds_only_the_final_total: expected 44.97, got 44.98" in summary


def test_agentic_loop_returns_error_result_when_max_iterations_reached(monkeypatch) -> None:
    context = build_context()

    monkeypatch.setattr(
        "ark.agentic_loop.get_next_tool_request",
        lambda _context, _history: ToolRequest(
            thought="still exploring",
            name="list_files",
            args=".",
        ),
    )
    monkeypatch.setattr("ark.agentic_loop.run_tool_with_status", lambda _tool_request, _context, _previous_request=None: ("src\ntests", None))

    result = agentic_loop(context)

    assert result.status == "max_iterations_reached"
    assert result.output is None
    assert result.error == MAX_ITERATIONS_REACHED_MESSAGE


def test_agentic_loop_records_finish_internal_tools_in_tools_called(monkeypatch) -> None:
    context = build_context()

    monkeypatch.setattr(
        "ark.agentic_loop.get_next_tool_request",
        lambda _context, _history: ToolRequest(
            thought="done",
            name="finish",
            args="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n",
        ),
    )
    monkeypatch.setattr(
        "ark.agentic_loop.apply_finish",
        lambda _context, tool_request: FinishResult(
            status="applied",
            request=tool_request,
            tools_called=["apply_patch", "run_tests"],
        ),
    )

    result = agentic_loop(context)

    assert result.tools_called == ["finish", "apply_patch", "run_tests"]


def test_agentic_loop_short_circuits_redundant_consecutive_read_file(monkeypatch, capsys) -> None:
    context = build_context()
    responses = iter(
        [
            ToolRequest(thought="inspect", name="read_file", args="src/products.py"),
            ToolRequest(thought="inspect again", name="read_file", args="src/products.py"),
            ToolRequest(
                thought="done",
                name="finish",
                args="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n",
            ),
        ]
    )
    seen_histories: list[str] = []

    def fake_get_next_tool_request(_context: AgentConfig, history: Memory) -> ToolRequest:
        seen_histories.append(history.to_text())
        return next(responses)

    read_file_calls: list[str] = []

    def fake_read_file(_action_input: str, _workspace_path: Path) -> str:
        read_file_calls.append(_action_input)
        return "file contents"

    monkeypatch.setattr("ark.agentic_loop.get_next_tool_request", fake_get_next_tool_request)
    monkeypatch.setattr("ark.tools.read_file", fake_read_file)
    monkeypatch.setattr(
        "ark.agentic_loop.apply_finish",
        lambda _context, tool_request: FinishResult(
            status="applied",
            request=tool_request,
            tools_called=["apply_patch", "run_tests"],
        ),
    )

    result = agentic_loop(context)
    captured = capsys.readouterr()

    assert result.status == "success"
    assert read_file_calls == ["src/products.py"]
    assert "[1] read_file products.py" in captured.out
    assert "[2] read_file products.py (skipped: redundant)" in captured.out
    assert REDUNDANT_READ_FILE_MESSAGE in seen_histories[2]
