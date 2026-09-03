from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TokenUsage
    from .protocol import ToolRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "agent_trace.log"
RESPONSE_COUNT = 0
TOTAL_TOKENS: int | None = None


def _append_trace(text: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(text)


def _append_event(event_type: str, lines: list[str]) -> None:
    content = "\n".join(lines)
    _append_trace(f"[{event_type}]\n{content}\n\n")


def _append_optional_field(lines: list[str], label: str, value: str | None) -> None:
    if value:
        lines.append(f"{label}: {value}")


def _append_optional_block(lines: list[str], label: str, value: str | None) -> None:
    if value:
        lines.extend((f"{label}:", value))


def clear_trace() -> None:
    global RESPONSE_COUNT, TOTAL_TOKENS
    RESPONSE_COUNT = 0
    TOTAL_TOKENS = None
    LOG_PATH.write_text("", encoding="utf-8")


def trace_request(user_prompt: str) -> None:
    _append_event("request", [user_prompt])


def get_total_tokens() -> int | None:
    return TOTAL_TOKENS


def record_response_usage(token_usage: TokenUsage | None = None) -> None:
    global RESPONSE_COUNT, TOTAL_TOKENS
    RESPONSE_COUNT += 1
    total_tokens = token_usage.total_tokens if token_usage is not None else None
    if total_tokens is not None:
        TOTAL_TOKENS = (TOTAL_TOKENS or 0) + total_tokens


def trace_validation_error(reason: str, response: str) -> None:
    _append_event(
        "validation_error",
        [f"reason: {reason}", "response:", response],
    )


def _edit_path(action_input: str) -> str | None:
    for line in action_input.splitlines():
        if line.startswith("path:"):
            return line.removeprefix("path:").strip() or None
    return None


def _format_action(tool_request: ToolRequest) -> str:
    args = tool_request.args.strip()
    action = tool_request.name
    if action == "list_files":
        return f"{action} {args or '.'}"
    if action == "edit_file":
        path = _edit_path(args)
        if path is not None:
            return f"{action} {path}"
    if args and action != "finish":
        return f"{action} {args}"
    return action


def trace_action(tool_request: ToolRequest) -> None:
    lines = [
        f"thought: {tool_request.thought}",
        f"action: {_format_action(tool_request)}",
    ]
    _append_event(f"response {RESPONSE_COUNT}", lines)


def trace_repair_attempt(repair_kind: str, reason: str) -> None:
    _append_event("repair_attempt", [f"kind: {repair_kind}", f"reason: {reason}"])


def trace_edit_event(
    status: str,
    path: str,
    diff: str | None = None,
    detail: str | None = None,
) -> None:
    lines = [f"status: {status}", f"path: {path}"]
    _append_optional_block(lines, "diff", diff)
    _append_optional_field(lines, "detail", detail)
    _append_event("edit_file", lines)


def trace_finish_event(status: str, stage: str, detail: str | None = None) -> None:
    lines = [f"status: {status}", f"stage: {stage}"]
    _append_optional_field(lines, "detail", detail)
    _append_event("finish", lines)


def trace_test_event(status: str, detail: str | None = None) -> None:
    lines = [f"status: {status}"]
    _append_optional_block(lines, "detail", detail)
    _append_event("tests", lines)


def trace_error(
    stage: str,
    error: str,
    error_type: str,
    tools_called: list[str],
) -> None:
    _append_event(
        "error",
        [
            f"stage: {stage}",
            f"type: {error_type}",
            f"detail: {error}",
            f"tools_called: {', '.join(tools_called)}",
        ],
    )


def _format_tool_counts(tools_called: list[str]) -> str:
    return ", ".join(
        f"{tool}={count}" for tool, count in Counter(tools_called).items()
    )


def trace_run_summary(elapsed_seconds: float, tools_called: list[str]) -> None:
    _append_event(
        "run_summary",
        [
            f"total_tokens: {get_total_tokens()}",
            f"elapsed_seconds: {elapsed_seconds:.2f}",
            f"tool_counts: {_format_tool_counts(tools_called)}",
        ],
    )
