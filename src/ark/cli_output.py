from __future__ import annotations

from math import floor
from pathlib import Path

from .protocol import ToolCall
from .traces import get_total_tokens


def format_iteration_action(tool_call: ToolCall) -> str:
    if tool_call.name == "list_files":
        path = tool_call.args.strip() or "."
        return f"{tool_call.name} {path}"

    if tool_call.name == "read_file":
        return f"{tool_call.name} {Path(tool_call.args).name}"

    if tool_call.name == "find_text":
        query = tool_call.args.split("|", maxsplit=1)[0].strip()
        if query:
            return f'{tool_call.name} "{query}"'

    return tool_call.name


def print_tool_call(iteration: int, tool_call: ToolCall, note: str | None = None) -> None:
    suffix = f" ({note})" if note else ""
    print(f"[{iteration}] {format_iteration_action(tool_call)}{suffix}", flush=True)


def print_total_tokens() -> None:
    total_tokens = get_total_tokens()
    if total_tokens is not None:
        print(f"Total tokens: {total_tokens}")


def format_elapsed_time(elapsed_seconds: float) -> str:
    total_seconds = max(0.0, elapsed_seconds)
    if total_seconds < 60:
        return f"Elapsed time: {total_seconds:.2f}s"

    minutes = floor(total_seconds / 60)
    seconds = total_seconds - (minutes * 60)
    return f"Elapsed time: {minutes}m {seconds:.2f}s"


def print_elapsed_time(elapsed_seconds: float) -> None:
    print(format_elapsed_time(elapsed_seconds))


def format_success_message(result: str) -> str:
    normalized_result = result.strip()
    if not normalized_result:
        return "Ark result: success."

    return f"Ark result: success. {normalized_result}"


def format_failure_message(error: Exception) -> str:
    return f"Ark result: failed. {error}"


def print_failure_summary(error: Exception, elapsed_seconds: float) -> None:
    print_total_tokens()
    print_elapsed_time(elapsed_seconds)
    print(format_failure_message(error))


def print_success_summary(result: str, elapsed_seconds: float) -> None:
    print_total_tokens()
    print_elapsed_time(elapsed_seconds)
    print_final_result(result)


def print_final_result(result: str) -> None:
    print(format_success_message(result))
