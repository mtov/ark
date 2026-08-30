from __future__ import annotations

from math import floor
from pathlib import Path

from .protocol import ToolRequest
from .traces import get_total_tokens


def format_iteration_action(tool_request: ToolRequest) -> str:
    if tool_request.name == "list_files":
        path = tool_request.args.strip() or "."
        return f"{tool_request.name} {path}"

    if tool_request.name == "read_file":
        return f"{tool_request.name} {Path(tool_request.args).name}"

    if tool_request.name == "find_text":
        query = tool_request.args.split("|", maxsplit=1)[0].strip()
        if query:
            return f'{tool_request.name} "{query}"'

    return tool_request.name


def print_iteration_action(iteration: int, tool_request: ToolRequest, note: str | None = None) -> None:
    suffix = f" ({note})" if note else ""
    print(f"[{iteration}] {format_iteration_action(tool_request)}{suffix}", flush=True)


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


def print_final_result(result: str) -> None:
    print(format_success_message(result))
