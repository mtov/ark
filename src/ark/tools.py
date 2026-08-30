from __future__ import annotations

import os
import subprocess
import sys
from difflib import unified_diff
from pathlib import Path

from .guards import resolve_tool_path
from .inputs import AgentConfig, create_workspace_snapshot
from .protocol import ToolRequest, parse_edit_file_request
from .traces import trace_edit_event

MAX_FIND_TEXT_MATCHES = 20
SKIPPED_DIRECTORIES = {".git", ".venv", "__pycache__"}
TEST_COMMAND = (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider")
TEST_TIMEOUT_SECONDS = 30
REDUNDANT_READ_FILE_MESSAGE = (
    "Redundant tool request skipped: your previous read_file call already read this same file. "
    "Do not request the same file again unless it changed or you need to verify a specific detail "
    "before finishing. Continue with a different action."
)


def _resolve_workspace_target(action_input: str, workspace_path: Path) -> Path | str:
    try:
        return resolve_tool_path(action_input, workspace_path)
    except ValueError as exc:
        return str(exc)


def list_files(action_input: str, workspace_path: Path) -> str:
    path = _resolve_workspace_target(action_input, workspace_path)
    if isinstance(path, str):
        return path

    if not path.exists():
        return f"Directory not found inside the workspace: {path}"
    if not path.is_dir():
        return f"Path is a file, not a directory. Use read_file instead: {path}"

    entries = sorted(item.name for item in path.iterdir())
    if not entries:
        return f"Directory is empty: {path}"

    return "\n".join(entries)


def read_file(action_input: str, workspace_path: Path) -> str:
    path = _resolve_workspace_target(action_input, workspace_path)
    if isinstance(path, str):
        return path

    if not path.exists():
        return f"File not found inside the workspace: {path}"
    if not path.is_file():
        return f"Path is not a file: {path}"

    return path.read_text(encoding="utf-8")


def parse_find_text_input(action_input: str) -> tuple[str, str] | str:
    try:
        query, directory = action_input.split("|", maxsplit=1)
    except ValueError:
        return "Invalid input. Use: search text | /path/to/directory"

    query = query.strip()
    directory = directory.strip()

    if not query:
        return "Search text cannot be empty."
    if not directory:
        return "Directory path cannot be empty."

    return query, directory


def find_text(action_input: str, workspace_path: Path) -> str:
    parsed_input = parse_find_text_input(action_input)
    if isinstance(parsed_input, str):
        return parsed_input

    query, directory = parsed_input
    path = _resolve_workspace_target(directory, workspace_path)
    if isinstance(path, str):
        return path

    if not path.exists():
        return f"Directory not found inside the workspace: {path}"
    if not path.is_dir():
        return f"Path is not a directory: {path}"

    matches: list[str] = []

    for file_path in sorted(path.rglob("*")):
        if any(part in SKIPPED_DIRECTORIES for part in file_path.parts):
            continue
        if not file_path.is_file():
            continue

        try:
            with file_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if query in line:
                        matches.append(f"{file_path}:{line_number}: {line.rstrip()}")
                        if len(matches) >= MAX_FIND_TEXT_MATCHES:
                            return "\n".join(matches)
        except (OSError, UnicodeDecodeError):
            continue

    if not matches:
        return f"No matches found for: {query}"

    return "\n".join(matches)


def run_tests_with_status(workspace_path: Path) -> tuple[bool, str]:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    print("Running tests...", flush=True)

    try:
        result = subprocess.run(
            TEST_COMMAND,
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
            env=environment,
        )
    except Exception as exc:  # noqa: BLE001
        print("Tests: failed", flush=True)
        return False, f"Could not run tests: {exc}"

    passed = result.returncode == 0
    if passed:
        print("Tests: passed", flush=True)
    else:
        print("Tests: failed", flush=True)

    output = "\n".join(
        part for part in (result.stdout.strip(), result.stderr.strip()) if part
    ).strip()

    if not output and passed:
        return True, "Tests passed."
    if not output:
        return False, f"Tests failed with exit code {result.returncode}."

    return passed, output


def run_tests(workspace_path: Path) -> str:
    _passed, output = run_tests_with_status(workspace_path)
    return output


def _build_edit_diff(path: str, old_text: str, new_text: str) -> str:
    return "".join(
        unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _request_edit_approval(path: str, diff: str) -> bool:
    print(f"Proposed edit: {path}")
    print(diff, end="" if diff.endswith("\n") else "\n")
    return input("Authorize edit? [y/N]: ").strip().lower() in {"y", "yes"}


def edit_file(action_input: str, config: AgentConfig) -> str:
    try:
        edit = parse_edit_file_request(action_input)
    except ValueError as exc:
        trace_edit_event("failed", "<invalid>", detail=str(exc))
        return str(exc)

    path = _resolve_workspace_target(edit.path, config.workspace_path)
    if isinstance(path, str):
        trace_edit_event("failed", edit.path, detail=path)
        return path
    if not path.exists():
        message = f"File not found inside the workspace: {path}"
        trace_edit_event("failed", edit.path, detail=message)
        return message
    if not path.is_file():
        message = f"Path is not a file: {path}"
        trace_edit_event("failed", edit.path, detail=message)
        return message

    current_text = path.read_text(encoding="utf-8")
    occurrences = current_text.count(edit.old)
    if occurrences != 1:
        message = (
            "edit_file requires old to appear exactly once in the target file; "
            f"found {occurrences} occurrences."
        )
        trace_edit_event("failed", edit.path, detail=message)
        return message

    updated_text = current_text.replace(edit.old, edit.new, 1)
    if updated_text == current_text:
        message = "edit_file requires new to differ from old."
        trace_edit_event("failed", edit.path, detail=message)
        return message

    diff = _build_edit_diff(edit.path, current_text, updated_text)
    if not _request_edit_approval(edit.path, diff):
        trace_edit_event("rejected", edit.path, diff=diff)
        return f"Edit rejected by user for {edit.path}."

    create_workspace_snapshot(config)
    path.write_text(updated_text, encoding="utf-8")
    trace_edit_event("succeeded", edit.path, diff=diff)
    return f"Edit applied successfully to {edit.path}."


def should_skip_redundant_tool_request(
    request: ToolRequest,
    previous_request: ToolRequest | None = None,
) -> str | None:
    if previous_request is None:
        return None

    if (
        request.name == "read_file"
        and previous_request.name == "read_file"
        and request.args.strip() == previous_request.args.strip()
    ):
        return REDUNDANT_READ_FILE_MESSAGE

    return None


def run_tool_with_status(
    request: ToolRequest,
    config: AgentConfig,
    previous_request: ToolRequest | None = None,
) -> tuple[str, str | None]:
    skipped_reason = should_skip_redundant_tool_request(request, previous_request)
    if skipped_reason is not None:
        return skipped_reason, "skipped: redundant"

    return run_tool(request, config, previous_request), None


def run_tool(
    request: ToolRequest,
    config: AgentConfig,
    previous_request: ToolRequest | None = None,
) -> str:
    workspace_path = config.workspace_path
    skipped_reason = should_skip_redundant_tool_request(request, previous_request)
    if skipped_reason is not None:
        return skipped_reason

    if request.name == "list_files":
        return list_files(request.args, workspace_path)
    if request.name == "read_file":
        return read_file(request.args, workspace_path)
    if request.name == "find_text":
        return find_text(request.args, workspace_path)
    if request.name == "run_tests":
        return run_tests(workspace_path)
    if request.name == "edit_file":
        return edit_file(request.args, config)

    return f"Unsupported action '{request.name}'. Use list_files, read_file, find_text, run_tests, edit_file, or finish."
