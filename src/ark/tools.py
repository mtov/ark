from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Iterator

from .guards import resolve_tool_path
from .inputs import Context, create_workspace_snapshot
from .memory import MAX_OBSERVATION_CHARS
from .protocol import ToolCall, parse_edit_file_request, parse_read_file_request
from .traces import trace_edit_event, trace_test_event


MAX_FIND_TEXT_MATCHES = 20
MAX_LIST_ENTRIES = 100
SKIPPED_DIRECTORIES = {".git", ".venv", "__pycache__", "target"}
TEST_COMMANDS = {
    "pytest": (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"),
    "maven": ("./mvnw", "test"),
}


@dataclass
class ToolResult:
    output: str
    note: str | None = None
    skipped: bool = False
    file_path: Path | None = None


def _resolve_workspace_target(action_input: str, workspace_path: Path) -> Path | str:
    try:
        return resolve_tool_path(action_input, workspace_path)
    except ValueError as exc:
        return str(exc)


def _workspace_entries(path: Path, workspace_path: Path) -> Iterator[Path]:
    for directory, directories, files in os.walk(path):
        directories[:] = sorted(
            name for name in directories
            if name not in SKIPPED_DIRECTORIES and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(directories + files):
            entry = Path(directory) / name
            try:
                resolve_tool_path(str(entry), workspace_path)
            except ValueError:
                continue
            yield entry


def list_files(action_input: str, workspace_path: Path) -> str:
    path = _resolve_workspace_target(action_input, workspace_path)
    if isinstance(path, str):
        return path

    if not path.exists():
        return f"Directory not found inside the workspace: {path}"
    if not path.is_dir():
        return f"Path is a file, not a directory. Use read_file instead: {path}"

    entries: list[str] = []
    footer = "Listing truncated. Use list_files on a subdirectory shown above."
    length = 0
    for entry in _workspace_entries(path, workspace_path):
        label = entry.relative_to(workspace_path.resolve()).as_posix()
        if entry.is_dir():
            label += "/"
        if len(entries) == MAX_LIST_ENTRIES or length + len(label) + 1 > MAX_OBSERVATION_CHARS - len(footer) - 1:
            return "\n".join([*entries, footer])
        entries.append(label)
        length += len(label) + 1
    if not entries:
        return f"Directory is empty: {path}"

    return "\n".join(entries)


def read_file(action_input: str, workspace_path: Path) -> ToolResult:
    try:
        request = parse_read_file_request(action_input)
    except ValueError as exc:
        return ToolResult(str(exc))
    path = _resolve_workspace_target(request.path, workspace_path)
    if isinstance(path, str):
        return ToolResult(path)

    if not path.exists():
        return ToolResult(f"File not found inside the workspace: {path}")
    if not path.is_file():
        return ToolResult(f"Path is not a file: {path}")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return ToolResult(f"Could not read file: {path}: {exc}")
    label = path.relative_to(workspace_path.resolve()).as_posix()
    if not lines and request.end is None:
        return ToolResult(f"File: {label}\nFile is empty.", file_path=path)
    if request.start > len(lines):
        return ToolResult(f"Start line {request.start} exceeds file length ({len(lines)} lines).")
    end = min(request.end or len(lines), len(lines))
    header = f"File: {label}\nLines: {request.start}-{end} of {len(lines)}\n"
    footer = f"\nRead truncated. Continue with read_file: {label} | {end}:{end}"
    budget = MAX_OBSERVATION_CHARS - len(header) - len(footer)
    rendered: list[str] = []
    length = 0
    for number in range(request.start, end + 1):
        line = f"{number}: {lines[number - 1]}"
        if length + len(line) + 1 > budget:
            break
        rendered.append(line)
        length += len(line) + 1
    if not rendered:
        return ToolResult(
            f"Line {request.start} is too long to fit in one observation ({MAX_OBSERVATION_CHARS} characters). "
            "Request a later line range to read the rest of the file."
        )
    last = request.start + len(rendered) - 1
    header = f"File: {label}\nLines: {request.start}-{last} of {len(lines)}\n"
    output = header + "\n".join(rendered)
    if last < end:
        output += f"\nRead truncated. Continue with read_file: {label} | {last + 1}:{end}"
    return ToolResult(output, file_path=path)


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

    for file_path in _workspace_entries(path, workspace_path):
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


def get_test_command(context: Context) -> tuple[str, ...]:
    if context.test_runner == "maven":
        wrapper = context.workspace_path / "mvnw"
        if not wrapper.is_file():
            raise FileNotFoundError("Maven wrapper not found: expected mvnw in the workspace root.")
        if not os.access(wrapper, os.X_OK):
            raise PermissionError("Maven wrapper is not executable. Run chmod +x mvnw in the source workspace.")
    return TEST_COMMANDS[context.test_runner]


def _run_test_command(
    command: tuple[str, ...], context: Context, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    with subprocess.Popen(
        command,
        cwd=context.workspace_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=os.name == "posix",
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=context.test_timeout_seconds)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                command, context.test_timeout_seconds, output=stdout, stderr=stderr
            ) from None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _test_output(*streams: str | bytes | None) -> str:
    parts = [stream.decode(errors="replace") if isinstance(stream, bytes) else stream for stream in streams]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def run_tests_with_status(context: Context) -> tuple[bool, str]:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    print(f"Running tests ({context.test_runner}, timeout: {context.test_timeout_seconds}s)...", flush=True)

    try:
        command = get_test_command(context)
        result = _run_test_command(command, context, environment)
    except subprocess.TimeoutExpired as exc:
        print("Tests: failed", flush=True)
        message = f"Tests timed out after {context.test_timeout_seconds} seconds using {context.test_runner}."
        return False, _test_output(message, exc.stdout, exc.stderr)
    except Exception as exc:  # noqa: BLE001
        print("Tests: failed", flush=True)
        return False, f"Could not run tests: {exc}"

    passed = result.returncode == 0
    print("Tests: passed" if passed else "Tests: failed", flush=True)

    output = _test_output(result.stdout, result.stderr)

    if not output and passed:
        return True, "Tests passed."
    if not output:
        return False, f"Tests failed with exit code {result.returncode}."

    return passed, output


def run_tests(context: Context) -> str:
    passed, output = run_tests_with_status(context)
    trace_test_event("passed" if passed else "failed", None if passed else output)
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


def edit_file(action_input: str, context: Context) -> str:
    try:
        edit = parse_edit_file_request(action_input)
    except ValueError as exc:
        trace_edit_event("failed", "<invalid>", detail=str(exc))
        return str(exc)

    path = _resolve_workspace_target(edit.path, context.workspace_path)
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

    create_workspace_snapshot(context)
    path.write_text(updated_text, encoding="utf-8")
    trace_edit_event("succeeded", edit.path, diff=diff)
    return f"Edit applied successfully to {edit.path}."


def run_tool(
    tool_call: ToolCall,
    context: Context,
) -> ToolResult:
    workspace_path = context.workspace_path

    if tool_call.name == "list_files":
        return ToolResult(list_files(tool_call.args, workspace_path))
    if tool_call.name == "read_file":
        if context.memory.has_current_file_read(tool_call.args, workspace_path):
            return ToolResult(
                "This read_file result is already available in the recent context. "
                "Use the existing observation instead of reading the file again.",
                note="skipped: already in context",
                skipped=True,
            )
        return read_file(tool_call.args, workspace_path)
    if tool_call.name == "find_text":
        return ToolResult(find_text(tool_call.args, workspace_path))
    if tool_call.name == "run_tests":
        return ToolResult(run_tests(context))
    if tool_call.name == "edit_file":
        output = edit_file(tool_call.args, context)
        file_path = None
        if output.startswith("Edit applied successfully to "):
            file_path = resolve_tool_path(parse_edit_file_request(tool_call.args).path, workspace_path)
        return ToolResult(output, file_path=file_path)

    return ToolResult(
        f"Unsupported action '{tool_call.name}'. Use list_files, read_file, find_text, run_tests, edit_file, or finish."
    )
