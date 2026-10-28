from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from ark import tools
from ark.inputs import Context, commit_workspace_changes
from ark.memory import MAX_HISTORY_ENTRIES, MAX_OBSERVATION_CHARS
from ark.models import Model
from ark.protocol import ToolCall


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


def test_list_files_suggests_read_file_when_path_is_a_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = tools.list_files("example.txt", tmp_path)

    assert result == f"Path is a file, not a directory. Use read_file instead: {file_path}"


def test_read_file_returns_file_contents(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = tools.read_file("example.txt", tmp_path)

    assert result.output == "File: example.txt\nLines: 1-1 of 1\n1: hello"
    assert result.file_path == file_path.resolve()


def test_list_files_returns_recursive_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "src" / "main" / "java" / "OrderService.java"
    source.parent.mkdir(parents=True)
    source.write_text("class OrderService {}", encoding="utf-8")
    for name in (".git", ".venv", "__pycache__", "target"):
        ignored = tmp_path / name
        ignored.mkdir()
        (ignored / "hidden.txt").write_text("hidden", encoding="utf-8")

    output = tools.list_files(".", tmp_path)

    assert "src/main/java/OrderService.java" in output.splitlines()
    assert "src/main/java/" in output.splitlines()
    assert "hidden.txt" not in output
    assert str(tmp_path) not in output
    assert tools.list_files("src/main", tmp_path).endswith("src/main/java/OrderService.java")


def test_list_files_reports_truncation_at_entry_limit(tmp_path: Path) -> None:
    for index in range(tools.MAX_LIST_ENTRIES + 1):
        (tmp_path / f"file_{index:03}.txt").touch()

    output = tools.list_files(".", tmp_path)

    assert len(output.splitlines()) == tools.MAX_LIST_ENTRIES + 1
    assert output.splitlines()[0] == "file_000.txt"
    assert "Listing truncated" in output.splitlines()[-1]
    assert "subdirectory" in output.splitlines()[-1]


def test_list_files_keeps_truncation_notice_inside_memory_budget(tmp_path: Path) -> None:
    for index in range(100):
        (tmp_path / (f"{index:03}_" + "x" * 150)).touch()

    output = tools.list_files(".", tmp_path)

    assert len(output) <= MAX_OBSERVATION_CHARS
    assert output.endswith("Use list_files on a subdirectory shown above.")


def test_listing_and_search_do_not_follow_external_symlinks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("needle", encoding="utf-8")
    (workspace / "outside-link.txt").symlink_to(outside)
    (workspace / "loop").symlink_to(workspace, target_is_directory=True)

    assert "outside-link" not in tools.list_files(".", workspace)
    assert "No matches found" in tools.find_text("needle | .", workspace)
    assert tools.read_file("outside-link.txt", workspace).file_path is None


def test_read_file_returns_inclusive_line_range(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text("first\nsecond\nthird\nfourth\n", encoding="utf-8")

    result = tools.read_file("source.txt | 2:3", tmp_path)

    assert result.output == "File: source.txt\nLines: 2-3 of 4\n2: second\n3: third"
    assert result.file_path == (tmp_path / "source.txt").resolve()
    assert "4: fourth" in tools.read_file("source.txt | 3:100", tmp_path).output


@pytest.mark.parametrize("action_input", ["", "source.txt | 0:2", "source.txt | 3:2", "source.txt | 1:",
                                       "source.txt | -1:2", "source.txt | abc", "../outside.txt"])
def test_read_file_rejects_invalid_requests(tmp_path: Path, action_input: str) -> None:
    (tmp_path / "source.txt").write_text("contents", encoding="utf-8")
    assert tools.read_file(action_input, tmp_path).file_path is None


def test_read_file_handles_empty_files_and_out_of_bounds_ranges(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").touch()
    assert "empty" in tools.read_file("empty.txt", tmp_path).output
    (tmp_path / "source.txt").write_text("one line", encoding="utf-8")
    result = tools.read_file("source.txt | 2:5", tmp_path)
    assert result.file_path is None
    assert "exceeds file length" in result.output


def test_long_file_can_be_read_completely_using_continuations(tmp_path: Path) -> None:
    lines = [f"content {number}: " + "x" * 90 for number in range(300)]
    (tmp_path / "long.txt").write_text("\n".join(lines), encoding="utf-8")
    context = build_context(tmp_path)
    action_input = "long.txt"
    recovered = []
    for iteration in range(1, 20):
        call = ToolCall("read", "read_file", action_input)
        result = tools.run_tool(call, context)
        assert result.skipped is False
        assert len(result.output) <= MAX_OBSERVATION_CHARS
        recovered.extend(re.findall(r"^\d+: (.*)$", result.output, flags=re.MULTILINE))
        context.memory.append(iteration, call, result.output, file_path=result.file_path)
        if "Read truncated." not in result.output:
            break
        continuation = result.output.splitlines()[-1]
        assert continuation in context.memory.to_text()
        action_input = continuation.split("Continue with read_file: ", 1)[1]
    assert recovered == lines


def test_oversized_single_line_is_reported_without_caching(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text("x" * (MAX_OBSERVATION_CHARS + 1), encoding="utf-8")
    result = tools.read_file("long.txt", tmp_path)
    assert result.file_path is None
    assert "Line 1 is too long" in result.output
    assert len(result.output) <= MAX_OBSERVATION_CHARS


def test_failed_read_does_not_block_retry(tmp_path: Path) -> None:
    context = build_context(tmp_path)
    call = ToolCall("read", "read_file", "missing.txt")
    failed = tools.run_tool(call, context)
    context.memory.append(1, call, failed.output, file_path=failed.file_path)
    (tmp_path / "missing.txt").write_text("now available", encoding="utf-8")
    result = tools.run_tool(call, context)
    assert result.skipped is False
    assert "now available" in result.output


def test_edit_invalidates_all_retained_ranges_and_path_aliases(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("old\nsecond line\n", encoding="utf-8")
    context = build_context(tmp_path)
    for iteration, action_input in enumerate(["./source.txt | 1:1", f"{source} | 2:2"], start=1):
        call = ToolCall("read", "read_file", action_input)
        result = tools.run_tool(call, context)
        context.memory.append(iteration, call, result.output, file_path=result.file_path)
    for iteration in range(3, MAX_HISTORY_ENTRIES + 4):
        context.memory.append(iteration, ToolCall("list", "list_files", "."), "files")
    assert "1: old" in context.memory.to_text()
    assert tools.run_tool(ToolCall("read", "read_file", "source.txt | 1:1"), context).skipped is True
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    edit = ToolCall("edit", "edit_file", "path: source.txt\nold:\n```\nold\n```\nnew:\n```\nnew\n```")
    result = tools.run_tool(edit, context)
    context.memory.append(8, edit, result.output, file_path=result.file_path)
    assert "1: old" not in context.memory.to_text()
    for action_input in ["./source.txt | 1:1", f"{source} | 2:2"]:
        assert tools.run_tool(ToolCall("read", "read_file", action_input), context).skipped is False
    assert "1: new" in tools.run_tool(ToolCall("read", "read_file", "source.txt | 1:1"), context).output
    commit_workspace_changes(context)


def test_run_tool_skips_file_read_already_in_recent_context(tmp_path: Path) -> None:
    context = build_context(tmp_path)
    context.memory.append(
        1, ToolCall("inspect", "read_file", "example.txt"), "hello", file_path=tmp_path / "example.txt"
    )

    result = tools.run_tool(ToolCall("inspect again", "read_file", "example.txt"), context)

    assert result.output == (
        "This read_file result is already available in the recent context. "
        "Use the existing observation instead of reading the file again."
    )
    assert result.note == "skipped: already in context"
    assert result.skipped is True


def test_run_tests_records_test_result(monkeypatch, tmp_path: Path) -> None:
    events: list[tuple[str, str | None]] = []
    monkeypatch.setattr(tools, "run_tests_with_status", lambda _context: (False, "1 failed"))
    monkeypatch.setattr(
        tools,
        "trace_test_event",
        lambda status, detail=None: events.append((status, detail)),
    )

    output = tools.run_tests(build_context(tmp_path))

    assert output == "1 failed"
    assert events == [("failed", "1 failed")]


@pytest.mark.parametrize(("passed", "summary"), [(True, "1 passed"), (False, "1 failed")])
def test_pytest_runner_executes_workspace_tests(tmp_path: Path, passed, summary) -> None:
    (tmp_path / "test_example.py").write_text(f"def test_example():\n    assert {passed}\n", encoding="utf-8")

    succeeded, output = tools.run_tests_with_status(build_context(tmp_path))

    assert succeeded is passed
    assert summary in output
    assert not (tmp_path / ".pytest_cache").exists()
    assert not (tmp_path / "__pycache__").exists()


@pytest.mark.skipif(os.name != "posix", reason="./mvnw requires a POSIX shell")
@pytest.mark.parametrize("exit_code", [0, 1])
def test_maven_runner_executes_wrapper_from_workspace(tmp_path: Path, exit_code: int) -> None:
    wrapper = tmp_path / "mvnw"
    wrapper.write_text(
        f'#!/bin/sh\nprintf "argument=%s\\n" "$1"\npwd\nexit {exit_code}\n', encoding="utf-8"
    )
    wrapper.chmod(0o755)
    context = build_context(tmp_path)
    context.test_runner = "maven"
    context.test_timeout_seconds = 300

    passed, output = tools.run_tests_with_status(context)

    assert passed is (exit_code == 0)
    assert "argument=test" in output
    assert str(tmp_path.resolve()) in output


@pytest.mark.parametrize("wrapper_exists", [False, True])
def test_maven_runner_reports_unusable_wrapper(tmp_path: Path, wrapper_exists: bool) -> None:
    context = build_context(tmp_path)
    context.test_runner = "maven"
    if wrapper_exists:
        wrapper = tmp_path / "mvnw"
        wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        wrapper.chmod(0o644)

    passed, output = tools.run_tests_with_status(context)

    assert passed is False
    assert ("not executable" if wrapper_exists else "not found") in output


@pytest.mark.parametrize(("runner", "timeout"), [("pytest", 45), ("maven", 600)])
def test_run_tests_uses_context_command_and_timeout(monkeypatch, tmp_path: Path, runner, timeout) -> None:
    context = build_context(tmp_path)
    context.test_runner = runner
    context.test_timeout_seconds = timeout
    if runner == "maven":
        wrapper = tmp_path / "mvnw"
        wrapper.touch()
        wrapper.chmod(0o755)
    executions = []

    def run(command, execution_context, environment):
        executions.append((command, execution_context, execution_context.test_timeout_seconds))
        return subprocess.CompletedProcess(command, 0, "passed", "")

    monkeypatch.setattr(tools, "_run_test_command", run)
    result = tools.run_tool(ToolCall("test", "run_tests", ""), context)

    assert result.output == "passed"
    expected = ("./mvnw", "test") if runner == "maven" else tools.TEST_COMMANDS["pytest"]
    assert executions == [(expected, context, timeout)]


@pytest.mark.parametrize(("runner", "timeout"), [("pytest", 30), ("maven", 300)])
def test_run_tests_reports_timeout_with_partial_output(monkeypatch, tmp_path: Path, runner, timeout) -> None:
    context = build_context(tmp_path)
    context.test_runner = runner
    context.test_timeout_seconds = timeout
    monkeypatch.setattr(tools, "get_test_command", lambda _context: ("command",))

    def run(command, _context, _environment):
        raise subprocess.TimeoutExpired(command, timeout, output=b"compiling\n", stderr=b"diagnostic\n")

    monkeypatch.setattr(tools, "_run_test_command", run)
    passed, output = tools.run_tests_with_status(context)

    assert passed is False
    assert output == f"Tests timed out after {timeout} seconds using {runner}.\ncompiling\ndiagnostic"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_timeout_kills_process_group_and_preserves_output(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path)
    context.test_timeout_seconds = 1
    command = (
        sys.executable, "-u", "-c",
        "import os, subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']); "
        "print(os.getpid(), flush=True); time.sleep(10)",
    )
    killed = []
    killpg = os.killpg

    def kill_group(pid, sig):
        killed.append((pid, sig))
        killpg(pid, sig)

    monkeypatch.setattr(tools.os, "killpg", kill_group)
    with pytest.raises(subprocess.TimeoutExpired) as error:
        tools._run_test_command(command, context, os.environ.copy())

    pid = int(error.value.stdout.strip())
    assert killed == [(pid, signal.SIGKILL)]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_find_text_skips_maven_build_output(tmp_path: Path) -> None:
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "generated.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "source.txt").write_text("needle", encoding="utf-8")

    output = tools.find_text("needle | .", tmp_path)

    assert "source.txt" in output
    assert "generated.txt" not in output


def test_edit_file_applies_approved_replacement(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "example.py"
    file_path.write_text("value = old\n", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    result = tools.edit_file(
        "path: example.py\nold:\n```python\nvalue = old\n```\nnew:\n```python\nvalue = new\n```",
        build_context(tmp_path),
    )

    assert result == "Edit applied successfully to example.py."
    assert file_path.read_text(encoding="utf-8") == "value = new\n"


def test_edit_file_rejects_ambiguous_old_content(tmp_path: Path) -> None:
    file_path = tmp_path / "example.py"
    file_path.write_text("old\nold\n", encoding="utf-8")

    result = tools.edit_file(
        "path: example.py\nold:\n```\nold\n```\nnew:\n```\nnew\n```",
        build_context(tmp_path),
    )

    assert "found 2 occurrences" in result
    assert file_path.read_text(encoding="utf-8") == "old\nold\n"


def test_edit_file_rejects_a_noop_replacement(tmp_path: Path) -> None:
    file_path = tmp_path / "example.py"
    file_path.write_text("old\n", encoding="utf-8")

    result = tools.edit_file(
        "path: example.py\nold:\n```\nold\n```\nnew:\n```\nold\n```",
        build_context(tmp_path),
    )

    assert result == "edit_file requires new to differ from old."
    assert file_path.read_text(encoding="utf-8") == "old\n"


def test_edit_file_keeps_file_unchanged_when_rejected(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "example.py"
    file_path.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    result = tools.edit_file(
        "path: example.py\nold:\n```\nold\n```\nnew:\n```\nnew\n```",
        build_context(tmp_path),
    )

    assert result == "Edit rejected by user for example.py."
    assert file_path.read_text(encoding="utf-8") == "old\n"
