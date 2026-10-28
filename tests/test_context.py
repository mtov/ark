from __future__ import annotations

from pathlib import Path

from ark.memory import MAX_HISTORY_ENTRIES, MAX_OBSERVATION_CHARS, MAX_RETAINED_READS, Memory
from ark.protocol import ToolCall, parse_read_file_request

WORKSPACE = Path("/workspace")


def record_read(history: Memory, iteration: int, action_input: str, contents: str) -> None:
    request = parse_read_file_request(action_input)
    history.append(
        iteration, ToolCall("inspect", "read_file", action_input), contents,
        file_path=(WORKSPACE / request.path).resolve(),
    )


def test_agent_history_to_text_omits_thought_and_limits_entries() -> None:
    history = Memory()

    for iteration in range(1, MAX_HISTORY_ENTRIES + MAX_RETAINED_READS + 3):
        history.append(
            iteration,
            ToolCall(thought=f"thought {iteration}", name="read_file", args=f"file_{iteration}.py"),
            f"contents {iteration}",
            file_path=WORKSPACE / f"file_{iteration}.py",
        )

    text = history.to_text()

    assert "Thought:" not in text
    assert "Iteration 1" not in text
    assert f"Iteration {MAX_HISTORY_ENTRIES + MAX_RETAINED_READS + 2}" in text
    assert "Files already read:" in text
    assert f"- file_{MAX_HISTORY_ENTRIES + MAX_RETAINED_READS + 2}.py" in text


def test_agent_history_to_text_truncates_long_observations() -> None:
    history = Memory()
    long_observation = "x" * (MAX_OBSERVATION_CHARS + 50)
    history.append(
        1,
        ToolCall(thought="inspect", name="read_file", args="demo.py"),
        long_observation,
    )

    text = history.to_text()

    assert "Observation: " in text
    assert long_observation not in text
    assert "..." in text


def test_agent_history_summarizes_edit_arguments() -> None:
    history = Memory()
    history.append(
        1,
        ToolCall(
            thought="edit",
            name="edit_file",
            args="path: src/orders.py\nold:\n```\nold content\n```\nnew:\n```\nnew content\n```",
        ),
        "Edit applied successfully to src/orders.py.",
    )

    text = history.to_text()

    assert "Tool Args: path: src/orders.py" in text
    assert "old content" not in text
    assert "new content" not in text


def test_agent_history_to_text_summarizes_unique_reads_and_searches() -> None:
    history = Memory()
    record_read(history, 1, "src/a.py", "a")
    record_read(history, 2, "src/a.py", "a")
    history.append(3, ToolCall(thought="search", name="find_text", args="coupon | src"), "match")
    history.append(4, ToolCall(thought="test", name="run_tests", args=""), "passed")

    text = history.to_text()

    assert text.count("- src/a.py") == 1
    assert "Searches already run:" in text
    assert "- coupon | src" in text
    assert "Tests already run: yes" in text


def test_agent_history_contains_tool_uses_structured_entries() -> None:
    history = Memory()
    history.append(1, ToolCall(thought="inspect", name="find_text", args="needle | ."), "match")

    assert history.contains_tool("find_text") is True
    assert history.contains_tool("run_tests") is False


def test_memory_records_tool_calls_in_order() -> None:
    history = Memory()

    history.record_tool_call("read_file")
    history.record_tool_call("finish")
    history.record_tool_call("run_tests")

    assert history.tools_called == ["read_file", "finish", "run_tests"]


def test_memory_finds_file_read_in_recent_context() -> None:
    history = Memory()
    record_read(history, 1, "src/example.py", "contents")

    assert history.has_current_file_read("src/example.py", WORKSPACE) is True


def test_memory_retains_file_read_outside_recent_actions() -> None:
    history = Memory()
    record_read(history, 1, "src/example.py", "contents")
    for iteration in range(2, MAX_HISTORY_ENTRIES + 2):
        history.append(iteration, ToolCall("list", "list_files", "."), "files")

    assert history.has_current_file_read("src/example.py", WORKSPACE) is True
    assert "Observation: contents" in history.to_text()


def test_skipped_reads_do_not_displace_retained_contents() -> None:
    history = Memory()
    read_call = ToolCall("inspect", "read_file", "src/example.py")
    record_read(history, 1, "src/example.py", "contents")

    for iteration in range(2, MAX_HISTORY_ENTRIES + 2):
        history.append(iteration, read_call, "Already available in recent context.", skipped=True)

    assert history.has_current_file_read("src/example.py", WORKSPACE) is True
    assert history.to_text().count("Observation: contents") == 1


def test_successful_edit_invalidates_recent_file_read() -> None:
    history = Memory()
    record_read(history, 1, "src/example.py", "old")
    history.append(
        2,
        ToolCall(
            "edit",
            "edit_file",
            "path: src/example.py\nold:\n```\nold\n```\nnew:\n```\nnew\n```",
        ),
        "Edit applied successfully to src/example.py.",
        file_path=WORKSPACE / "src/example.py",
    )

    assert history.has_current_file_read("src/example.py", WORKSPACE) is False
    assert "Observation: old" not in history.to_text()


def test_retained_reads_are_bounded_and_not_duplicated() -> None:
    history = Memory()
    for iteration in range(1, MAX_RETAINED_READS + 2):
        record_read(history, iteration, f"file_{iteration}.py", f"contents_{iteration}")
    for iteration in range(10, 10 + MAX_HISTORY_ENTRIES):
        history.append(iteration, ToolCall("list", "list_files", "."), "files")
    text = history.to_text()
    assert "Observation: contents_1" not in text
    for iteration in range(2, MAX_RETAINED_READS + 2):
        assert text.count(f"Observation: contents_{iteration}") == 1
    assert history.has_current_file_read("file_1.py", WORKSPACE) is False
    assert history.has_current_file_read("file_2.py", WORKSPACE) is True


def test_different_ranges_are_not_skipped() -> None:
    history = Memory()
    record_read(history, 1, "src/example.py | 1:10", "first page")
    assert history.has_current_file_read("./src/example.py | 1:10", WORKSPACE) is True
    assert history.has_current_file_read("src/example.py | 11:20", WORKSPACE) is False


def test_retained_read_can_be_refreshed_after_edit() -> None:
    history = Memory()
    record_read(history, 1, "src/example.py", "old contents")
    history.append(
        2, ToolCall("edit", "edit_file", "path: src/example.py"), "Edit applied successfully to src/example.py.",
        file_path=WORKSPACE / "src/example.py",
    )
    record_read(history, 3, "src/example.py", "new contents")
    assert history.has_current_file_read("src/example.py", WORKSPACE) is True
    text = history.to_text()
    assert "Observation: old contents" not in text
    assert text.count("Observation: new contents") == 1
