from __future__ import annotations

from pathlib import Path

from ark import tools
from ark.inputs import AgentConfig
from ark.models import ModelConfig


def build_context(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        model_config=ModelConfig("openai-compatible", 30, None, "model", "OPENAI_API_KEY"),
        system_prompt="system",
        user_prompt="prompt",
        source_workspace_path=tmp_path,
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

    content = tools.read_file("example.txt", tmp_path)

    assert content == "hello"


def test_run_tests_records_test_result(monkeypatch, tmp_path: Path) -> None:
    events: list[tuple[str, str | None]] = []
    monkeypatch.setattr(tools, "run_tests_with_status", lambda _path: (False, "1 failed"))
    monkeypatch.setattr(
        tools,
        "trace_test_event",
        lambda status, detail=None: events.append((status, detail)),
    )

    output = tools.run_tests(tmp_path)

    assert output == "1 failed"
    assert events == [("failed", "1 failed")]


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
