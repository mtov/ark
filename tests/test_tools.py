from __future__ import annotations

from pathlib import Path

from ark import tools


def test_read_file_returns_file_contents(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello", encoding="utf-8")

    content = tools.read_file("example.txt", tmp_path)

    assert content == "hello"
