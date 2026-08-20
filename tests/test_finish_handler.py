from __future__ import annotations

from pathlib import Path

import pytest

from ark.finish_handler import INVALID_FINISH_PATCH_MESSAGE, apply_finish
from ark.inputs import AgentConfig
from ark.models import ModelConfig
from ark.protocol import ToolRequest


def build_context(tmp_path: Path, user_prompt: str) -> AgentConfig:
    return AgentConfig(
        model_config=ModelConfig(
            model="openai-compatible",
            timeout_seconds=30,
            openai_base_url=None,
            openai_model="gpt-5.4-mini",
            openai_api_key_env="OPENAI_API_KEY",
        ),
        system_prompt="system",
        user_prompt=user_prompt,
        source_workspace_path=tmp_path,
        workspace_path=tmp_path,
    )

def test_handle_finish_runs_post_apply_tests_for_bug_fix(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path, "Fix the bug in slugify and make tests pass.")
    tool_request = ToolRequest(
        thought="done",
        name="finish",
        args="--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n",
    )
    calls: list[str] = []

    monkeypatch.setattr("ark.finish_handler.validate_and_repair_patch", lambda *_args: tool_request.args)
    monkeypatch.setattr("ark.finish_handler.save_patch", lambda *_args: calls.append("save_patch"))
    monkeypatch.setattr("ark.finish_handler.apply_patch", lambda *_args: calls.append("apply_patch"))
    monkeypatch.setattr("ark.finish_handler.run_tests_with_status", lambda *_args: (True, "1 passed"))
    monkeypatch.setattr("ark.finish_handler.trace_finish_event", lambda *_args: None)

    result = apply_finish(context, tool_request)

    assert result.status == "applied"
    assert result.request is tool_request
    assert result.tools_called == ["apply_patch", "run_tests"]
    assert calls == ["save_patch", "apply_patch"]


def test_handle_finish_runs_post_apply_tests_for_feature_task(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path, "Implement the BUY2GET50 feature in checkout.")
    tool_request = ToolRequest(
        thought="done",
        name="finish",
        args="--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n",
    )
    run_tests_called = False

    def fake_run_tests(*_args):
        nonlocal run_tests_called
        run_tests_called = True
        return True, "3 passed"

    monkeypatch.setattr("ark.finish_handler.validate_and_repair_patch", lambda *_args: tool_request.args)
    monkeypatch.setattr("ark.finish_handler.save_patch", lambda *_args: None)
    monkeypatch.setattr("ark.finish_handler.apply_patch", lambda *_args: None)
    monkeypatch.setattr("ark.finish_handler.run_tests_with_status", fake_run_tests)
    monkeypatch.setattr("ark.finish_handler.trace_finish_event", lambda *_args: None)

    result = apply_finish(context, tool_request)

    assert result.status == "applied"
    assert result.request is tool_request
    assert result.tools_called == ["apply_patch", "run_tests"]
    assert run_tests_called is True


def test_handle_finish_runs_post_apply_tests_for_bugfix_without_keywords(monkeypatch, tmp_path: Path) -> None:
    context = build_context(
        tmp_path,
        "Users reported that date-based exports are missing the last day of the selected range.",
    )
    tool_request = ToolRequest(
        thought="done",
        name="finish",
        args="--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n",
    )
    run_tests_called = False

    def fake_run_tests(*_args):
        nonlocal run_tests_called
        run_tests_called = True
        return True, "2 passed"

    monkeypatch.setattr("ark.finish_handler.validate_and_repair_patch", lambda *_args: tool_request.args)
    monkeypatch.setattr("ark.finish_handler.save_patch", lambda *_args: None)
    monkeypatch.setattr("ark.finish_handler.apply_patch", lambda *_args: None)
    monkeypatch.setattr("ark.finish_handler.run_tests_with_status", fake_run_tests)
    monkeypatch.setattr("ark.finish_handler.trace_finish_event", lambda *_args: None)

    result = apply_finish(context, tool_request)

    assert result.status == "applied"
    assert result.request is tool_request
    assert result.tools_called == ["apply_patch", "run_tests"]
    assert run_tests_called is True


def test_handle_finish_raises_when_post_apply_tests_fail(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path, "Fix the bug in slugify and make tests pass.")
    tool_request = ToolRequest(
        thought="done",
        name="finish",
        args="--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n",
    )

    monkeypatch.setattr("ark.finish_handler.validate_and_repair_patch", lambda *_args: tool_request.args)
    monkeypatch.setattr("ark.finish_handler.save_patch", lambda *_args: None)
    monkeypatch.setattr("ark.finish_handler.apply_patch", lambda *_args: None)
    monkeypatch.setattr("ark.finish_handler.run_tests_with_status", lambda *_args: (False, "1 failed"))
    finish_events: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr("ark.finish_handler.trace_finish_event", lambda *args: finish_events.append(args))

    result = apply_finish(context, tool_request)

    assert result.status == "post_apply_tests_failed"
    assert result.request is tool_request
    assert result.test_output == "1 failed"
    assert result.tools_called == ["apply_patch", "run_tests"]
    assert finish_events == [("failed", "post_apply_tests", "1 failed")]


def test_handle_finish_runs_post_apply_tests_for_readme_patch(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path, "Update the README wording.")
    tool_request = ToolRequest(
        thought="done",
        name="finish",
        args="--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n",
    )
    run_tests_called = False

    def fake_run_tests(*_args):
        nonlocal run_tests_called
        run_tests_called = True
        return True, "1 passed"

    monkeypatch.setattr("ark.finish_handler.validate_and_repair_patch", lambda *_args: tool_request.args)
    monkeypatch.setattr("ark.finish_handler.save_patch", lambda *_args: None)
    monkeypatch.setattr("ark.finish_handler.apply_patch", lambda *_args: None)
    monkeypatch.setattr("ark.finish_handler.run_tests_with_status", fake_run_tests)
    monkeypatch.setattr("ark.finish_handler.trace_finish_event", lambda *_args: None)

    result = apply_finish(context, tool_request)

    assert result.status == "applied"
    assert result.request is tool_request
    assert result.tools_called == ["apply_patch", "run_tests"]
    assert run_tests_called is True


def test_handle_finish_repairs_patch_when_request_args_looks_like_patch(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path, "Please fix slugify in src/text_utils.py.")
    original_patch = (
        "--- a/src/text_utils.py\n"
        "+++ b/src/text_utils.py\n"
        "@@ -1,6 +1,11 @@\n"
        " import re\n"
        "+import unicodedata\n"
        " \n"
        " def slugify(title: str) -> str:\n"
        "-    value = title.lower().strip()\n"
        "+    value = unicodedata.normalize(\"NFKD\", title)\n"
        "+    value = value.encode(\"ascii\", \"ignore\").decode(\"ascii\")\n"
        "+    value = value.lower().strip()\n"
        "     value = re.sub(r\"[^a-z0-9]+\", \"-\", value)\n"
        "-    return value.strip(\"-\")\n"
        "+    value = value.strip(\"-\")\n"
        "+    return value or \"item\"\n"
    )
    repaired_patch = original_patch.replace("@@ -1,6 +1,11 @@", "@@ -1,6 +1,10 @@")
    tool_request = ToolRequest(
        thought="done",
        name="finish",
        args=original_patch,
    )
    saved_patches: list[str] = []

    monkeypatch.setattr("ark.finish_handler.validate_and_repair_patch", lambda *_args: repaired_patch)
    monkeypatch.setattr("ark.finish_handler.save_patch", lambda _path, patch: saved_patches.append(patch))
    monkeypatch.setattr("ark.finish_handler.apply_patch", lambda *_args: None)
    monkeypatch.setattr("ark.finish_handler.run_tests_with_status", lambda *_args: (True, "1 passed"))
    monkeypatch.setattr("ark.finish_handler.trace_finish_event", lambda *_args: None)

    result = apply_finish(context, tool_request)

    assert result.status == "applied"
    assert result.request is tool_request
    assert result.tools_called == ["apply_patch", "run_tests"]
    assert tool_request.args == repaired_patch
    assert saved_patches == [repaired_patch]


def test_handle_finish_builds_patch_from_full_file_response(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path, "Fix the bug in src/text_utils.py.")
    source_path = tmp_path / "src"
    source_path.mkdir()
    (source_path / "text_utils.py").write_text("def slugify(value):\n    return value\n", encoding="utf-8")
    tool_request = ToolRequest(
        thought="done",
        name="finish",
        args=(
            "FILE: src/text_utils.py\n"
            "```python\n"
            "def slugify(value):\n"
            "    return value.strip().lower()\n"
            "```\n"
        ),
    )
    saved_patches: list[str] = []

    monkeypatch.setattr("ark.finish_handler.save_patch", lambda _path, patch: saved_patches.append(patch))
    monkeypatch.setattr("ark.finish_handler.apply_patch", lambda *_args: None)
    monkeypatch.setattr("ark.finish_handler.run_tests_with_status", lambda *_args: (True, "1 passed"))
    monkeypatch.setattr("ark.finish_handler.trace_finish_event", lambda *_args: None)

    result = apply_finish(context, tool_request)

    assert result.status == "applied"
    assert result.tools_called == ["apply_patch", "run_tests"]
    assert tool_request.args.startswith("--- a/src/text_utils.py\n+++ b/src/text_utils.py\n")
    assert saved_patches == [tool_request.args]


def test_handle_finish_rejects_mixed_output_when_tests_failed(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path, "Implement the BUY2GET50 feature in checkout.")
    tool_request = ToolRequest(
        thought="done",
        name="finish",
        args="--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n",
    )
    finish_events: list[tuple[str, str, str | None]] = []

    monkeypatch.setattr("ark.finish_handler.validate_and_repair_patch", lambda *_args: tool_request.args)
    monkeypatch.setattr("ark.finish_handler.save_patch", lambda *_args: None)
    monkeypatch.setattr("ark.finish_handler.apply_patch", lambda *_args: None)
    monkeypatch.setattr(
        "ark.finish_handler.run_tests_with_status",
        lambda *_args: (False, "1 failed, 5 passed"),
    )
    monkeypatch.setattr("ark.finish_handler.trace_finish_event", lambda *args: finish_events.append(args))

    result = apply_finish(context, tool_request)

    assert result.status == "post_apply_tests_failed"
    assert result.test_output == "1 failed, 5 passed"
    assert result.tools_called == ["apply_patch", "run_tests"]
    assert finish_events == [("failed", "post_apply_tests", "1 failed, 5 passed")]


def test_handle_finish_traces_patch_validation_failures(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path, "Refactor the checkout flow.")
    tool_request = ToolRequest(
        thought="done",
        name="finish",
        args="--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n",
    )
    finish_events: list[tuple[str, str, str | None]] = []

    def raise_patch_error(*_args):
        raise ValueError("corrupt patch")

    monkeypatch.setattr("ark.finish_handler.validate_and_repair_patch", raise_patch_error)
    monkeypatch.setattr("ark.finish_handler.trace_finish_event", lambda *args: finish_events.append(args))

    with pytest.raises(ValueError, match="corrupt patch"):
        apply_finish(context, tool_request)

    assert finish_events == [("failed", "patch_validation", "corrupt patch")]


def test_handle_finish_rejects_non_patch_finish_output(monkeypatch, tmp_path: Path) -> None:
    context = build_context(tmp_path, "Any Ark task.")
    tool_request = ToolRequest(
        thought="done",
        name="finish",
        args="Task completed successfully.",
    )
    finish_events: list[tuple[str, str, str | None]] = []

    monkeypatch.setattr("ark.finish_handler.trace_finish_event", lambda *args: finish_events.append(args))

    with pytest.raises(ValueError, match="Finish output must be a unified diff patch or FILE: sections with complete file contents."):
        apply_finish(context, tool_request)

    assert finish_events == [
        ("failed", "finish_validation", INVALID_FINISH_PATCH_MESSAGE),
    ]
