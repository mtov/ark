from __future__ import annotations

import json
from pathlib import Path

from ark import inputs


def test_prepare_runtime_workspace_recreates_fixed_directory(tmp_path: Path, monkeypatch) -> None:
    source_workspace = tmp_path / "source"
    source_workspace.mkdir()
    (source_workspace / "prompt.txt").write_text("first prompt", encoding="utf-8")

    runtime_workspace = tmp_path / "ark-workspace"
    monkeypatch.setattr(inputs, "RUNTIME_WORKSPACE_PATH", runtime_workspace)

    prepared_workspace = inputs.prepare_runtime_workspace(source_workspace)

    assert prepared_workspace == runtime_workspace
    assert (runtime_workspace / "prompt.txt").read_text(encoding="utf-8") == "first prompt"

    (runtime_workspace / "stale.txt").write_text("stale", encoding="utf-8")
    (source_workspace / "prompt.txt").write_text("second prompt", encoding="utf-8")

    prepared_workspace = inputs.prepare_runtime_workspace(source_workspace)

    assert prepared_workspace == runtime_workspace
    assert not (runtime_workspace / "stale.txt").exists()
    assert (runtime_workspace / "prompt.txt").read_text(encoding="utf-8") == "second prompt"


def test_build_user_prompt_appends_workspace_instructions() -> None:
    result = inputs.build_user_prompt(
        "Fix the bug.",
        "Always read the failing test before editing.",
    )

    assert result == (
        "Fix the bug.\n\n"
        "Workspace instructions:\n"
        "Always read the failing test before editing."
    )


def test_prepare_run_keeps_agents_md_out_of_system_prompt(tmp_path: Path, monkeypatch) -> None:
    source_workspace = tmp_path / "source"
    source_workspace.mkdir()
    (source_workspace / "prompt.txt").write_text("Fix the bug.", encoding="utf-8")
    (source_workspace / "AGENTS.md").write_text(
        "Always read the failing test before editing.",
        encoding="utf-8",
    )

    runtime_workspace = tmp_path / "ark-workspace"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "openai-compatible",
                "openai_base_url": None,
                "openai_model": "gpt-5.4-mini",
                "timeout_seconds": 30,
                "openai_api_key_env": "OPENAI_API_KEY",
            }
        ),
        encoding="utf-8",
    )
    system_prompt_path = config_dir / "system_prompt.txt"
    system_prompt_path.write_text("Base system prompt.", encoding="utf-8")

    traced_requests: list[str] = []

    monkeypatch.setattr(inputs, "RUNTIME_WORKSPACE_PATH", runtime_workspace)
    monkeypatch.setattr(inputs, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(inputs, "CONFIG_PATH", config_path)
    monkeypatch.setattr(inputs, "SYSTEM_PROMPT_PATH", system_prompt_path)
    monkeypatch.setattr(inputs, "trace_request", lambda prompt: traced_requests.append(prompt))

    config = inputs.prepare_run(str(source_workspace))

    assert config.system_prompt == (
        "Base system prompt.\n\n"
        "Workspace root:\n"
        f"{runtime_workspace}"
    )
    assert config.user_prompt == (
        "Fix the bug.\n\n"
        "Workspace instructions:\n"
        "Always read the failing test before editing."
    )
    assert traced_requests == [config.user_prompt]
