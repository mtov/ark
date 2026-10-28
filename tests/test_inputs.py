from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def test_prepare_runtime_workspace_ignores_evaluation_directory(tmp_path: Path, monkeypatch) -> None:
    source_workspace = tmp_path / "source"
    source_workspace.mkdir()
    (source_workspace / "prompt.txt").write_text("prompt", encoding="utf-8")
    evaluation_dir = source_workspace / "evaluation"
    evaluation_dir.mkdir()
    (evaluation_dir / "results.json").write_text('{"score": 1.0}', encoding="utf-8")

    runtime_workspace = tmp_path / "ark-workspace"
    monkeypatch.setattr(inputs, "RUNTIME_WORKSPACE_PATH", runtime_workspace)

    prepared_workspace = inputs.prepare_runtime_workspace(source_workspace)

    assert prepared_workspace == runtime_workspace
    assert (runtime_workspace / "prompt.txt").exists()
    assert not (runtime_workspace / "evaluation").exists()


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


def test_build_system_prompt_appends_workspace_root(tmp_path: Path) -> None:
    result = inputs.build_system_prompt("Base system prompt.", tmp_path)

    assert result == f"Base system prompt.\n\nWorkspace root:\n{tmp_path}"


def test_load_model_supports_openai(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "openai_base_url": "http://localhost:8000/v1",
                "openai_model": "local-model",
                "openai_api_key_env": "OPENAI_API_KEY",
                "timeout_seconds": 45,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(inputs, "CONFIG_PATH", config_path)

    model = inputs.load_model("system prompt", inputs.load_config())

    assert model.name == "local-model"
    assert model.system_prompt == "system prompt"
    assert model.base_url == "http://localhost:8000/v1"
    assert model.api_key_env == "OPENAI_API_KEY"
    assert model.timeout_seconds == 45


def test_load_config_reports_invalid_json(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr(inputs, "CONFIG_PATH", config_path)

    with pytest.raises(ValueError, match="contains invalid JSON"):
        inputs.load_config()


@pytest.mark.parametrize(
    ("config_options", "overrides", "runner", "timeout", "limit"),
    [
        ({}, {}, "pytest", 30, 20),
        ({"test_runner": "maven"}, {}, "maven", 300, 20),
        ({"test_runner": "maven", "test_timeout_seconds": 180}, {}, "maven", 180, 20),
        (
            {"test_runner": "pytest", "test_timeout_seconds": 180},
            {"test_runner": "maven", "test_timeout": 600},
            "maven", 600, 20,
        ),
        ({"test_runner": "maven"}, {"test_runner": "pytest"}, "pytest", 30, 20),
        ({"test_runner": "maven"}, {"test_timeout": 600}, "maven", 600, 20),
        ({"test_timeout_seconds": 180}, {"test_runner": "maven"}, "maven", 180, 20),
        ({"max_iterations": 30}, {}, "pytest", 30, 30),
        ({"max_iterations": 30}, {"max_iterations": 40}, "pytest", 30, 40),
    ],
)
def test_prepare_run_keeps_agents_md_out_of_system_prompt(
    tmp_path: Path, monkeypatch, config_options, overrides, runner, timeout, limit
) -> None:
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
                "openai_base_url": None,
                "openai_model": "gpt-6-luna",
                "timeout_seconds": 30,
                "openai_api_key_env": "OPENAI_API_KEY",
                **config_options,
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

    context = inputs.prepare_run(str(source_workspace), **overrides)

    assert context.model.system_prompt == (
        "Base system prompt.\n\n"
        "Workspace root:\n"
        f"{runtime_workspace}"
    )
    assert context.user_prompt == (
        "Fix the bug.\n\n"
        "Workspace instructions:\n"
        "Always read the failing test before editing."
    )
    assert traced_requests == [context.user_prompt]
    assert context.test_runner == runner
    assert context.test_timeout_seconds == timeout
    assert context.max_iterations == limit
    assert context.model.timeout_seconds == 30


@pytest.mark.parametrize("runner", ["gradle", None, [], 1])
def test_resolve_test_options_rejects_invalid_runner(runner) -> None:
    with pytest.raises(ValueError, match="test_runner must be"):
        inputs.resolve_test_options({"test_runner": runner})


@pytest.mark.parametrize("timeout", [0, -1, True, "30", 0.5, None])
def test_resolve_test_options_rejects_invalid_timeout(timeout) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        inputs.resolve_test_options({"test_timeout_seconds": timeout})


@pytest.mark.parametrize("config", [[], None, "config"])
def test_load_config_requires_json_object(tmp_path: Path, monkeypatch, config) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(inputs, "CONFIG_PATH", config_path)
    with pytest.raises(ValueError, match="JSON object"):
        inputs.load_config()


@pytest.mark.parametrize(
    ("options", "runner", "timeout"),
    [([], None, None), (["--test-runner", "maven", "--test-timeout", "600"], "maven", 600)],
)
def test_parse_args_accepts_test_options(monkeypatch, options, runner, timeout) -> None:
    monkeypatch.setattr(sys, "argv", ["ark", "workspace", *options])
    args = inputs.parse_args()
    assert args.workspace_path == "workspace"
    assert args.test_runner == runner
    assert args.test_timeout == timeout


@pytest.mark.parametrize(
    "options",
    [
        ["--test-runner", "gradle"],
        ["--test-timeout", "0"],
        ["--test-timeout", "-1"],
        ["--test-timeout", "1.5"],
        ["--test-timeout", "abc"],
        ["--max-iterations", "0"],
        ["--max-iterations", "-1"],
        ["--max-iterations", "1.5"],
    ],
)
def test_parse_args_rejects_invalid_test_options(monkeypatch, options) -> None:
    monkeypatch.setattr(sys, "argv", ["ark", "workspace", *options])
    with pytest.raises(SystemExit) as error:
        inputs.parse_args()
    assert error.value.code == 2


def test_prepare_run_validates_options_before_copying_workspace(tmp_path: Path, monkeypatch) -> None:
    copies = []
    monkeypatch.setattr(inputs, "clear_trace", lambda: None)
    monkeypatch.setattr(inputs, "load_config", lambda: {"test_runner": "gradle"})
    monkeypatch.setattr(inputs, "prepare_runtime_workspace", lambda path: copies.append(path))
    with pytest.raises(ValueError, match="test_runner must be"):
        inputs.prepare_run(str(tmp_path))
    assert copies == []


def test_parse_args_accepts_iteration_limit(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ark", "workspace", "--max-iterations", "40"])
    assert inputs.parse_args().max_iterations == 40


@pytest.mark.parametrize("limit", [0, -1, True, 0.5, "20", None])
def test_prepare_run_rejects_invalid_iteration_limit_before_copying(tmp_path: Path, monkeypatch, limit) -> None:
    copies = []
    monkeypatch.setattr(inputs, "clear_trace", lambda: None)
    monkeypatch.setattr(inputs, "load_config", lambda: {"max_iterations": limit})
    monkeypatch.setattr(inputs, "prepare_runtime_workspace", lambda path: copies.append(path))
    with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
        inputs.prepare_run(str(tmp_path))
    assert copies == []
