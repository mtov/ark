from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .models import ModelConfig
from .traces import clear_trace, trace_request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_WORKSPACE_PATH = PROJECT_ROOT / "ark-workspace"
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.json"
SYSTEM_PROMPT_PATH = CONFIG_DIR / "system_prompt.txt"
WORKSPACE_INSTRUCTIONS_FILE_NAME = "AGENTS.md"
USER_PROMPT_FILE_NAME = "prompt.txt"
IGNORED_WORKSPACE_NAMES = ("evaluation",)


@dataclass
class AgentConfig:
    model_config: ModelConfig
    system_prompt: str
    user_prompt: str
    source_workspace_path: Path
    workspace_path: Path
    snapshot_path: Path | None = None


def _read_text_file(
    path: Path,
    *,
    missing_message: str,
    read_message: str,
    empty_message: str | None = None,
) -> str:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(missing_message) from exc
    except OSError as exc:
        raise OSError(read_message) from exc

    if not content and empty_message is not None:
        raise ValueError(empty_message)

    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ark")
    parser.add_argument("workspace_path", help="Path to the test workspace directory")
    return parser.parse_args()


def load_model_config() -> ModelConfig:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            raw_config = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file contains invalid JSON: {CONFIG_PATH}") from exc
    except OSError as exc:
        raise OSError(f"Could not read config file: {CONFIG_PATH}") from exc

    selected_model = raw_config.get("model")
    if selected_model is None:
        raise KeyError("Config file must define 'model'.")

    return ModelConfig(
        model=selected_model,
        timeout_seconds=raw_config["timeout_seconds"],
        openai_base_url=raw_config.get("openai_base_url"),
        openai_model=raw_config.get("openai_model"),
        openai_api_key_env=raw_config.get("openai_api_key_env"),
        ollama_base_url=raw_config.get("ollama_base_url"),
        ollama_model=raw_config.get("ollama_model"),
    )


def load_system_prompt() -> str:
    return _read_text_file(
        SYSTEM_PROMPT_PATH,
        missing_message=f"System prompt file not found: {SYSTEM_PROMPT_PATH}",
        read_message=f"Could not read system prompt file: {SYSTEM_PROMPT_PATH}",
        empty_message="System prompt file is empty.",
    )


def load_workspace_instructions(workspace_path: Path) -> str | None:
    instructions_path = workspace_path / WORKSPACE_INSTRUCTIONS_FILE_NAME
    if not instructions_path.exists():
        return None

    return _read_text_file(
        instructions_path,
        missing_message=f"Workspace instructions file not found: {instructions_path}",
        read_message=f"Could not read workspace instructions file: {instructions_path}",
    ) or None


def resolve_workspace_path(raw_path: str) -> Path:
    workspace_path = Path(raw_path).expanduser().resolve()
    if not workspace_path.exists():
        raise FileNotFoundError(f"Workspace path not found: {workspace_path}")
    if not workspace_path.is_dir():
        raise NotADirectoryError(f"Workspace path is not a directory: {workspace_path}")
    return workspace_path


def prepare_runtime_workspace(source_workspace_path: Path) -> Path:
    if RUNTIME_WORKSPACE_PATH.exists():
        shutil.rmtree(RUNTIME_WORKSPACE_PATH)

    shutil.copytree(
        source_workspace_path,
        RUNTIME_WORKSPACE_PATH,
        ignore=shutil.ignore_patterns(*IGNORED_WORKSPACE_NAMES),
    )
    return RUNTIME_WORKSPACE_PATH


def create_workspace_snapshot(config: AgentConfig) -> None:
    if config.snapshot_path is not None:
        return

    snapshot_parent = Path(tempfile.mkdtemp(prefix="ark-workspace-snapshot-"))
    snapshot_path = snapshot_parent / "workspace"
    shutil.copytree(config.workspace_path, snapshot_path)
    config.snapshot_path = snapshot_path


def _discard_workspace_snapshot(config: AgentConfig) -> None:
    snapshot_path = config.snapshot_path
    if snapshot_path is None:
        return

    shutil.rmtree(snapshot_path.parent)
    config.snapshot_path = None


def commit_workspace_changes(config: AgentConfig) -> None:
    if config.snapshot_path is None:
        return

    _discard_workspace_snapshot(config)


def rollback_workspace_changes(config: AgentConfig) -> None:
    if config.snapshot_path is None:
        return

    shutil.rmtree(config.workspace_path)
    shutil.copytree(config.snapshot_path, config.workspace_path)
    _discard_workspace_snapshot(config)


def load_user_prompt(workspace_path: Path) -> str:
    prompt_path = workspace_path / USER_PROMPT_FILE_NAME
    return _read_text_file(
        prompt_path,
        missing_message=f"User prompt file not found: {prompt_path}",
        read_message=f"Could not read user prompt file: {prompt_path}",
        empty_message="User prompt file is empty.",
    )


def build_system_prompt(system_prompt: str, workspace_path: Path) -> str:
    return f"{system_prompt}\n\nWorkspace root:\n{workspace_path}"


def build_user_prompt(user_prompt: str, workspace_instructions: str | None = None) -> str:
    if workspace_instructions is None:
        return user_prompt

    return (
        f"{user_prompt}\n\n"
        "Workspace instructions:\n"
        f"{workspace_instructions}"
    )


def print_model_summary(model_config: ModelConfig) -> None:
    model_name = model_config.openai_model or model_config.ollama_model or model_config.model
    print(f"Using model: {model_config.model} ({model_name})")


def print_user_prompt(user_prompt: str) -> None:
    print("Prompt:")
    print(user_prompt)


def prepare_run(workspace_path_arg: str) -> AgentConfig:
    clear_trace()
    source_workspace_path = resolve_workspace_path(workspace_path_arg)
    workspace_path = prepare_runtime_workspace(source_workspace_path)
    model_config = load_model_config()
    print_model_summary(model_config)
    system_prompt = build_system_prompt(load_system_prompt(), workspace_path)
    workspace_instructions = load_workspace_instructions(workspace_path)
    if workspace_instructions is not None:
        print("Loading AGENTS.md")
    user_prompt = build_user_prompt(
        load_user_prompt(workspace_path),
        workspace_instructions,
    )
    print_user_prompt(user_prompt)
    trace_request(user_prompt)
    return AgentConfig(
        model_config=model_config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source_workspace_path=source_workspace_path,
        workspace_path=workspace_path,
    )
