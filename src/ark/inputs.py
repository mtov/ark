from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cli_output import print_model_summary, print_user_prompt
from .memory import Memory
from .models import Model
from .traces import clear_trace, trace_request, trace_workspace_event

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_WORKSPACE_PATH = PROJECT_ROOT / "ark-workspace"
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.json"
SYSTEM_PROMPT_PATH = CONFIG_DIR / "system_prompt.txt"
WORKSPACE_INSTRUCTIONS_FILE_NAME = "AGENTS.md"
USER_PROMPT_FILE_NAME = "prompt.txt"
IGNORED_WORKSPACE_NAMES = ("evaluation",)
TEST_TIMEOUTS = {"pytest": 30, "maven": 300}
DEFAULT_MAX_ITERATIONS = 20


@dataclass
class Context:
    model: Model
    user_prompt: str
    workspace_path: Path
    test_runner: str = "pytest"
    test_timeout_seconds: int = TEST_TIMEOUTS["pytest"]
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    memory: Memory = field(default_factory=Memory)
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
    parser.add_argument("--test-runner", choices=TEST_TIMEOUTS, help="Test runner (default: config or pytest)")
    parser.add_argument("--test-timeout", type=_positive_int, help="Timeout for each test run, in seconds")
    parser.add_argument("--max-iterations", type=_positive_int, help="Maximum agent steps (default: config or 20)")
    return parser.parse_args()


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def load_config() -> dict[str, Any]:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            raw_config = json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file contains invalid JSON: {CONFIG_PATH}") from exc
    except OSError as exc:
        raise OSError(f"Could not read config file: {CONFIG_PATH}") from exc

    if not isinstance(raw_config, dict):
        raise ValueError("Config file must contain a JSON object.")
    return raw_config


def resolve_test_options(
    config: dict[str, Any], test_runner: str | None = None, test_timeout: int | None = None
) -> tuple[str, int]:
    runner = test_runner if test_runner is not None else config.get("test_runner", "pytest")
    if not isinstance(runner, str) or runner not in TEST_TIMEOUTS:
        raise ValueError("test_runner must be 'pytest' or 'maven'.")
    timeout = (
        test_timeout
        if test_timeout is not None
        else config.get("test_timeout_seconds", TEST_TIMEOUTS[runner])
    )
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("Test timeout must be a positive integer in seconds.")
    return runner, timeout


def load_model(system_prompt: str, config: dict[str, Any]) -> Model:
    return Model(
        name=config["openai_model"],
        system_prompt=system_prompt,
        timeout_seconds=config["timeout_seconds"],
        base_url=config.get("openai_base_url"),
        api_key_env=config.get("openai_api_key_env"),
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


def create_workspace_snapshot(context: Context) -> None:
    if context.snapshot_path is not None:
        return

    snapshot_parent = Path(tempfile.mkdtemp(prefix="ark-workspace-snapshot-"))
    snapshot_path = snapshot_parent / "workspace"
    shutil.copytree(context.workspace_path, snapshot_path)
    context.snapshot_path = snapshot_path


def _discard_workspace_snapshot(context: Context) -> None:
    snapshot_path = context.snapshot_path
    if snapshot_path is None:
        return

    shutil.rmtree(snapshot_path.parent)
    context.snapshot_path = None


def commit_workspace_changes(context: Context) -> None:
    if context.snapshot_path is None:
        return

    _discard_workspace_snapshot(context)
    trace_workspace_event("committed")


def rollback_workspace_changes(context: Context, reason: str | None = None) -> None:
    if context.snapshot_path is None:
        return

    shutil.rmtree(context.workspace_path)
    shutil.copytree(context.snapshot_path, context.workspace_path)
    _discard_workspace_snapshot(context)
    trace_workspace_event("rolled_back", reason)


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


def prepare_run(
    workspace_path_arg: str,
    *,
    test_runner: str | None = None,
    test_timeout: int | None = None,
    max_iterations: int | None = None,
) -> Context:
    clear_trace()
    config = load_config()
    runner, timeout = resolve_test_options(config, test_runner, test_timeout)
    limit = max_iterations if max_iterations is not None else config.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    if type(limit) is not int or limit <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    source_workspace_path = resolve_workspace_path(workspace_path_arg)
    workspace_path = prepare_runtime_workspace(source_workspace_path)
    system_prompt = build_system_prompt(load_system_prompt(), workspace_path)
    model = load_model(system_prompt, config)
    print_model_summary(model)
    workspace_instructions = load_workspace_instructions(workspace_path)
    if workspace_instructions is not None:
        print("Loading AGENTS.md")
    user_prompt = build_user_prompt(load_user_prompt(workspace_path), workspace_instructions)
    print_user_prompt(user_prompt)
    trace_request(user_prompt)
    return Context(
        model=model,
        user_prompt=user_prompt,
        workspace_path=workspace_path,
        test_runner=runner,
        test_timeout_seconds=timeout,
        max_iterations=limit,
    )
