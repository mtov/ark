from __future__ import annotations

from dataclasses import dataclass

from .inputs import AgentConfig
from .patches import (
    apply_patch,
    build_patch_from_full_file_response,
    looks_like_full_file_response,
    save_patch,
    validate_and_repair_patch,
)
from .protocol import ToolRequest, looks_like_patch
from .traces import trace_finish_event
from .tools import run_tests_with_status

INVALID_FINISH_PATCH_MESSAGE = (
    "Finish output must be a unified diff patch or FILE: sections with complete file contents."
)


@dataclass
class FinishResult:
    status: str
    request: ToolRequest
    test_output: str | None = None
    tools_called: list[str] | None = None


def _trace_and_raise(stage: str, exc: Exception) -> None:
    trace_finish_event("failed", stage, str(exc))
    raise exc


def apply_finish(
    config: AgentConfig,
    tool_request: ToolRequest,
) -> FinishResult:
    workspace_path = config.workspace_path
    finish_text = tool_request.args

    if looks_like_patch(finish_text):
        patch_text = finish_text
    elif looks_like_full_file_response(finish_text):
        patch_text = build_patch_from_full_file_response(workspace_path, finish_text)
    else:
        trace_finish_event(
            "failed",
            "finish_validation",
            INVALID_FINISH_PATCH_MESSAGE,
        )
        raise ValueError(INVALID_FINISH_PATCH_MESSAGE)

    try:
        patch_text = validate_and_repair_patch(workspace_path, patch_text)
        tool_request.args = patch_text
    except Exception as exc:  # noqa: BLE001
        _trace_and_raise("patch_validation", exc)

    try:
        save_patch(workspace_path, patch_text)
    except Exception as exc:  # noqa: BLE001
        _trace_and_raise("save_patch", exc)

    try:
        apply_patch(workspace_path)
    except Exception as exc:  # noqa: BLE001
        _trace_and_raise("apply_patch", exc)

    tests_succeeded, test_output = run_tests_with_status(workspace_path)
    if not tests_succeeded:
        trace_finish_event("failed", "post_apply_tests", test_output)
        return FinishResult(
            status="post_apply_tests_failed",
            request=tool_request,
            test_output=test_output,
            tools_called=["apply_patch", "run_tests"],
        )

    trace_finish_event("completed", "finish")
    return FinishResult(
        status="applied",
        request=tool_request,
        tools_called=["apply_patch", "run_tests"],
    )
