from __future__ import annotations

from dataclasses import dataclass

from .inputs import AgentConfig
from .protocol import ToolRequest
from .traces import trace_finish_event
from .tools import run_tests_with_status

INVALID_FINISH_MESSAGE = "Finish action must have an empty Action Input."


@dataclass
class ApplyFinishResult:
    status: str
    test_output: str | None = None


def apply_finish(
    config: AgentConfig,
    tool_request: ToolRequest,
) -> ApplyFinishResult:
    if tool_request.args.strip():
        trace_finish_event(
            "failed",
            "finish_validation",
            INVALID_FINISH_MESSAGE,
        )
        return ApplyFinishResult(status="invalid_finish")

    tests_succeeded, test_output = run_tests_with_status(config.workspace_path)
    if not tests_succeeded:
        trace_finish_event("failed", "post_apply_tests", test_output)
        return ApplyFinishResult(
            status="post_apply_tests_failed",
            test_output=test_output,
        )

    trace_finish_event("completed", "finish")
    return ApplyFinishResult(status="completed")
