from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from .cli_output import (
    print_failure_summary,
    print_success_summary,
    print_tool_call,
)
from .inputs import (
    Context,
    commit_workspace_changes,
    parse_args,
    prepare_run,
    rollback_workspace_changes,
)
from .protocol import ToolCall, parse_response, repair_response
from .tools import run_tests_with_status, run_tool
from .traces import (
    trace_action,
    trace_error,
    trace_finish_event,
    trace_run_summary,
    trace_validation_error,
)

MAX_ITERATIONS_REACHED_MESSAGE = "Agent stopped after reaching the maximum number of steps."
AGENTIC_LOOP_ERROR_MESSAGE = "agentic loop error"
MAX_ITERATIONS = 20
FINISH_SUCCESS_MESSAGE = "Changes applied and final tests passed."
FINISH_WITHOUT_EDIT_MESSAGE = "Finish requires at least one approved edit_file action."
INVALID_FINISH_MESSAGE = "Finish action must have an empty Action Input."


@dataclass
class LoopResult:
    status: Literal["success", "max_iterations_reached", "failed"]
    error: str | None = None
    error_type: str | None = None


def invoke_model(context: Context) -> ToolCall:
    model_response = context.model.call(context.user_prompt, context.memory.to_text())

    try:
        tool_call = parse_response(model_response.content)
    except ValueError as exc:
        trace_validation_error(str(exc), model_response.content)
        tool_call = repair_response(context, str(exc))

    trace_action(tool_call)
    context.memory.record_tool_call(tool_call.name)
    return tool_call


def try_finish(
    context: Context,
    iteration: int,
    tool_call: ToolCall,
) -> bool:
    print_tool_call(iteration, tool_call)

    if not context.memory.has_successful_edit():
        trace_finish_event("failed", "finish_validation", FINISH_WITHOUT_EDIT_MESSAGE)
        context.memory.append(iteration, tool_call, FINISH_WITHOUT_EDIT_MESSAGE)
        return False

    if tool_call.args.strip():
        trace_finish_event("failed", "finish_validation", INVALID_FINISH_MESSAGE)
        context.memory.append(iteration, tool_call, INVALID_FINISH_MESSAGE)
        return False

    tests_succeeded, test_output = run_tests_with_status(context.workspace_path)
    if tests_succeeded:
        trace_finish_event("completed", "finish")
    else:
        trace_finish_event("failed", "post_apply_tests", test_output)
    context.memory.record_tool_call("run_tests")
    if not tests_succeeded:
        context.memory.append(iteration, tool_call, test_output or "Tests failed without output.")
        return False

    return True


def agentic_loop(context: Context) -> LoopResult:
    try:
        for iteration in range(1, MAX_ITERATIONS + 1):
            tool_call = invoke_model(context)

            if tool_call.name == "finish":
                if not try_finish(context, iteration, tool_call):
                    continue

                commit_workspace_changes(context)
                return LoopResult(status="success")

            tool_result = run_tool(tool_call, context)
            print_tool_call(iteration, tool_call, tool_result.note)
            context.memory.append(
                iteration,
                tool_call,
                tool_result.output,
                skipped=tool_result.skipped,
            )

        rollback_workspace_changes(context, MAX_ITERATIONS_REACHED_MESSAGE)
        return LoopResult(status="max_iterations_reached")
    except Exception as exc:  # noqa: BLE001
        rollback_workspace_changes(context, AGENTIC_LOOP_ERROR_MESSAGE)
        return LoopResult(status="failed", error=str(exc), error_type=exc.__class__.__name__)


def main() -> int:
    args = parse_args()
    start_time = perf_counter()
    context: Context | None = None

    try:
        context = prepare_run(args.workspace_path)
        loop_result = agentic_loop(context)
    except Exception as exc:  # noqa: BLE001
        if context is not None:
            rollback_workspace_changes(context)
        elapsed_seconds = perf_counter() - start_time
        trace_error("prepare_run", str(exc), exc.__class__.__name__, [])
        trace_run_summary(elapsed_seconds, [])
        print_failure_summary(str(exc), elapsed_seconds)
        return 1

    elapsed_seconds = perf_counter() - start_time

    if loop_result.status != "success":
        error_message = (
            MAX_ITERATIONS_REACHED_MESSAGE
            if loop_result.status == "max_iterations_reached"
            else loop_result.error or "Unknown error."
        )
        if loop_result.status == "failed":
            trace_error(
                "agentic_loop",
                error_message,
                loop_result.error_type or "Exception",
                context.memory.tools_called,
            )
        trace_run_summary(elapsed_seconds, context.memory.tools_called)
        print_failure_summary(error_message, elapsed_seconds)
        return 1

    trace_run_summary(elapsed_seconds, context.memory.tools_called)
    print_success_summary(FINISH_SUCCESS_MESSAGE, elapsed_seconds)
    return 0
