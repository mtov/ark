from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from .cli_output import (
    print_failure_summary,
    print_tool_request,
    print_success_summary,
)
from .finish_handler import apply_finish
from .inputs import (
    AgentConfig,
    commit_workspace_changes,
    parse_args,
    prepare_run,
    rollback_workspace_changes,
)
from .memory import Memory
from .models import call_model
from .protocol import ToolRequest, parse_response, repair_response
from .test_failures import summarize_test_failure_output
from .tools import run_tool
from .traces import trace_action, trace_finish_event, trace_run_summary, trace_validation_error

MAX_ITERATIONS_REACHED_MESSAGE = "Agent stopped after reaching the maximum number of steps."
MAX_ITERATIONS = 20
FINISH_SUCCESS_MESSAGE = "Changes applied and final tests passed."
FINISH_WITHOUT_EDIT_MESSAGE = "Finish requires at least one approved edit_file action."


@dataclass
class LoopResult:
    status: str
    output: str | None = None
    error: str | None = None
    tools_called: list[str] = field(default_factory=list)

    @classmethod
    def success(
        cls,
        output: str,
        *,
        tools_called: list[str] | None = None,
    ) -> LoopResult:
        return cls(
            status="success",
            output=output,
            tools_called=tools_called or [],
        )

    @classmethod
    def max_iterations_reached(cls, *, tools_called: list[str] | None = None) -> LoopResult:
        return cls(
            status="max_iterations_reached",
            error=MAX_ITERATIONS_REACHED_MESSAGE,
            tools_called=tools_called or [],
        )


def get_next_tool_request(config: AgentConfig, memory: Memory) -> ToolRequest:
    user_message = (
        "User task:\n"
        f"{config.user_prompt}\n\n"
        "Agent history:\n"
        f"{memory.to_text()}"
    )
    model_response = call_model(config, user_message)

    try:
        tool_request = parse_response(model_response.content)
    except ValueError as exc:
        trace_validation_error(str(exc), model_response.content)
        tool_request = repair_response(config, user_message, str(exc))

    trace_action(tool_request)
    return tool_request


def handle_finish(
    config: AgentConfig,
    memory: Memory,
    iteration: int,
    tool_request: ToolRequest,
    tools_called: list[str],
) -> str | None:
    if not memory.has_successful_edit():
        print_tool_request(iteration, tool_request)
        trace_finish_event("failed", "finish_validation", FINISH_WITHOUT_EDIT_MESSAGE)
        memory.append(iteration, tool_request, FINISH_WITHOUT_EDIT_MESSAGE)
        return None

    finish_result = apply_finish(config, tool_request)
    print_tool_request(iteration, tool_request)

    if finish_result.status == "invalid_finish":
        memory.append(
            iteration,
            tool_request,
            "Finish action must have an empty Action Input.",
        )
        return None

    tools_called.append("run_tests")
    if finish_result.status == "post_apply_tests_failed":
        memory.append(
            iteration,
            tool_request,
            summarize_test_failure_output(finish_result.test_output or ""),
        )
        return None

    return FINISH_SUCCESS_MESSAGE


def agentic_loop(config: AgentConfig) -> LoopResult:
    try:
        memory = Memory()
        tools_called: list[str] = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            tool_request = get_next_tool_request(config, memory)
            tools_called.append(tool_request.name)

            if tool_request.name == "finish":
                finish_output = handle_finish(
                    config,
                    memory,
                    iteration,
                    tool_request,
                    tools_called,
                )
                if finish_output is None:
                    continue

                commit_workspace_changes(config)
                return LoopResult.success(
                    finish_output,
                    tools_called=tools_called,
                )

            previous_request = memory.last_tool_request()
            tool_result = run_tool(tool_request, config, previous_request)
            print_tool_request(iteration, tool_request, tool_result.note)
            memory.append(iteration, tool_request, tool_result.output)

        rollback_workspace_changes(config)
        return LoopResult.max_iterations_reached(tools_called=tools_called)
    except Exception:
        rollback_workspace_changes(config)
        raise


def main() -> int:
    args = parse_args()
    start_time = perf_counter()
    config: AgentConfig | None = None

    try:
        config = prepare_run(args.workspace_path)
        loop_result = agentic_loop(config)
    except Exception as exc:  # noqa: BLE001
        if config is not None:
            rollback_workspace_changes(config)
        elapsed_seconds = perf_counter() - start_time
        trace_run_summary(elapsed_seconds, [])
        print_failure_summary(exc, elapsed_seconds)
        return 1

    elapsed_seconds = perf_counter() - start_time

    if loop_result.status != "success":
        trace_run_summary(elapsed_seconds, loop_result.tools_called)
        print_failure_summary(
            ValueError(loop_result.error or "Unknown error."),
            elapsed_seconds,
        )
        return 1

    trace_run_summary(elapsed_seconds, loop_result.tools_called)
    print_success_summary(loop_result.output or "", elapsed_seconds)
    return 0
