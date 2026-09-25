from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from .cli_output import (
    print_failure_summary,
    print_tool_call,
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
from .models import call_model_api
from .protocol import ToolCall, parse_response, repair_response
from .tools import run_tool
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


@dataclass
class LoopResult:
    status: str
    output: str | None = None
    error: str | None = None
    error_type: str | None = None
    tools_called: list[str] = field(default_factory=list)

    @classmethod
    def success(
        cls,
        output: str,
        memory: Memory,
    ) -> LoopResult:
        return cls(
            status="success",
            output=output,
            tools_called=memory.tools_called.copy(),
        )

    @classmethod
    def max_iterations_reached(cls, memory: Memory) -> LoopResult:
        return cls(
            status="max_iterations_reached",
            error=MAX_ITERATIONS_REACHED_MESSAGE,
            tools_called=memory.tools_called.copy(),
        )

    @classmethod
    def failure(cls, error: Exception, memory: Memory) -> LoopResult:
        return cls(
            status="failed",
            error=str(error),
            error_type=error.__class__.__name__,
            tools_called=memory.tools_called.copy(),
        )


def invoke_model(config: AgentConfig, memory: Memory) -> ToolCall:
    user_message = (
        "User task:\n"
        f"{config.user_prompt}\n\n"
        "Agent history:\n"
        f"{memory.to_text()}"
    )
    model_response = call_model_api(config, user_message)

    try:
        tool_call = parse_response(model_response.content)
    except ValueError as exc:
        trace_validation_error(str(exc), model_response.content)
        tool_call = repair_response(config, user_message, str(exc))

    trace_action(tool_call)
    memory.record_tool_call(tool_call.name)
    return tool_call


def handle_finish(
    config: AgentConfig,
    memory: Memory,
    iteration: int,
    tool_call: ToolCall,
) -> str | None:
    if not memory.has_successful_edit():
        print_tool_call(iteration, tool_call)
        trace_finish_event("failed", "finish_validation", FINISH_WITHOUT_EDIT_MESSAGE)
        memory.append(iteration, tool_call, FINISH_WITHOUT_EDIT_MESSAGE)
        return None

    finish_result = apply_finish(config, tool_call)
    print_tool_call(iteration, tool_call)

    if finish_result.status == "invalid_finish":
        memory.append(
            iteration,
            tool_call,
            "Finish action must have an empty Action Input.",
        )
        return None

    memory.record_tool_call("run_tests")
    if finish_result.status == "post_apply_tests_failed":
        memory.append(
            iteration,
            tool_call,
            finish_result.test_output or "Tests failed without output.",
        )
        return None

    return FINISH_SUCCESS_MESSAGE


def agentic_loop(config: AgentConfig) -> LoopResult:
    memory = Memory()
    try:
        for iteration in range(1, MAX_ITERATIONS + 1):
            tool_call = invoke_model(config, memory)

            if tool_call.name == "finish":
                finish_output = handle_finish(
                    config,
                    memory,
                    iteration,
                    tool_call,
                )
                if finish_output is None:
                    continue

                commit_workspace_changes(config)
                return LoopResult.success(finish_output, memory)

            tool_result = run_tool(tool_call, config, memory)
            print_tool_call(iteration, tool_call, tool_result.note)
            memory.append(
                iteration,
                tool_call,
                tool_result.output,
                skipped=tool_result.skipped,
            )

        rollback_workspace_changes(config, MAX_ITERATIONS_REACHED_MESSAGE)
        return LoopResult.max_iterations_reached(memory)
    except Exception as exc:  # noqa: BLE001
        rollback_workspace_changes(config, AGENTIC_LOOP_ERROR_MESSAGE)
        return LoopResult.failure(exc, memory)


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
        trace_error("prepare_run", str(exc), exc.__class__.__name__, [])
        trace_run_summary(elapsed_seconds, [])
        print_failure_summary(exc, elapsed_seconds)
        return 1

    elapsed_seconds = perf_counter() - start_time

    if loop_result.status != "success":
        if loop_result.status == "failed":
            trace_error(
                "agentic_loop",
                loop_result.error or "Unknown error.",
                loop_result.error_type or "Exception",
                loop_result.tools_called,
            )
        trace_run_summary(elapsed_seconds, loop_result.tools_called)
        print_failure_summary(
            ValueError(loop_result.error or "Unknown error."),
            elapsed_seconds,
        )
        return 1

    trace_run_summary(elapsed_seconds, loop_result.tools_called)
    print_success_summary(loop_result.output or "", elapsed_seconds)
    return 0
