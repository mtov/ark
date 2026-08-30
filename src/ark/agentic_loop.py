from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from .cli_output import (
    format_failure_message,
    print_elapsed_time,
    print_final_result,
    print_iteration_action,
    print_total_tokens,
)
from .finish_handler import apply_finish
from .inputs import (
    AgentConfig,
    commit_workspace_transaction,
    parse_args,
    prepare_run,
    rollback_workspace_transaction,
)
from .models import call_model
from .protocol import ToolRequest, parse_response, repair_response
from .test_failures import summarize_test_failure_output
from .tools import REDUNDANT_READ_FILE_MESSAGE, run_tool_with_status
from .traces import trace_action, trace_run_summary, trace_validation_error

MAX_HISTORY_ENTRIES = 4
MAX_OBSERVATION_CHARS = 1200
MAX_ITERATIONS_REACHED_MESSAGE = "Agent stopped after reaching the maximum number of steps."
MAX_ITERATIONS = 15
FINISH_SUCCESS_MESSAGE = "Changes applied and final tests passed."


@dataclass
class LoopResult:
    status: str
    output: str | None = None
    error: str | None = None
    post_apply_tests_passed: bool = False
    tools_called: list[str] = field(default_factory=list)

    @classmethod
    def success(
        cls,
        output: str,
        *,
        post_apply_tests_passed: bool = False,
        tools_called: list[str] | None = None,
    ) -> LoopResult:
        return cls(
            status="success",
            output=output,
            post_apply_tests_passed=post_apply_tests_passed,
            tools_called=tools_called or [],
        )

    @classmethod
    def error(
        cls,
        message: str,
        *,
        status: str = "error",
        tools_called: list[str] | None = None,
    ) -> LoopResult:
        return cls(status=status, error=message, tools_called=tools_called or [])

    @classmethod
    def max_iterations_reached(cls, *, tools_called: list[str] | None = None) -> LoopResult:
        return cls(
            status="max_iterations_reached",
            error=MAX_ITERATIONS_REACHED_MESSAGE,
            tools_called=tools_called or [],
        )


@dataclass
class FinishResult:
    output: str


@dataclass
class MemoryEntry:
    iteration: int
    tool_request: ToolRequest
    result: str


@dataclass
class Memory:
    entries: list[MemoryEntry] = field(default_factory=list)

    def append(self, iteration: int, request: ToolRequest, result: str) -> None:
        self.entries.append(
            MemoryEntry(
                iteration=iteration,
                tool_request=request,
                result=result,
            )
        )

    def contains_tool(self, name: str) -> bool:
        return any(entry.tool_request.name == name for entry in self.entries)

    def last_tool_request(self) -> ToolRequest | None:
        if not self.entries:
            return None
        return self.entries[-1].tool_request

    def _unique_tool_args(self, name: str) -> list[str]:
        seen: set[str] = set()
        items: list[str] = []

        for entry in self.entries:
            if entry.tool_request.name != name:
                continue
            value = entry.tool_request.args.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            items.append(value)

        return items

    def to_text(self) -> str:
        if not self.entries:
            return "No previous steps."

        sections: list[str] = []
        read_files = self._unique_tool_args("read_file")
        find_queries = self._unique_tool_args("find_text")

        if read_files:
            sections.append(
                "Files already read:\n"
                + "\n".join(f"- {path}" for path in read_files)
            )
        if find_queries:
            sections.append(
                "Searches already run:\n"
                + "\n".join(f"- {query}" for query in find_queries)
            )
        if self.contains_tool("run_tests"):
            sections.append("Tests already run: yes")

        formatted_entries = []
        for entry in self.entries[-MAX_HISTORY_ENTRIES:]:
            result = entry.result.strip()
            if len(result) > MAX_OBSERVATION_CHARS:
                result = f"{result[:MAX_OBSERVATION_CHARS].rstrip()}..."
            formatted_entries.append(
                f"Iteration {entry.iteration}\n"
                f"Tool: {entry.tool_request.name}\n"
                f"Tool Args: {entry.tool_request.args}\n"
                f"Observation: {result}\n"
            )
        sections.append("Recent steps:\n" + "\n".join(formatted_entries))
        return "\n\n".join(sections)

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
) -> FinishResult | None:
    finish_result = apply_finish(config, tool_request)
    print_iteration_action(iteration, tool_request)
    if finish_result.status == "invalid_finish":
        memory.append(iteration, tool_request, "Finish action must have an empty Action Input.")
        return None

    tools_called.append("run_tests")
    if finish_result.status == "post_apply_tests_failed":
        memory.append(
            iteration,
            tool_request,
            summarize_test_failure_output(finish_result.test_output or ""),
        )
        return None

    return FinishResult(output=FINISH_SUCCESS_MESSAGE)

def agentic_loop(config: AgentConfig) -> LoopResult:
    try:
        memory = Memory()
        tools_called: list[str] = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            tool_request = get_next_tool_request(config, memory)
            tools_called.append(tool_request.name)

            if tool_request.name == "finish":
                finish_output = handle_finish(config, memory, iteration, tool_request, tools_called)
                if finish_output is None:
                    continue
                commit_workspace_transaction(config)
                return LoopResult.success(
                    finish_output.output,
                    post_apply_tests_passed=True,
                    tools_called=tools_called,
                )

            previous_request = memory.last_tool_request()
            result, note = run_tool_with_status(tool_request, config, previous_request)
            print_iteration_action(iteration, tool_request, note)
            memory.append(iteration, tool_request, result)

        rollback_workspace_transaction(config)
        return LoopResult.max_iterations_reached(tools_called=tools_called)
    except Exception:
        rollback_workspace_transaction(config)
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
            rollback_workspace_transaction(config)
        elapsed_seconds = perf_counter() - start_time
        trace_run_summary(elapsed_seconds, [])
        print_total_tokens()
        print_elapsed_time(elapsed_seconds)
        print(format_failure_message(exc))
        return 1

    elapsed_seconds = perf_counter() - start_time

    if loop_result.status != "success":
        trace_run_summary(elapsed_seconds, loop_result.tools_called)
        print_total_tokens()
        print_elapsed_time(elapsed_seconds)
        print(format_failure_message(ValueError(loop_result.error or "Unknown error.")))
        return 1

    trace_run_summary(elapsed_seconds, loop_result.tools_called)
    print_total_tokens()
    print_elapsed_time(elapsed_seconds)
    print_final_result(loop_result.output or "", loop_result)
    return 0
