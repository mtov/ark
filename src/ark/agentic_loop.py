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
from .inputs import AgentConfig, parse_args, prepare_run, reset_runtime_workspace
from .models import call_model
from .protocol import ToolRequest, looks_like_patch, parse_response, repair_response
from .test_failures import summarize_test_failure_output
from .tools import run_tool
from .traces import trace_validation_error

MAX_HISTORY_ENTRIES = 4
MAX_OBSERVATION_CHARS = 1200
MAX_ITERATIONS_REACHED_MESSAGE = "Agent stopped after reaching the maximum number of steps."
INVALID_FINISH_MESSAGE = (
    "Finish output must contain only a unified diff patch. "
    "Do not end the run yet; inspect any remaining files you need and then return the patch."
)
MAX_ITERATIONS = 15


@dataclass
class LoopResult:
    status: str
    output: str | None = None
    error: str | None = None
    post_apply_tests_passed: bool = False

    @classmethod
    def success(
        cls,
        output: str,
        *,
        post_apply_tests_passed: bool = False,
    ) -> LoopResult:
        return cls(
            status="success",
            output=output,
            post_apply_tests_passed=post_apply_tests_passed,
        )

    @classmethod
    def error(cls, message: str, *, status: str = "error") -> LoopResult:
        return cls(status=status, error=message)

    @classmethod
    def max_iterations_reached(cls) -> LoopResult:
        return cls(
            status="max_iterations_reached",
            error=MAX_ITERATIONS_REACHED_MESSAGE,
        )


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
        return parse_response(model_response.content)
    except ValueError as exc:
        trace_validation_error(str(exc), model_response.content)
        return repair_response(config, user_message, str(exc))


def handle_post_apply_test_failure(
    config: AgentConfig,
    memory: Memory,
    iteration: int,
    tool_request: ToolRequest,
    test_output: str | None,
) -> None:
    print_iteration_action(iteration, tool_request)
    reset_runtime_workspace(config)
    memory.append(
        iteration,
        tool_request,
        summarize_test_failure_output(test_output or ""),
    )


def handle_finish(
    config: AgentConfig,
    memory: Memory,
    iteration: int,
    tool_request: ToolRequest,
) -> str | None:
    if not looks_like_patch(tool_request.args):
        print_iteration_action(iteration, tool_request)
        memory.append(iteration, tool_request, INVALID_FINISH_MESSAGE)
        return None

    finish_result = apply_finish(config, tool_request)
    if finish_result.status == "post_apply_tests_failed":
        handle_post_apply_test_failure(
            config,
            memory,
            iteration,
            tool_request,
            finish_result.test_output,
        )
        return None

    tool_request = finish_result.request
    print_iteration_action(iteration, tool_request)
    return tool_request.args


def agentic_loop(config: AgentConfig) -> LoopResult:
    memory = Memory()

    for iteration in range(1, MAX_ITERATIONS + 1):
        tool_request = get_next_tool_request(config, memory)

        if tool_request.name == "finish":
            output = handle_finish(config, memory, iteration, tool_request)
            if output is None:
                continue
            return LoopResult.success(output, post_apply_tests_passed=True)

        print_iteration_action(iteration, tool_request)

        result = run_tool(tool_request, config)
        memory.append(iteration, tool_request, result)

    return LoopResult.max_iterations_reached()


def main() -> int:
    args = parse_args()
    start_time = perf_counter()

    try:
        config = prepare_run(args.workspace_path)
        loop_result = agentic_loop(config)
    except Exception as exc:  # noqa: BLE001
        print_total_tokens()
        print_elapsed_time(perf_counter() - start_time)
        print(format_failure_message(exc))
        return 1

    if loop_result.status != "success":
        print_total_tokens()
        print_elapsed_time(perf_counter() - start_time)
        print(format_failure_message(ValueError(loop_result.error or "Unknown error.")))
        return 1

    print_total_tokens()
    print_elapsed_time(perf_counter() - start_time)
    print_final_result(loop_result.output or "", loop_result)
    return 0
