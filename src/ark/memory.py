from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import ToolCall

MAX_HISTORY_ENTRIES = 4
MAX_OBSERVATION_CHARS = 8000


@dataclass
class MemoryEntry:
    iteration: int
    tool_call: ToolCall
    result: str


@dataclass
class Memory:
    entries: list[MemoryEntry] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)

    def append(self, iteration: int, tool_call: ToolCall, result: str) -> None:
        self.entries.append(MemoryEntry(iteration, tool_call, result))

    def record_tool_call(self, name: str) -> None:
        self.tools_called.append(name)

    def contains_tool(self, name: str) -> bool:
        return any(entry.tool_call.name == name for entry in self.entries)

    def has_successful_edit(self) -> bool:
        return any(
            entry.tool_call.name == "edit_file"
            and entry.result.startswith("Edit applied successfully to ")
            for entry in self.entries
        )

    def unique_args_for(self, name: str) -> list[str]:
        seen: set[str] = set()
        items: list[str] = []

        for entry in self.entries:
            if entry.tool_call.name != name:
                continue
            value = entry.tool_call.args.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            items.append(value)

        return items

    @staticmethod
    def _context_args(tool_call: ToolCall) -> str:
        if tool_call.name != "edit_file":
            return tool_call.args

        for line in tool_call.args.splitlines():
            if line.startswith("path:"):
                return line

        return "<edit content omitted>"

    def to_text(self) -> str:
        if not self.entries:
            return "No previous steps."

        sections: list[str] = []
        read_files = self.unique_args_for("read_file")
        find_queries = self.unique_args_for("find_text")

        if read_files:
            sections.append("Files already read:\n" + "\n".join(f"- {path}" for path in read_files))
        if find_queries:
            sections.append("Searches already run:\n" + "\n".join(f"- {query}" for query in find_queries))
        if self.contains_tool("run_tests"):
            sections.append("Tests already run: yes")
        formatted_entries = []
        for entry in self.entries[-MAX_HISTORY_ENTRIES:]:
            result = entry.result.strip()
            if len(result) > MAX_OBSERVATION_CHARS:
                result = f"{result[:MAX_OBSERVATION_CHARS].rstrip()}..."
            formatted_entries.append(
                f"Iteration {entry.iteration}\n"
                f"Tool: {entry.tool_call.name}\n"
                f"Tool Args: {self._context_args(entry.tool_call)}\n"
                f"Observation: {result}\n"
            )
        sections.append("Recent steps:\n" + "\n".join(formatted_entries))
        return "\n\n".join(sections)
