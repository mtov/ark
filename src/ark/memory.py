from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import ToolRequest

MAX_HISTORY_ENTRIES = 4
MAX_OBSERVATION_CHARS = 1200


@dataclass
class MemoryEntry:
    iteration: int
    tool_request: ToolRequest
    result: str


@dataclass
class Memory:
    entries: list[MemoryEntry] = field(default_factory=list)

    def append(self, iteration: int, request: ToolRequest, result: str) -> None:
        self.entries.append(MemoryEntry(iteration, request, result))

    def contains_tool(self, name: str) -> bool:
        return any(entry.tool_request.name == name for entry in self.entries)

    def has_successful_edit(self) -> bool:
        return any(
            entry.tool_request.name == "edit_file"
            and entry.result.startswith("Edit applied successfully to ")
            for entry in self.entries
        )

    def last_tool_request(self) -> ToolRequest | None:
        if not self.entries:
            return None
        return self.entries[-1].tool_request

    def unique_args_for(self, name: str) -> list[str]:
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

    @staticmethod
    def _context_args(tool_request: ToolRequest) -> str:
        if tool_request.name != "edit_file":
            return tool_request.args

        for line in tool_request.args.splitlines():
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
                f"Tool: {entry.tool_request.name}\n"
                f"Tool Args: {self._context_args(entry.tool_request)}\n"
                f"Observation: {result}\n"
            )
        sections.append("Recent steps:\n" + "\n".join(formatted_entries))
        return "\n\n".join(sections)
