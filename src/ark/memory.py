from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .guards import resolve_tool_path
from .protocol import ToolCall, parse_read_file_request

MAX_HISTORY_ENTRIES = 4
MAX_OBSERVATION_CHARS = 8000
MAX_RETAINED_READS = 3


@dataclass
class MemoryEntry:
    iteration: int
    tool_call: ToolCall
    result: str
    skipped: bool = False
    file_path: Path | None = None


@dataclass
class Memory:
    entries: list[MemoryEntry] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)

    def append(
        self,
        iteration: int,
        tool_call: ToolCall,
        result: str,
        *,
        skipped: bool = False,
        file_path: Path | None = None,
    ) -> None:
        self.entries.append(MemoryEntry(iteration, tool_call, result, skipped, file_path))

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

    def _current_reads(self) -> list[MemoryEntry]:
        invalidated: set[Path] = set()
        seen: set[tuple[Path, int, int | None]] = set()
        reads = []
        for entry in reversed(self.entries):
            if entry.tool_call.name == "edit_file" and entry.file_path is not None:
                invalidated.add(entry.file_path)
            if entry.tool_call.name != "read_file" or entry.skipped or entry.file_path is None:
                continue
            if entry.file_path in invalidated:
                continue
            request = parse_read_file_request(entry.tool_call.args)
            key = (entry.file_path, request.start, request.end)
            if key not in seen:
                reads.append(entry)
                seen.add(key)
        return reads

    def _context_entries(self) -> list[MemoryEntry]:
        recent = self.entries[-MAX_HISTORY_ENTRIES:]
        recent_ids = {id(entry) for entry in recent}
        retained = [entry for entry in self._current_reads() if id(entry) not in recent_ids]
        return sorted(retained[:MAX_RETAINED_READS] + recent, key=lambda entry: entry.iteration)

    def has_current_file_read(self, action_input: str, workspace_path: Path) -> bool:
        try:
            request = parse_read_file_request(action_input)
            path = resolve_tool_path(request.path, workspace_path)
        except ValueError:
            return False
        visible_ids = {id(entry) for entry in self._context_entries()}
        for entry in self._current_reads():
            previous = parse_read_file_request(entry.tool_call.args)
            if id(entry) in visible_ids and entry.file_path == path:
                if (previous.start, previous.end) == (request.start, request.end):
                    return True
        return False

    def unique_args_for(self, name: str) -> list[str]:
        seen: set[str] = set()
        items: list[str] = []

        for entry in self.entries:
            if entry.tool_call.name != name:
                continue
            if name == "read_file" and entry.file_path is None:
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

        edit_path = tool_call.edit_path()
        if edit_path is not None:
            return f"path: {edit_path}"

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
        current_read_ids = {id(entry) for entry in self._current_reads()}
        formatted_entries = []
        for entry in self._context_entries():
            result = entry.result.strip()
            if entry.tool_call.name == "read_file" and entry.file_path is not None:
                if id(entry) not in current_read_ids:
                    result = "File contents superseded by a later read or invalidated by an approved edit."
            if len(result) > MAX_OBSERVATION_CHARS:
                result = f"{result[:MAX_OBSERVATION_CHARS].rstrip()}..."
            formatted_entries.append(
                f"Iteration {entry.iteration}\n"
                f"Tool: {entry.tool_call.name}\n"
                f"Tool Args: {self._context_args(entry.tool_call)}\n"
                f"Observation: {result}\n"
            )
        sections.append("Recent steps and retained reads:\n" + "\n".join(formatted_entries))
        return "\n\n".join(sections)
