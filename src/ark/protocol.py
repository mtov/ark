from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .traces import trace_repair_attempt

if TYPE_CHECKING:
    from .inputs import Context


@dataclass
class ToolCall:
    thought: str
    name: str
    args: str

    def edit_path(self) -> str | None:
        for line in self.args.splitlines():
            if line.startswith("path:"):
                return line.removeprefix("path:").strip()
        return None


@dataclass(frozen=True)
class EditFileRequest:
    path: str
    old: str
    new: str


@dataclass(frozen=True)
class ReadFileRequest:
    path: str
    start: int = 1
    end: int | None = None


def parse_read_file_request(text: str) -> ReadFileRequest:
    path, separator, interval = text.strip().partition("|")
    path = path.strip()
    if not path:
        raise ValueError("read_file requires a file path.")
    if not separator:
        return ReadFileRequest(path)
    match = re.fullmatch(r"\s*(\d+):(\d+)\s*", interval)
    if match is None:
        raise ValueError("Invalid read_file range. Use: path | start:end (inclusive, starting at 1).")
    start, end = map(int, match.groups())
    if start < 1 or end < start:
        raise ValueError("read_file requires 1 <= start <= end.")
    return ReadFileRequest(path, start, end)


REPAIR_PROMPT = (
    "Your previous response was invalid. "
    "Respond using only: "
    "Thought: ... "
    "Action: ... "
    "Action Input: ... "
    "Do not include Observation."
)
EDIT_FILE_PATTERN = re.compile(
    r"\Apath:\s*(?P<path>[^\n]+)\n"
    r"old:\s*\n```[^\n]*\n(?P<old>.*?)\n```\n"
    r"new:\s*\n```[^\n]*\n(?P<new>.*?)\n```\s*\Z",
    re.DOTALL,
)

def parse_response(text: str) -> ToolCall:
    thought = ""
    action = ""
    action_input_lines: list[str] = []
    current_section = None

    for line in text.splitlines():
        if line.startswith("Thought:"):
            thought = line.removeprefix("Thought:").strip()
            current_section = None
        elif line.startswith("Action:"):
            action = line.removeprefix("Action:").strip()
            current_section = None
        elif line.startswith("Action Input:"):
            action_input_lines = [line.removeprefix("Action Input:").strip()]
            current_section = "action_input"
        elif line.startswith("Observation:"):
            raise ValueError("Model response must not contain Observation.")
        elif current_section == "action_input":
            action_input_lines.append(line)

    if not action:
        raise ValueError("Model response is missing the required Action field.")

    action_input = "\n".join(action_input_lines).strip()
    return ToolCall(thought=thought, name=action, args=action_input)


def parse_edit_file_request(text: str) -> EditFileRequest:
    match = EDIT_FILE_PATTERN.match(text.strip())
    if match is None:
        raise ValueError(
            "Invalid edit_file input. Use path:, old:, and new: with triple-backtick blocks."
        )

    return EditFileRequest(
        path=match.group("path").strip(),
        old=match.group("old"),
        new=match.group("new"),
    )


def repair_response(context: Context, reason: str) -> ToolCall:
    trace_repair_attempt("Protocol repair", reason)
    repair_history = f"{context.memory.to_text()}\n\n{REPAIR_PROMPT}"
    response = context.model.call(context.user_prompt, repair_history)
    return parse_response(response.content)
