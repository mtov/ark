from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import call_model
from .traces import trace_repair_attempt

if TYPE_CHECKING:
    from .inputs import AgentConfig


@dataclass
class ToolRequest:
    thought: str
    name: str
    args: str


@dataclass(frozen=True)
class EditFileRequest:
    path: str
    old: str
    new: str

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

def parse_response(text: str) -> ToolRequest:
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
    return ToolRequest(thought=thought, name=action, args=action_input)


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


def repair_response(config: AgentConfig, user_message: str, reason: str) -> ToolRequest:
    trace_repair_attempt("Protocol repair", reason)
    repair_message = f"{user_message}\n\n{REPAIR_PROMPT}"
    response = call_model(config, repair_message)
    return parse_response(response.content)
