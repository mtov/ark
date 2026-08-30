# Ark

<p align="center">
  <img src="ark.png" alt="Ark logo" width="160">
</p>

Ark is a small, didactic coding agent for studying agent loops, constrained workspace tools, user-approved edits, and test-based validation. It is an ongoing research project developed by [ASERG](https://aserg.labsoft.dcc.ufmg.br/) at DCC/UFMG. See the [paper](https://arxiv.org/abs/2608.10934) for additional context.

## Overview

For each run, Ark copies a task workspace into `ark-workspace`, where the model can inspect files, run tests, and propose exact replacements in existing files. Ark previews each replacement as a diff and applies it only after user approval. The model ends with `finish`; Ark then runs the final tests.

Ark is intentionally narrow. It is designed for experiments and small curated tasks, not as a general-purpose autonomous coding environment.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
python run_ark.py ./test_workspace/bugfix_001_date_range
```

The default configuration uses the OpenAI API. To use an OpenAI-compatible endpoint, set `openai_base_url` in `config/config.json`. To use Ollama, set `model` to `ollama` and provide `ollama_model`:

```json
{
  "model": "ollama",
  "ollama_base_url": "http://localhost:11434",
  "ollama_model": "qwen2.5-coder:14b",
  "timeout_seconds": 600
}
```

`pytest` must be installed in the Python environment that runs Ark.

## Workspace

The command-line argument identifies the source workspace. It must contain:

- `prompt.txt`, describing the task.
- Project files that Ark may inspect and edit.
- Tests runnable from the workspace root.

`AGENTS.md` is optional. When present, Ark appends it to the task context as workspace-specific guidance.

Ark copies the source workspace to `ark-workspace` before each run and excludes `evaluation/` from that copy. All reads, edits, and tests operate only on this copied runtime workspace; Ark never modifies the source workspace.

Tests run with:

```bash
python -m pytest -q -p no:cacheprovider
```

## Agent Protocol

The model must return exactly one action per turn:

```text
Thought: brief reasoning
Action: tool_name
Action Input: tool-specific input
```

Ark generates tool results and adds them to the next model request as observations. The model must not generate `Observation`.

| Action | Action Input | Effect |
| --- | --- | --- |
| `list_files` | Relative directory; blank means `.` | Lists directory entries. |
| `read_file` | Relative file path | Returns UTF-8 file contents. |
| `find_text` | `search text | directory` | Searches files under a directory. |
| `run_tests` | Blank | Runs the fixed pytest command. |
| `edit_file` | `path`, `old`, and `new` blocks | Proposes one exact file replacement. |
| `finish` | Blank | Runs final tests and, on success, completes the run. |

Tool paths are constrained to `ark-workspace`. Ark discourages repeated reads and searches, and skips an identical consecutive `read_file` request.

## Editing

The model does not generate patches. Instead, it uses `edit_file` to replace an exact block in an existing file:

````text
path: src/orders.py
old:
```python
def subtotal(items):
    return sum(item["unit_price"] * item["quantity"] for item in items)
```
new:
```python
def subtotal(items):
    return calculate_subtotal(items)
```
````

Before asking for approval, Ark verifies that:

1. The path is inside `ark-workspace` and identifies an existing regular file.
2. The request has the required `path`, `old`, and `new` structure.
3. `old` occurs exactly once in the current file.
4. `new` differs from `old`.

Ark then generates and prints a unified diff:

```text
Authorize edit? [y/N]:
```

Only `y` or `yes` applies the edit. Rejected, malformed, ambiguous, out-of-workspace, and no-op edits leave the workspace unchanged. This version supports edits to existing files only; it does not create or delete files.

## Finishing And Rollback

`finish` is valid only with an empty `Action Input` and after at least one approved `edit_file` action. A valid finish runs the fixed test command.

- If tests pass, Ark keeps the edited runtime workspace and reports success.
- If tests fail, Ark returns a concise failure summary to the model and preserves approved edits so it can make corrective changes.
- Ark allows at most 20 model iterations. Reaching this limit, or an unexpected execution error, restores the runtime workspace to its state before the first approved edit.

The snapshot is created only when the first edit is approved and is discarded after a successful finish or rollback.

## Trace

Ark writes the most recent execution to `agent_trace.log`, clearing it at the start of each run. The relevant sections are:

- `[request]`: task prompt.
- `[response N]`: parsed model thought and requested action.
- `[edit_file]`: validation result, generated diff, or rejection reason.
- `[finish]`: finish validation or final-test outcome.
- `[validation_error]` and `[repair_attempt]`: protocol failures and repair requests.
- `[run_summary]`: token total, elapsed time, and ordered tools called.

`agent_trace.log` is intentionally excluded from Git.

## Structure

- `src/ark/agentic_loop.py`: main loop, completion handling, and transaction control.
- `src/ark/memory.py`: action history and context summary for the next model turn.
- `src/ark/tools.py`: workspace inspection, tests, and approved `edit_file` application.
- `src/ark/protocol.py`: parsing and validation of model responses.
- `src/ark/finish_handler.py`: final-test execution.
- `src/ark/inputs.py`: configuration, prompts, runtime workspace, and snapshots.
- `src/ark/traces.py`: execution trace output.

## Limits

Ark is a cooperative local tool, not a security sandbox. It restricts its own paths and test command, but it does not provide OS-level isolation. It is best suited to small workspaces whose changes can be expressed as exact replacements.
