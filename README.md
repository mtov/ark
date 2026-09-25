# Ark

<p align="center">
  <img src="ark.png" alt="Ark logo" width="160">
</p>

Ark is a small, didactic coding agent for studying how language models inspect code, propose changes, and validate them with tests. It is an ongoing research project developed by [ASERG](https://aserg.labsoft.dcc.ufmg.br/) at DCC/UFMG. See the [paper](https://arxiv.org/abs/2608.10934) for additional context.

## 1. Goal

Ark makes the core mechanics of a coding agent easy to inspect and experiment with. Its focus is not feature breadth, but a clear implementation of the agent loop:

1. Give a model a programming task and a local workspace.
2. Let it inspect files and run tests through a small set of constrained tools.
3. Ask for user approval before changing a file.
4. Run the tests before accepting the result.
5. Record the complete interaction in a readable trace.

Ark is intended for research, teaching, and small curated benchmarks. It is not a general-purpose autonomous development environment or an operating-system security sandbox.

## 2. Running Ark

### Requirements

- Python 3.11 or newer.
- An OpenAI API key, an OpenAI-compatible endpoint, or a local Ollama server.
- A task workspace containing a `prompt.txt` file, source code, and Pytest tests.

### Installation

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Ark runs task tests with the same Python environment. Install any task-specific dependencies in this virtual environment as well.

### Included test workspaces

The repository includes a `test_workspace/` directory with small, ready-to-run examples covering bug fixes, feature implementation, and refactoring. Each subdirectory is an independent task with its own prompt, source code, and tests.

For example:

```bash
python run_ark.py ./test_workspace/bugfix_001_date_range
```

Ark copies the selected example to `ark-workspace`, so the original files under `test_workspace/` remain unchanged and can be reused in later runs.

### OpenAI

The default configuration in `config/config.json` uses the OpenAI API. Export your key and pass a task workspace to Ark:

```bash
export OPENAI_API_KEY="your_key_here"
python run_ark.py ./test_workspace/bugfix_001_date_range
```

To use another OpenAI-compatible service, set `openai_base_url`, `openai_model`, and, when necessary, `openai_api_key_env` in `config/config.json`.

### Ollama

To run with Ollama, start the Ollama server, make sure the selected model is available, and update `config/config.json`:

```json
{
  "model": "ollama",
  "ollama_base_url": "http://localhost:11434",
  "ollama_model": "qwen2.5-coder:14b",
  "timeout_seconds": 600
}
```

Then run the same command:

```bash
python run_ark.py ./test_workspace/bugfix_001_date_range
```

### During a run

Ark copies the supplied task into `ark-workspace` and works only on that copy. Whenever the model proposes an edit, Ark prints a unified diff and asks:

```text
Authorize edit? [y/N]:
```

Only `y` or `yes` applies the change. When the run succeeds, the final files remain in `ark-workspace`; the original task directory is never modified. The full execution trace is written to `agent_trace.log`.

### Task workspace format

A task directory should look like this:

```text
my-task/
├── prompt.txt
├── src/
│   └── ...
└── tests/
    └── ...
```

- `prompt.txt` contains the task given to the model.
- Source and test files are available to the agent inside the copied workspace.
- `AGENTS.md` is optional and adds task-specific instructions to the prompt.
- `evaluation/` is intentionally excluded from the runtime copy, so private evaluation tests are not visible to the agent.

Tests are always run from the workspace root with:

```bash
python -m pytest -q -p no:cacheprovider
```

## 3. Architecture

At a high level, one run follows this flow:

```text
task workspace
      |
      v
copy to ark-workspace
      |
      v
model response -> parse one tool call -> execute tool -> store observation
      ^                                              |
      |______________________________________________|
      |
      v
finish -> run tests -> keep changes or continue/rollback
```

The main modules are deliberately small and have distinct responsibilities:

- `inputs.py` loads configuration and prompts, prepares `ark-workspace`, and manages workspace snapshots.
- `agentic_loop.py` coordinates model calls, tool execution, completion, and rollback.
- `models.py` provides the OpenAI-compatible and Ollama integrations.
- `protocol.py` parses and validates the model's requested action.
- `tools.py` implements workspace inspection, test execution, and approved edits.
- `memory.py` builds the compact history sent with the next model request.
- `finish_handler.py` validates completion and runs the final tests.
- `traces.py` records the execution in `agent_trace.log`.

## Agent loop

The model returns exactly one action per iteration:

```text
Thought: brief reasoning
Action: tool_name
Action Input: tool-specific input
```

Ark executes the action, adds its result to memory as an observation, and calls the model again. If the response does not follow the protocol, Ark makes one repair request.

The available actions are:

| Action | Input | Purpose |
| --- | --- | --- |
| `list_files` | Relative directory, or blank for `.` | List workspace entries. |
| `read_file` | Relative file path | Read a UTF-8 file. |
| `find_text` | `search text \| directory` | Search files below a directory. |
| `run_tests` | Blank | Run the fixed Pytest command. |
| `edit_file` | `path`, `old`, and `new` blocks | Propose one exact replacement. |
| `finish` | Blank | Request final validation. |

All tool paths are restricted to `ark-workspace`.

## Edits and approval

`edit_file` replaces one exact block in an existing file. For example:

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

Before requesting approval, Ark checks that the path is inside the runtime workspace, the file exists, `old` occurs exactly once, and `new` is different. This version edits existing files only; it does not create or delete files.

The first approved edit creates a temporary snapshot. A successful `finish` keeps the runtime changes and discards the snapshot. Reaching the 20-iteration limit or encountering an unexpected loop error restores the runtime workspace from that snapshot.

## Memory and trace

Each model request receives a compact summary rather than the entire raw conversation. Memory includes files already read, searches already made, whether tests have run, and the four most recent steps. Each observation is limited to 8,000 characters, and large `edit_file` inputs are represented only by their target path.

`agent_trace.log` is cleared at the beginning of each run and records:

- The task prompt and parsed model actions.
- Edit validation, approval outcomes, and diffs.
- Test and finish results.
- Protocol validation and repair attempts.
- Errors, rollback events, elapsed time, token usage, and tool counts.

The trace is intended to make agent behavior reproducible and easy to analyze. It is excluded from Git.

## Limits

Ark constrains its own file paths and test command, but task tests still execute as local Python processes. Run only trusted task workspaces. Ark is best suited to small tasks whose changes can be expressed as exact replacements in existing files.
