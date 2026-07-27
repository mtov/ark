from __future__ import annotations

from pathlib import Path


def resolve_tool_path(target_path: str, workspace_path: Path) -> Path:
    raw_path = Path(target_path.strip()).expanduser()
    resolved_path = raw_path.resolve() if raw_path.is_absolute() else (workspace_path / raw_path).resolve()
    workspace_root = workspace_path.resolve()

    try:
        resolved_path.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("Path is outside the workspace.") from exc

    return resolved_path
