from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ark import traces  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_trace(monkeypatch, tmp_path_factory) -> None:
    monkeypatch.setattr(traces, "LOG_PATH", tmp_path_factory.mktemp("trace") / "agent_trace.log")
    traces.clear_trace()
