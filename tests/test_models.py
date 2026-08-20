from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ark.inputs import AgentConfig
from ark.models import (
    ModelConfig,
    call_ollama,
    extract_ollama_content,
    extract_ollama_usage,
    extract_openai_content,
    extract_openai_usage,
)


def test_extract_openai_usage_uses_prompt_and_completion_tokens() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_tokens=20)
    )

    usage = extract_openai_usage(response)

    assert usage.input_tokens == 12
    assert usage.output_tokens == 8
    assert usage.total_tokens == 20


def test_extract_openai_usage_returns_unavailable_without_usage() -> None:
    usage = extract_openai_usage(SimpleNamespace())

    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None


def test_extract_openai_content_reads_string_message_content() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" final answer "))]
    )

    content = extract_openai_content(response)

    assert content == "final answer"


def test_extract_openai_content_reads_text_parts() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "part one"},
                        {"type": "text", "text": " and part two"},
                    ]
                }
            }
        ]
    }

    content = extract_openai_content(response)

    assert content == "part one and part two"


def test_extract_ollama_content_reads_response_field() -> None:
    content = extract_ollama_content({"response": " final answer "})

    assert content == "final answer"


def test_extract_ollama_usage_reads_eval_counts() -> None:
    usage = extract_ollama_usage({"prompt_eval_count": 12, "eval_count": 8})

    assert usage.input_tokens == 12
    assert usage.output_tokens == 8
    assert usage.total_tokens == 20


def test_extract_ollama_usage_returns_unavailable_without_usage() -> None:
    usage = extract_ollama_usage({})

    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None


def test_call_ollama_uses_local_endpoint(monkeypatch) -> None:
    config = AgentConfig(
        model_config=ModelConfig(
            model="ollama",
            timeout_seconds=30,
            openai_base_url=None,
            openai_model=None,
            openai_api_key_env=None,
            ollama_base_url="http://localhost:11434",
            ollama_model="qwen2.5-coder:14b",
        ),
        system_prompt="system prompt",
        user_prompt="user prompt",
        source_workspace_path=SimpleNamespace(),
        workspace_path=SimpleNamespace(),
    )
    seen: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "response": "Action: list_files\nAction Input: \n",
                    "prompt_eval_count": 10,
                    "eval_count": 4,
                }
            ).encode("utf-8")

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("ark.models.request.urlopen", fake_urlopen)

    response = call_ollama(config, "workspace prompt")

    assert response.content == "Action: list_files\nAction Input:"
    assert response.token_usage.total_tokens == 14
    assert seen["url"] == "http://localhost:11434/api/generate"
    assert seen["timeout"] == 30
    assert seen["body"] == {
        "model": "qwen2.5-coder:14b",
        "system": "system prompt",
        "prompt": "workspace prompt",
        "stream": False,
    }


def test_call_ollama_requires_model_name() -> None:
    config = AgentConfig(
        model_config=ModelConfig(
            model="ollama",
            timeout_seconds=30,
            openai_base_url=None,
            openai_model=None,
            openai_api_key_env=None,
            ollama_base_url=None,
            ollama_model=None,
        ),
        system_prompt="system prompt",
        user_prompt="user prompt",
        source_workspace_path=SimpleNamespace(),
        workspace_path=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="Missing ollama_model"):
        call_ollama(config, "workspace prompt")
