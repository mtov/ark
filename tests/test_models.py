from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from ark.inputs import AgentConfig
from ark.models import (
    EmptyModelResponseError,
    ModelConfig,
    call_model_api,
    call_ollama,
    call_openai_compatible,
    extract_ollama_content,
    extract_ollama_usage,
    extract_openai_content,
    extract_openai_usage,
)


def build_openai_context() -> AgentConfig:
    return AgentConfig(
        model_config=ModelConfig(
            model="openai-compatible",
            timeout_seconds=30,
            openai_base_url="http://localhost:8000/v1",
            openai_model="local-model",
            openai_api_key_env=None,
        ),
        system_prompt="system prompt",
        user_prompt="user prompt",
        source_workspace_path=SimpleNamespace(),
        workspace_path=SimpleNamespace(),
    )


def install_fake_openai(monkeypatch, responses: list[dict[str, object]]) -> dict[str, int]:
    seen = {"calls": 0}

    class FakeOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        def create_completion(self, **_kwargs: object) -> dict[str, object]:
            response = responses[seen["calls"]]
            seen["calls"] += 1
            return response

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    return seen


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


def test_extract_openai_content_reports_empty_response_details() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": None, "refusal": "policy"},
            }
        ]
    }

    with pytest.raises(ValueError, match="without message content") as error:
        extract_openai_content(response)

    message = str(error.value)
    assert "finish_reason='length'" in message
    assert "refusal='policy'" in message
    assert 'response={"choices"' in message


def test_call_model_api_retries_empty_openai_response(monkeypatch) -> None:
    responses = [
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "", "refusal": None},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        },
        {
            "choices": [{"message": {"content": "final answer"}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
        },
    ]
    seen = install_fake_openai(monkeypatch, responses)
    recorded_usage: list[int | None] = []
    monkeypatch.setattr(
        "ark.models.record_response_usage",
        lambda usage: recorded_usage.append(usage.total_tokens),
    )

    response = call_model_api(build_openai_context(), "workspace prompt")

    assert response.content == "final answer"
    assert seen["calls"] == 2
    assert recorded_usage == [11, 8]


def test_call_model_api_stops_after_second_empty_openai_response(monkeypatch) -> None:
    empty_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "", "refusal": None},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }
    seen = install_fake_openai(monkeypatch, [empty_response, empty_response])
    recorded_usage: list[int | None] = []
    monkeypatch.setattr(
        "ark.models.record_response_usage",
        lambda usage: recorded_usage.append(usage.total_tokens),
    )

    with pytest.raises(EmptyModelResponseError, match="without message content"):
        call_model_api(build_openai_context(), "workspace prompt")

    assert seen["calls"] == 2
    assert recorded_usage == [11, 11]


def test_call_model_api_does_not_retry_openai_refusal(monkeypatch) -> None:
    refusal_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "", "refusal": "policy"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }
    seen = install_fake_openai(monkeypatch, [refusal_response])

    with pytest.raises(ValueError, match="refusal='policy'") as error:
        call_model_api(build_openai_context(), "workspace prompt")

    assert not isinstance(error.value, EmptyModelResponseError)
    assert seen["calls"] == 1


def test_call_openai_compatible_uses_client_timeout(monkeypatch) -> None:
    config = AgentConfig(
        model_config=ModelConfig(
            model="openai-compatible",
            timeout_seconds=30,
            openai_base_url="http://localhost:8000/v1",
            openai_model="local-model",
            openai_api_key_env=None,
        ),
        system_prompt="system prompt",
        user_prompt="user prompt",
        source_workspace_path=SimpleNamespace(),
        workspace_path=SimpleNamespace(),
    )
    seen: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            seen["client"] = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        def create_completion(self, **kwargs: object) -> dict[str, object]:
            seen["completion"] = kwargs
            return {"choices": [{"message": {"content": " final answer "}}]}

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    response = call_openai_compatible(config, "workspace prompt")

    assert response.content == "final answer"
    assert seen["client"] == {
        "api_key": "ark",
        "base_url": "http://localhost:8000/v1",
        "timeout": 30,
    }
    assert seen["completion"] == {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "workspace prompt"},
        ],
    }


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
