from __future__ import annotations

from types import SimpleNamespace

from ark.models import extract_openai_content, extract_openai_usage


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
