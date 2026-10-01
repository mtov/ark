from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
import os
from typing import Any

from .traces import record_response_usage


MAX_RESPONSE_DEBUG_CHARS = 2_000
MAX_EMPTY_RESPONSE_RETRIES = 1


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    content: str
    token_usage: TokenUsage


class EmptyModelResponseError(ValueError):
    pass


def build_model_response(content: str, token_usage: TokenUsage | None = None) -> ModelResponse:
    return ModelResponse(content=content, token_usage=token_usage or TokenUsage())


def _response_field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _serialize_response_for_error(response: object) -> str:
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(exclude_none=True)
    elif isinstance(response, dict):
        payload = response
    elif hasattr(response, "__dict__"):
        payload = vars(response)
    else:
        payload = repr(response)

    try:
        serialized = json.dumps(payload, default=str, ensure_ascii=True)
    except (TypeError, ValueError):
        serialized = repr(payload)

    if len(serialized) > MAX_RESPONSE_DEBUG_CHARS:
        return f"{serialized[:MAX_RESPONSE_DEBUG_CHARS]}... (truncated)"
    return serialized


def extract_openai_content(response: object) -> str:
    choices = _response_field(response, "choices")

    if not choices:
        raise ValueError("OpenAI-compatible model returned a response without choices.")

    first_choice = choices[0]
    message = _response_field(first_choice, "message") or {}

    content = _response_field(message, "content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
                continue
            text = getattr(part, "text", None)
            if text:
                text_parts.append(str(text))
        joined = "".join(text_parts).strip()
        if joined:
            return joined

    finish_reason = _response_field(first_choice, "finish_reason")
    refusal = _response_field(message, "refusal")
    serialized_response = _serialize_response_for_error(response)
    error_message = (
        "OpenAI-compatible model returned a response without message content. "
        f"finish_reason={finish_reason!r}; refusal={refusal!r}; "
        f"response={serialized_response}"
    )
    if refusal is None:
        raise EmptyModelResponseError(error_message)
    raise ValueError(error_message)


def extract_openai_usage(response: object) -> TokenUsage:
    usage = _response_field(response, "usage")
    if usage is None:
        return TokenUsage()

    prompt_tokens = _response_field(usage, "prompt_tokens")
    output_tokens = _response_field(usage, "completion_tokens")
    total_tokens = _response_field(usage, "total_tokens")

    if total_tokens is None and (prompt_tokens is not None or output_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (output_tokens or 0)

    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


@dataclass(frozen=True, kw_only=True)
class Model:
    name: str
    system_prompt: str
    timeout_seconds: int
    base_url: str | None = None
    api_key_env: str | None = None

    def call(self, user_prompt: str) -> ModelResponse:
        for _ in range(MAX_EMPTY_RESPONSE_RETRIES):
            try:
                return self._parse_response(self._request(user_prompt))
            except EmptyModelResponseError:
                pass

        return self._parse_response(self._request(user_prompt))

    def _parse_response(self, response: object) -> ModelResponse:
        usage = extract_openai_usage(response)
        record_response_usage(usage)
        return build_model_response(extract_openai_content(response), usage)

    def _request(self, user_prompt: str) -> object:
        try:
            return self._client.chat.completions.create(
                model=self.name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            endpoint = self.base_url or "default OpenAI endpoint"
            raise RuntimeError(
                f"OpenAI-compatible request failed for {endpoint}: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc

    @cached_property
    def _client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Python package not installed. Run pip install -r requirements.txt."
            ) from exc

        return OpenAI(
            api_key=self._api_key(),
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )

    def _api_key(self) -> str:
        if self.api_key_env is None:
            return os.environ.get("OPENAI_API_KEY", "ark")

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing {self.api_key_env} for OpenAI-compatible model."
            )
        return api_key
