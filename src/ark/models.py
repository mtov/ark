from __future__ import annotations

from dataclasses import dataclass
import json
import os

from .traces import record_response_usage


@dataclass
class ModelConfig:
    timeout_seconds: int
    openai_base_url: str | None
    openai_model: str | None
    openai_api_key_env: str | None


@dataclass
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class ModelResponse:
    content: str
    token_usage: TokenUsage


MAX_RESPONSE_DEBUG_CHARS = 2_000
MAX_EMPTY_RESPONSE_RETRIES = 1


class EmptyModelResponseError(ValueError):
    pass


def require_config_value(value: str | None, key: str, model_name: str) -> str:
    if not value:
        raise ValueError(f"Missing {key} for {model_name} model.")
    return value


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


def resolve_openai_api_key(config: ModelConfig) -> str:
    api_key_env = config.openai_api_key_env
    if api_key_env:
        return require_config_value(
            os.environ.get(api_key_env),
            api_key_env,
            "OpenAI-compatible",
        )

    return os.environ.get("OPENAI_API_KEY", "ark")


class Model:
    def __init__(self, config: ModelConfig, system_prompt: str) -> None:
        self.config = config
        self.system_prompt = system_prompt
        self._openai_client: object | None = None

    def call(self, user_prompt: str) -> ModelResponse:
        openai_model = require_config_value(
            self.config.openai_model,
            "openai_model",
            "OpenAI-compatible",
        )
        client = self._get_openai_client()

        for attempt in range(MAX_EMPTY_RESPONSE_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            except Exception as exc:  # noqa: BLE001
                base_url = self.config.openai_base_url or "default OpenAI endpoint"
                raise RuntimeError(
                    f"OpenAI-compatible request failed for {base_url}: "
                    f"{exc.__class__.__name__}: {exc}"
                ) from exc

            usage = extract_openai_usage(response)
            try:
                content = extract_openai_content(response)
            except EmptyModelResponseError:
                record_response_usage(usage)
                if attempt < MAX_EMPTY_RESPONSE_RETRIES:
                    continue
                raise

            model_response = build_model_response(content, usage)
            record_response_usage(model_response.token_usage)
            return model_response

        raise RuntimeError("Model retry loop ended unexpectedly.")

    def _get_openai_client(self) -> object:
        if self._openai_client is not None:
            return self._openai_client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Python package not installed. Run pip install -r requirements.txt."
            ) from exc

        self._openai_client = OpenAI(
            api_key=resolve_openai_api_key(self.config),
            base_url=self.config.openai_base_url,
            timeout=self.config.timeout_seconds,
        )
        return self._openai_client
