from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

from .traces import trace_response

if TYPE_CHECKING:
    from .inputs import AgentConfig


@dataclass
class ModelConfig:
    model: str
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


def require_config_value(value: str | None, key: str, model_name: str) -> str:
    if not value:
        raise ValueError(f"Missing {key} for {model_name} model.")
    return value


def build_model_response(content: str, token_usage: TokenUsage | None = None) -> ModelResponse:
    return ModelResponse(content=content, token_usage=token_usage or TokenUsage())


def extract_openai_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")

    if not choices:
        raise ValueError("OpenAI-compatible model returned a response without choices.")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None and isinstance(first_choice, dict):
        message = first_choice.get("message", {})

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

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

    raise ValueError("OpenAI-compatible model returned a response without message content.")


def extract_openai_usage(response: object) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return TokenUsage()

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
        output_tokens = usage.get("completion_tokens", output_tokens)
        total_tokens = usage.get("total_tokens", total_tokens)

    if total_tokens is None and (prompt_tokens is not None or output_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (output_tokens or 0)

    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def resolve_openai_api_key(config: AgentConfig) -> str:
    api_key_env = config.model_config.openai_api_key_env
    if api_key_env:
        return require_config_value(
            os.environ.get(api_key_env),
            api_key_env,
            "OpenAI-compatible",
        )

    return os.environ.get("OPENAI_API_KEY", "ark")


def call_openai_compatible(config: AgentConfig, user_prompt: str) -> ModelResponse:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI Python package not installed. Run pip install -r requirements.txt.") from exc

    openai_model = require_config_value(
        config.model_config.openai_model,
        "openai_model",
        "OpenAI-compatible",
    )
    client = OpenAI(
        api_key=resolve_openai_api_key(config),
        base_url=config.model_config.openai_base_url,
    )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client.chat.completions.create,
                model=openai_model,
                messages=[
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            response = future.result(timeout=config.model_config.timeout_seconds)
    except FuturesTimeoutError as exc:
        raise RuntimeError(
            f"OpenAI-compatible request timed out after {config.model_config.timeout_seconds} seconds."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        base_url = config.model_config.openai_base_url or "default OpenAI endpoint"
        raise RuntimeError(
            f"OpenAI-compatible request failed for {base_url}: {exc.__class__.__name__}: {exc}"
        ) from exc

    return build_model_response(
        extract_openai_content(response),
        extract_openai_usage(response),
    )


def call_model(
    config: AgentConfig,
    user_prompt: str,
    response_label: str = "Response",
) -> ModelResponse:
    model = config.model_config.model

    if model == "openai-compatible":
        response = call_openai_compatible(config, user_prompt)
    else:
        raise ValueError(f"Unsupported model: {model}")

    trace_response(response.content, response_label, response.token_usage)
    return response
