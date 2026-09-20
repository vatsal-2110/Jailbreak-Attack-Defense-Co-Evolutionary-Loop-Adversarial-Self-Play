"""Thin OpenRouter/OpenAI-compatible client with explicit failure modes.

The original notebook lost every judge label to this line::

    raw = result.choices[0].message.content.strip()   # content was None

Reasoning-tuned and classifier-tuned models routinely return ``content=None``
(the text lands in a vendor ``reasoning`` field, or the model declines the
requested format). Silently swallowing that turned a 100%-broken judge into a
reported "0% attack success rate". This client raises instead.
"""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from .utils import get_logger

LOGGER = get_logger(__name__)


class EmptyCompletionError(RuntimeError):
    """The provider returned a response with no usable text content."""


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        default_retries: int = 4,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._default_retries = default_retries

    def complete(
        self,
        model: str,
        user_prompt: str,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        retries: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Return the completion text, or raise after exhausting retries.

        Raises
        ------
        EmptyCompletionError
            Every attempt produced an empty/None content field.
        Exception
            The last transport/API error, re-raised.
        """
        retries = self._default_retries if retries is None else retries
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                text = _extract_text(response)
                if text:
                    return text
                last_error = EmptyCompletionError(
                    f"{model} returned empty content "
                    f"(finish_reason={_finish_reason(response)}). "
                    f"If this is a reasoning model, raise max_tokens; if it is "
                    f"a classifier model, it may be rejecting the requested "
                    f"output format."
                )
                LOGGER.warning(
                    "Empty completion from %s (attempt %d/%d)",
                    model,
                    attempt + 1,
                    retries,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced after retries
                last_error = exc
                LOGGER.warning(
                    "API error from %s (attempt %d/%d): %s",
                    model,
                    attempt + 1,
                    retries,
                    exc,
                )
            if attempt < retries - 1:
                time.sleep(2**attempt)

        assert last_error is not None
        raise last_error


def _extract_text(response: Any) -> str:
    if not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    # Some providers place the visible answer in a list of content parts.
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        joined = "".join(parts).strip()
        if joined:
            return joined
    return ""


def _finish_reason(response: Any) -> str:
    try:
        return str(response.choices[0].finish_reason)
    except Exception:  # noqa: BLE001
        return "unknown"
