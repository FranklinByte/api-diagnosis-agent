"""Minimal OpenAI-compatible chat client used by the general diagnosis path.

Only ``httpx`` is required.  Credentials are read from the process
environment (optionally seeded from a local ``.env`` file) and are never
logged, returned, or placed in a prompt.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from .paths import project_root


class LLMUnavailable(RuntimeError):
    """The model is not configured, is rate limited locally, or failed."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float


def load_dotenv(path: Optional[Path] = None) -> None:
    """Seed missing environment variables from ``.env``; never override."""

    target = path or project_root() / ".env"
    if not target.is_file():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config() -> Optional[LLMConfig]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not api_key or not model:
        return None
    return LLMConfig(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        api_key=api_key,
        model=model,
        timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
    )


class _RateLimiter:
    """Process-wide cap so an exposed demo cannot burn the model budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[float] = []

    def acquire(self) -> None:
        limit = int(os.getenv("API_DIAGNOSIS_LLM_CALLS_PER_MINUTE", "30"))
        now = time.monotonic()
        with self._lock:
            self._calls = [stamp for stamp in self._calls if now - stamp < 60]
            if len(self._calls) >= limit:
                raise LLMUnavailable("local model call budget exceeded")
            self._calls.append(now)


_limiter = _RateLimiter()


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMUnavailable("model returned no JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMUnavailable("model returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise LLMUnavailable("model returned a non-object JSON value")
    return value


def chat_json(
    config: LLMConfig,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 6000,
    transport: Optional[httpx.BaseTransport] = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """One chat completion that must return a JSON object.

    Returns ``(parsed_object, usage)``.  Any transport, HTTP, or parsing
    problem is raised as :class:`LLMUnavailable` without echoing the key.
    """

    _limiter.acquire()
    body = {
        "model": config.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    try:
        with httpx.Client(timeout=config.timeout_seconds, transport=transport) as client:
            response = client.post(
                f"{config.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {config.api_key}"},
                json=body,
            )
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"model request failed: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise LLMUnavailable(f"model returned HTTP {response.status_code}")
    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailable("model response has an unexpected shape") from exc
    raw_usage = payload.get("usage") if isinstance(payload, dict) else None
    usage = {
        key: int(raw_usage[key])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(raw_usage, dict) and isinstance(raw_usage.get(key), int)
    }
    return _parse_json_object(content), usage


__all__ = ["LLMConfig", "LLMUnavailable", "chat_json", "load_config", "load_dotenv"]
