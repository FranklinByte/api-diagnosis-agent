"""Shared defensive helpers for secrets, bounded values, and network policy.

The diagnostic pipeline receives JSON-like data from failed integrations.  The
same data can later be placed in traces, validation errors, replay results, or
an LLM prompt, so redaction must be recursive and consistent at every boundary.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Mapping, Sequence
from typing import Any, Optional
from urllib.parse import urlsplit


REDACTED = "[REDACTED]"

SENSITIVE_NAME_MARKERS = (
    "authorization",
    "proxy_authorization",
    "cookie",
    "set_cookie",
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "token",
    "password",
    "credential",
    "private_key",
    "client_key",
)

SENSITIVE_HEADER_NAMES = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "x-auth-token",
    "x-access-token",
}

# These headers can make an otherwise allowlisted origin route or proxy a
# request using attacker-controlled authority information.
UNSAFE_FORWARD_HEADER_NAMES = {
    "host",
    "forwarded",
    "proxy",
    "proxy-connection",
    "x-forwarded-host",
    "x-forwarded-for",
    "x-forwarded-proto",
    "x-original-host",
    "x-original-url",
    "x-rewrite-url",
}

_SECRET_TEXT_RE = re.compile(
    r"(?i)(bearer\s+|basic\s+|(?:api[-_ ]?key|token|secret|password|credential)\s*[:=]\s*|(?=sk-[a-z0-9_-]{16,}))([^\s,;]+)"
)


def is_sensitive_name(name: Any) -> bool:
    """Return whether a field/header name should be redacted."""

    lowered = str(name).strip().lower().replace("-", "_")
    if lowered in {item.replace("-", "_") for item in SENSITIVE_HEADER_NAMES}:
        return True
    return any(marker in lowered for marker in SENSITIVE_NAME_MARKERS)


def is_unsafe_forward_header(name: Any) -> bool:
    """Return whether a request header must not override URL authority."""

    return str(name).strip().lower() in UNSAFE_FORWARD_HEADER_NAMES


def redact_text(value: Any, *, limit: int = 4096) -> str:
    """Redact common inline credentials and bound the resulting text."""

    text = str(value)
    text = _SECRET_TEXT_RE.sub(lambda match: f"{match.group(1)}{REDACTED}", text)
    return text[:limit]


def sanitize_value(
    value: Any,
    *,
    max_depth: int = 8,
    max_items: int = 100,
    max_string_length: int = 4096,
    replacement: str = REDACTED,
) -> Any:
    """Recursively redact JSON-like values without mutating the input.

    This function is intended for evidence, traces, and error details.  It is
    deliberately separate from request validation so redaction never changes
    the value being checked against an OpenAPI contract.
    """

    if max_depth < 0:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = _SECRET_TEXT_RE.sub(lambda match: f"{match.group(1)}{replacement}", value)
        return text[:max_string_length] + ("..." if len(text) > max_string_length else "")
    if isinstance(value, bytes):
        return f"<binary: {len(value)} bytes>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                result["[TRUNCATED_ITEMS]"] = True
                break
            name = str(key)
            result[name] = replacement if is_sensitive_name(name) else sanitize_value(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string_length=max_string_length,
                replacement=replacement,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [
            sanitize_value(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string_length=max_string_length,
                replacement=replacement,
            )
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            items.append("[TRUNCATED_ITEMS]")
        return items
    return redact_text(value, limit=max_string_length)


def sanitize_headers(headers: Optional[Mapping[str, Any]], replacement: str = REDACTED) -> dict[str, str]:
    """Return a bounded, secret-safe header mapping."""

    values = headers or {}
    result: dict[str, str] = {}
    for index, (key, value) in enumerate(values.items()):
        if index >= 100:
            result["[TRUNCATED_HEADERS]"] = replacement
            break
        name = str(key)[:256]
        result[name] = replacement if is_sensitive_name(name) else redact_text(value)
    return result


def sanitize_payload(value: Any) -> Any:
    """Stable public alias used before request data enters prompts or traces."""

    return sanitize_value(value)


def _default_port(scheme: str) -> int:
    return 443 if scheme.lower() == "https" else 80


def canonical_origin(url: str) -> Optional[tuple[str, str, int]]:
    """Return a normalized origin tuple, or ``None`` for malformed URLs."""

    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.scheme.lower() not in {"http", "https"} or not host:
            return None
        if parsed.username or parsed.password:
            return None
        port = parsed.port or _default_port(parsed.scheme)
    except ValueError:
        return None
    try:
        host = host.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError:
        return None
    return parsed.scheme.lower(), host, port


def address_is_private_or_reserved(address: str) -> bool:
    """Classify all non-public IP ranges conservatively for SSRF policy."""

    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return True
    return bool(
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


def resolve_host_addresses(host: str, port: int) -> set[str]:
    """Resolve a host once and return numeric addresses for policy checking."""

    results: set[str] = set()
    for family, _socktype, _proto, _canonname, sockaddr in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    ):
        if family in {socket.AF_INET, socket.AF_INET6}:
            results.add(str(sockaddr[0]))
    return results


def is_explicit_local_origin(origin: tuple[str, str, int]) -> bool:
    """Allow the local Mock target when it is explicitly allowlisted."""

    _scheme, host, _port = origin
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


__all__ = [
    "REDACTED",
    "SENSITIVE_HEADER_NAMES",
    "UNSAFE_FORWARD_HEADER_NAMES",
    "address_is_private_or_reserved",
    "canonical_origin",
    "is_explicit_local_origin",
    "is_sensitive_name",
    "is_unsafe_forward_header",
    "redact_text",
    "resolve_host_addresses",
    "sanitize_headers",
    "sanitize_payload",
    "sanitize_value",
]
