"""Safe, bounded HTTP replay for the local V0 Mock API."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Optional
from urllib.parse import urlencode, urlsplit, urlunsplit

from .models import ReplayPolicy, ReplayRequest, ReplayResult
from .security import (
    address_is_private_or_reserved,
    canonical_origin,
    is_explicit_local_origin,
    is_sensitive_name,
    is_unsafe_forward_header,
    redact_text,
    resolve_host_addresses,
    sanitize_headers,
    sanitize_value,
)


def _forward_headers(headers: Mapping[str, Any], policy: ReplayPolicy) -> dict[str, str]:
    if policy.forward_sensitive_headers:
        # This is opt-in for synthetic Mock credentials only.  The public
        # result still contains the redacted representation.
        return {
            str(key): str(value)
            for key, value in headers.items()
            if not is_unsafe_forward_header(key)
        }
    return {
        str(key): str(value)
        for key, value in headers.items()
        if not is_sensitive_name(key) and not is_unsafe_forward_header(key)
    }


def _encode_body(body: Any) -> bytes:
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    if isinstance(body, str):
        return body.encode("utf-8")
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _append_query(url: str, params: Mapping[str, Any]) -> str:
    if not params:
        return url
    parsed = urlsplit(url)
    pairs: list[tuple[str, str]] = []
    for key, value in params.items():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            pairs.extend((str(key), str(item)) for item in value)
        else:
            pairs.append((str(key), str(value)))
    query = parsed.query
    encoded = urlencode(pairs, doseq=True)
    query = f"{query}&{encoded}" if query else encoded
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))


def _validate_url(url: str, allowed_base_urls: Sequence[str]) -> Optional[str]:
    if any(ord(char) < 32 for char in url):
        return "Control characters are not allowed in replay URLs."
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "Malformed replay URL."
    target_origin = canonical_origin(url)
    if target_origin is None or not parsed.netloc:
        return "Only absolute HTTP(S) URLs are accepted."
    # Reject traversal before the request is handed to urllib.  Encoded dot
    # segments are rejected conservatively as well.
    if (
        any(part == ".." for part in parsed.path.split("/"))
        or "%2e" in parsed.path.lower()
        or "\\" in parsed.path
    ):
        return "Path traversal is not allowed."
    matching_base = False
    for base in allowed_base_urls:
        base_text = str(base)
        base_origin = canonical_origin(base_text)
        if base_origin is None:
            continue
        base_parsed = urlsplit(base_text)
        base_path = base_parsed.path.rstrip("/") or "/"
        same_origin = target_origin == base_origin
        if same_origin and (
            base_path == "/"
            or parsed.path == base_path
            or parsed.path.startswith(base_path + "/")
        ):
            matching_base = True
            break
    if not matching_base:
        return "URL is outside the configured allowlist."
    # Resolve only after the exact allowlist check, avoiding DNS probes for
    # arbitrary attacker-controlled hosts.  Check every returned address so a
    # DNS name cannot mix a public answer with an internal one.
    try:
        target_addresses = resolve_host_addresses(target_origin[1], target_origin[2])
    except (OSError, socket.gaierror):
        return "The replay host could not be resolved."
    if not target_addresses:
        return "The replay host did not resolve to an address."
    target_is_local = is_explicit_local_origin(target_origin)
    if any(address_is_private_or_reserved(address) for address in target_addresses) and not target_is_local:
        return "Private, loopback, link-local, multicast, or reserved destinations are blocked."
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):  # noqa: D401
        return None


class MockRequestReplayer:
    """A synchronous, allowlisted, request-counted HTTP replayer."""

    def __init__(
        self,
        policy: Optional[ReplayPolicy] = None,
        *,
        allowed_base_urls: Optional[Sequence[str]] = None,
        allowed_methods: Optional[Sequence[str]] = None,
        opener: Any = None,
    ):
        if policy is None:
            policy = ReplayPolicy(
                allowed_urls=list(allowed_base_urls or []),
                allowed_methods=set(allowed_methods or {"GET", "HEAD"}),
            )
        elif allowed_base_urls is not None or allowed_methods is not None:
            data = policy.model_dump()
            if allowed_base_urls is not None:
                data["allowed_base_urls"] = list(allowed_base_urls)
            if allowed_methods is not None:
                data["allowed_methods"] = set(allowed_methods)
            policy = ReplayPolicy.model_validate(data)
        self.policy = policy
        self._count = 0
        self._lock = threading.Lock()
        # Never inherit HTTP(S)_PROXY from the host process.  A proxy can turn
        # an allowlisted local request into an unintended external disclosure.
        self._opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
        )

    @property
    def request_count(self) -> int:
        return self._count

    def _reserve(self) -> Optional[int]:
        with self._lock:
            if self._count >= self.policy.max_requests:
                return None
            self._count += 1
            return self._count

    def replay(self, request: ReplayRequest | Mapping[str, Any]) -> ReplayResult:
        model = request if isinstance(request, ReplayRequest) else ReplayRequest.model_validate(request)
        safe_headers = sanitize_headers(model.headers)
        url = _append_query(model.url, model.query_params)
        url_error = _validate_url(url, self.policy.allowed_base_urls)
        if url_error:
            return ReplayResult(
                status="blocked",
                error=url_error,
                sanitized_request_headers=safe_headers,
                request_count=self._count,
            )
        target_origin = canonical_origin(url)
        if self.policy.forward_sensitive_headers and (
            target_origin is None or not is_explicit_local_origin(target_origin)
        ):
            return ReplayResult(
                status="blocked",
                error="Sensitive header forwarding is restricted to an explicit local Mock origin.",
                sanitized_request_headers=safe_headers,
                request_count=self._count,
            )
        if model.method not in self.policy.allowed_methods:
            return ReplayResult(
                status="blocked",
                error=f"HTTP method {model.method} is not allowlisted.",
                sanitized_request_headers=safe_headers,
                request_count=self._count,
            )
        if (
            self.policy.require_approval_for_non_idempotent
            and model.method not in {"GET", "HEAD", "OPTIONS"}
            and not model.approved
        ):
            return ReplayResult(
                status="blocked",
                error="Manual approval is required before a non-idempotent replay.",
                sanitized_request_headers=safe_headers,
                request_count=self._count,
            )
        body = _encode_body(model.body)
        if len(body) > self.policy.max_request_bytes:
            return ReplayResult(
                status="invalid_request",
                error="Request body exceeds the configured size limit.",
                sanitized_request_headers=safe_headers,
                request_count=self._count,
            )
        number = self._reserve()
        if number is None:
            return ReplayResult(
                status="blocked",
                error="Replay request limit has been reached.",
                sanitized_request_headers=safe_headers,
                request_count=self._count,
            )

        headers = _forward_headers(model.headers, self.policy)
        if body and "content-type" not in {name.lower() for name in headers}:
            headers["Content-Type"] = "application/json"
        request_object = urllib.request.Request(
            url=url,
            data=body or None,
            headers=headers,
            method=model.method,
        )
        started = time.monotonic()
        timeout = min(model.timeout_seconds or self.policy.timeout_seconds, self.policy.timeout_seconds)
        try:
            response = self._opener.open(request_object, timeout=timeout)
            try:
                return self._read_response(response, started, number, safe_headers, timeout_seconds=timeout)
            finally:
                _close_response(response)
        except urllib.error.HTTPError as exc:
            # HTTP errors are still valid observations and should be returned
            # as structured replay results rather than raised.
            try:
                return self._read_response(
                    exc,
                    started,
                    number,
                    safe_headers,
                    timeout_seconds=timeout,
                    http_error=True,
                )
            finally:
                _close_response(exc)
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            message = str(getattr(exc, "reason", exc))
            message = redact_text(message)
            is_timeout = isinstance(exc, TimeoutError) or "timed out" in message.lower()
            return ReplayResult(
                status="timeout" if is_timeout else "network_error",
                elapsed_ms=(time.monotonic() - started) * 1000,
                error=message,
                trace_id=uuid.uuid4().hex,
                request_count=number,
                sanitized_request_headers=safe_headers,
            )

    def _read_response(
        self,
        response: Any,
        started: float,
        number: int,
        safe_headers: Mapping[str, str],
        *,
        timeout_seconds: float,
        http_error: bool = False,
    ) -> ReplayResult:
        headers = dict(response.headers.items()) if getattr(response, "headers", None) else {}
        safe_response_headers = sanitize_headers(headers)
        deadline = started + timeout_seconds
        maximum = self.policy.max_response_bytes + 1
        chunks: list[bytes] = []
        total = 0
        # Test doubles and file-like adapters without an underlying socket
        # conventionally return the complete payload for one read.  Avoid
        # repeatedly asking such an object for the same bytes.
        if not getattr(response, "fp", None):
            raw = bytes(response.read(maximum))
            total = len(raw)
            chunks = [raw]
        else:
            raw = b""
        while getattr(response, "fp", None) and total < maximum:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return ReplayResult(
                    status="timeout",
                    response_headers=safe_response_headers,
                    elapsed_ms=(time.monotonic() - started) * 1000,
                    trace_id=uuid.uuid4().hex,
                    error="Response read exceeded the configured timeout budget.",
                    request_count=number,
                    sanitized_request_headers=dict(safe_headers),
                )
            _set_response_socket_timeout(response, remaining)
            chunk = response.read(min(65_536, maximum - total))
            if not chunk:
                break
            chunks.append(bytes(chunk))
            total += len(chunk)
        if chunks and not raw:
            raw = b"".join(chunks)
        status_code = int(getattr(response, "status", getattr(response, "code", 0)) or 0)
        trace_id = next(
            (value for key, value in headers.items() if key.lower() == "x-trace-id"),
            None,
        ) or uuid.uuid4().hex
        elapsed = (time.monotonic() - started) * 1000
        if len(raw) > self.policy.max_response_bytes:
            return ReplayResult(
                status="response_too_large",
                status_code=status_code or None,
                response_headers=safe_response_headers,
                response_size_bytes=len(raw),
                elapsed_ms=elapsed,
                trace_id=trace_id,
                error="Response exceeds the configured size limit.",
                request_count=number,
                sanitized_request_headers=dict(safe_headers),
            )
        try:
            decoded = raw.decode("utf-8")
            try:
                body: Any = sanitize_value(json.loads(decoded))
            except json.JSONDecodeError:
                body = redact_text(decoded)
        except UnicodeDecodeError:
            body = f"<binary response: {len(raw)} bytes>"
        return ReplayResult(
            status="http_error" if http_error or status_code >= 400 else "success",
            status_code=status_code or None,
            response_headers=safe_response_headers,
            response_body=body,
            response_size_bytes=len(raw),
            elapsed_ms=elapsed,
            trace_id=trace_id,
            request_count=number,
            sanitized_request_headers=dict(safe_headers),
        )


def _set_response_socket_timeout(response: Any, timeout_seconds: float) -> None:
    """Apply the remaining total budget to urllib's underlying socket."""

    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is None:
        return
    try:
        sock.settimeout(max(0.001, timeout_seconds))
    except (AttributeError, OSError):
        return


def _close_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except OSError:
            return


def replay_request(
    request: ReplayRequest | Mapping[str, Any],
    *,
    policy: Optional[ReplayPolicy] = None,
    **policy_overrides: Any,
) -> ReplayResult:
    """One-shot convenience wrapper around :class:`MockRequestReplayer`."""

    if policy_overrides:
        data = policy.model_dump() if policy else {}
        data.update(policy_overrides)
        policy = ReplayPolicy.model_validate(data)
    return MockRequestReplayer(policy).replay(request)


__all__ = [
    "MockRequestReplayer",
    "ReplayPolicy",
    "ReplayRequest",
    "ReplayResult",
    "replay_request",
    "sanitize_headers",
]
