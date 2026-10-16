from __future__ import annotations

import time

import api_diagnosis.replay as replay_module
from api_diagnosis.models import ReplayPolicy, ReplayRequest
from api_diagnosis.replay import MockRequestReplayer
from api_diagnosis.resilience import RetryPolicy, call_with_fallback
from api_diagnosis.security import sanitize_headers, sanitize_payload


def _blocking_operation():
    time.sleep(5)
    return "late"


def test_recursive_payload_and_header_redaction_covers_provider_key_shapes():
    payload = {
        "Authorization": "Bearer live-secret",
        "nested": {
            "client_secret": "secret-value",
            "private_key": "pem-value",
            "message": "sk-1234567890123456",
        },
        "items": [{"password": "pw", "safe": "hello"}],
    }
    redacted = sanitize_payload(payload)
    assert redacted["Authorization"] == "[REDACTED]"
    assert redacted["nested"]["client_secret"] == "[REDACTED]"
    assert redacted["nested"]["private_key"] == "[REDACTED]"
    assert redacted["nested"]["message"] == "[REDACTED]"
    assert redacted["items"][0]["password"] == "[REDACTED]"
    assert redacted["items"][0]["safe"] == "hello"
    assert sanitize_headers({"X-Api-Key": "secret", "X-Trace": "ok"}) == {
        "X-Api-Key": "[REDACTED]",
        "X-Trace": "ok",
    }


def test_replay_checks_allowlist_before_dns_and_blocks_private_resolution(monkeypatch):
    def should_not_resolve(*_args):
        raise AssertionError("outside-allowlist hosts must not be resolved")

    monkeypatch.setattr(replay_module, "resolve_host_addresses", should_not_resolve)
    policy = ReplayPolicy(allowed_urls=["https://api.example"], allowed_methods={"GET"})
    outside = MockRequestReplayer(policy).replay(
        ReplayRequest(url="https://attacker.example/", method="GET")
    )
    assert outside.status == "blocked"
    assert "allowlist" in (outside.error or "")

    monkeypatch.setattr(replay_module, "resolve_host_addresses", lambda *_args: {"10.0.0.5"})
    private = MockRequestReplayer(policy).replay(
        ReplayRequest(url="https://api.example/health", method="GET")
    )
    assert private.status == "blocked"
    assert "Private" in (private.error or "")


def test_explicit_loopback_mock_remains_available_and_authority_headers_are_removed():
    class Response:
        status = 200
        headers = {"content-type": "application/json"}
        fp = None

        def read(self, _limit=-1):
            return b'{"ok": true}'

        def close(self):
            return None

    class Opener:
        def open(self, incoming, timeout):
            assert incoming.headers.get("Host") is None
            assert incoming.headers.get("X-forwarded-host") is None
            return Response()

    policy = ReplayPolicy(allowed_urls=["http://127.0.0.1:8000"], allowed_methods={"POST"})
    result = MockRequestReplayer(policy, opener=Opener()).replay(
        ReplayRequest(
            url="http://127.0.0.1:8000/weather/forecast",
            method="POST",
            headers={"Host": "internal.example", "X-Forwarded-Host": "internal.example"},
            body={"city": "Singapore"},
            approved=True,
        )
    )
    assert result.status == "success"


def test_sensitive_header_forwarding_cannot_be_enabled_for_public_origins(monkeypatch):
    # Exercise the credential-forwarding guard independently of the host's DNS.
    # A reserved example hostname need not resolve on a clean CI runner.
    monkeypatch.setattr(replay_module, "resolve_host_addresses", lambda *_args: {"93.184.216.34"})

    class RejectNetwork:
        def open(self, *_args, **_kwargs):
            raise AssertionError("public-origin credentials must be blocked before any HTTP request")

    policy = ReplayPolicy(
        allowed_urls=["https://api.example"],
        allowed_methods={"GET"},
        forward_sensitive_headers=True,
    )
    result = MockRequestReplayer(policy, opener=RejectNetwork()).replay(
        ReplayRequest(
            url="https://api.example/health",
            method="GET",
            headers={"Authorization": "Bearer secret"},
        )
    )
    assert result.status == "blocked"
    assert "local Mock" in (result.error or "")
    assert result.request_count == 0
    assert result.sanitized_request_headers["Authorization"] == "[REDACTED]"


def test_picklable_blocking_operation_is_terminated_at_total_deadline():
    started = time.perf_counter()
    outcome = call_with_fallback(
        _blocking_operation,
        policy=RetryPolicy(max_attempts=1, timeout_seconds=0.25, backoff_seconds=0),
    )
    elapsed = time.perf_counter() - started
    assert outcome.value is None
    assert "timeout" in (outcome.error or "").lower()
    assert elapsed < 2.0


def test_retry_backoff_and_attempts_share_one_total_budget():
    outcome = call_with_fallback(
        _blocking_operation,
        policy=RetryPolicy(max_attempts=4, timeout_seconds=0.3, backoff_seconds=0.2),
    )
    assert outcome.value is None
    assert outcome.attempts <= 4
    assert outcome.elapsed_ms < 1500
