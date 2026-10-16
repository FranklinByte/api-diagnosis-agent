from __future__ import annotations

from api_diagnosis.checkpoint import ApprovalGate, new_memory_saver
from api_diagnosis.resilience import RetryPolicy, call_with_fallback


def test_hitl_gate_and_langgraph_memory_checkpoint():
    gate = ApprovalGate()
    token = gate.request("replay", details={"method": "POST"})
    assert not gate.is_approved(token)
    assert gate.approve(token)
    assert gate.is_approved(token)
    assert new_memory_saver() is not None


def test_retry_and_fallback_are_bounded():
    calls = {"count": 0}

    def operation():
        calls["count"] += 1
        raise RuntimeError("offline")

    outcome = call_with_fallback(operation, fallback=lambda exc: {"fallback": str(exc)}, policy=RetryPolicy(max_attempts=2, backoff_seconds=0))
    assert outcome.used_fallback
    assert outcome.value == {"fallback": "offline"}
    assert calls["count"] == 2
