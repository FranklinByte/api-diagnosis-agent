"""Regression coverage for the October review, using no external services."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import api_diagnosis.agent as agent_module
import api_diagnosis.replay as replay_module
from api_diagnosis.api import app
from api_diagnosis.models import DiagnosisResult, Evidence, FaultCategory, ReplayResult


def valid_request():
    return {
        "request_id": uuid4().hex,
        "base_url": "http://127.0.0.1:8001",
        "method": "POST",
        "path": "/weather/forecast",
        "headers": {"X-API-Version": "v2"},
        "content_type": "application/json",
        "body": {"city": "Singapore", "units": "metric", "days": 3},
    }


def valid_response():
    return {
        "api_version": "v2", "city": "Singapore", "units": "metric", "days": 3,
        "forecast": [{"day": 1, "temperature": 24.0, "condition": "clear"}],
    }


def approve_replay(request):
    thread_id = uuid4().hex
    state = agent_module.run_agent_state(request, thread_id=thread_id, replay_requested=True)
    pending = agent_module.pending_approval(state)
    assert pending is not None
    resumed = agent_module.resume_agent_state(
        thread_id=thread_id, approval_token=pending["approval_token"], approved=True,
    )
    return resumed, thread_id, pending["approval_token"]


@pytest.mark.parametrize(("observation", "category", "status"), [
    (ReplayResult(status="http_error", status_code=500), "server_error", "confirmed"),
    (ReplayResult(status="http_error", status_code=429), "rate_limited", "confirmed"),
    (ReplayResult(status="timeout", error="timed out"), "timeout", "confirmed"),
    (ReplayResult(status="success", status_code=200, response_body={"unexpected": True}),
     "response_schema_change", "confirmed"),
    (ReplayResult(status="success", status_code=200, response_body=valid_response()),
     None, "confirmed"),
    (ReplayResult(status="blocked", error="Outside allowlist"), None, "unconfirmed"),
    (ReplayResult(status="network_error", error="Connection refused"), None, "unconfirmed"),
    (ReplayResult(status="response_too_large", status_code=200), None, "unconfirmed"),
])
def test_replay_observation_controls_final_diagnosis(monkeypatch, observation, category, status):
    monkeypatch.setattr(agent_module, "get_replayer", lambda: SimpleNamespace(replay=lambda _: observation))
    state, _, _ = approve_replay(valid_request())
    report = state["report"]
    assert report["fault_category"] == category
    assert report["status"] == status
    assert report["replay_result"]["status"] == observation.status
    if status == "unconfirmed":
        assert report["needs_manual"]
        assert f"replay_not_verified:{observation.status}" in report["limitations"]


def test_reused_thread_resets_previous_report_hypotheses_and_approval():
    client = TestClient(app)
    thread_id = uuid4().hex
    old = valid_request()
    old["headers"]["X-API-Version"] = "v1"
    first = client.post("/diagnose", json={"request": old, "thread_id": thread_id})
    assert first.json()["report"]["fault_category"] == "version_incompatible"
    new = valid_request()
    new["headers"]["Authorization"] = "Bearer regression-secret"
    second = client.post("/diagnose", json={
        "request": new, "thread_id": thread_id, "replay_requested": True,
    })
    assert second.status_code == 200
    payload = second.json()
    assert payload["report"]["request_id"] == new["request_id"]
    assert payload["report"]["status"] == "unconfirmed"
    assert payload["report"]["fault_category"] is None
    assert payload["pending_approval"]
    snapshot = agent_module._GRAPH.get_state({"configurable": {"thread_id": thread_id}})
    assert "regression-secret" not in json.dumps(snapshot.values, default=str)
    assert not snapshot.values["report"]
    assert all(h["category"] != "version_incompatible" for h in snapshot.values["hypotheses"])
    resumed = client.post("/diagnose/resume", json={
        "thread_id": thread_id,
        "approval_token": payload["pending_approval"]["approval_token"], "approved": False,
    })
    assert resumed.status_code == 200
    third = client.post("/diagnose", json={"request": valid_request(), "thread_id": thread_id})
    report = third.json()["report"]
    assert report["execution_metadata"]["tool_history"]
    assert "replay_rejected_by_user" not in report["limitations"]


def test_replay_budget_is_per_task_and_approval_cannot_be_reused(monkeypatch):
    calls = []

    class Response:
        status = 200
        headers = {"Content-Type": "application/json"}

        def read(self, maximum):
            return json.dumps(valid_response()).encode()[:maximum]

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            return Response()

    monkeypatch.setenv("API_DIAGNOSIS_REPLAY_ALLOWLIST", "http://127.0.0.1:8001")
    monkeypatch.setattr(replay_module.urllib.request, "build_opener", lambda *args: Opener())
    for _ in range(3):
        state, thread_id, token = approve_replay(valid_request())
        assert state["report"]["replay_result"]["status"] == "success"
        assert state["report"]["replay_result"]["request_count"] == 1
        with pytest.raises(ValueError, match="no pending approval"):
            agent_module.resume_agent_state(thread_id=thread_id, approval_token=token, approved=True)
    assert len(calls) == 3


def test_matching_label_cannot_bypass_guardrail_with_fabricated_report():
    request = valid_request()
    request["headers"]["X-API-Version"] = "v1"
    authoritative = agent_module.run_agent(request)
    fabricated = authoritative.model_copy(update={
        "request_id": "wrong-request", "root_cause": "Invented database corruption",
        "repair_suggestion": "Delete all tables", "corrected_request": None,
        "evidence": [Evidence(source="invented")], "hypotheses": [],
        "verification_steps": [], "revalidation_steps": ["Disable all validation"],
        "needs_manual": not authoritative.needs_manual,
        "limitations": ["provider-specific limitation"],
    })
    guarded = agent_module.apply_deterministic_guardrail(fabricated, authoritative)
    for field in set(DiagnosisResult.model_fields) - {"limitations", "execution_metadata"}:
        assert getattr(guarded, field) == getattr(authoritative, field)
    assert "provider-specific limitation" in guarded.limitations
    assert "deterministic_guardrail_preserved_authoritative_fields" in guarded.limitations


class BrokenProvider:
    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        raise TimeoutError("regression-secret-provider-token")


def test_provider_failure_returns_annotated_fallback_not_client_error(monkeypatch):
    monkeypatch.setenv("API_DIAGNOSIS_ENABLE_PROVIDER", "1")
    monkeypatch.setenv("OPENAI_STRUCTURED_OUTPUT", "1")
    monkeypatch.setenv("OPENAI_JSON_FALLBACK_RETRY", "0")
    monkeypatch.setattr(agent_module, "provider_chat_model", BrokenProvider)
    request = valid_request()
    request["headers"]["X-API-Version"] = "v1"
    response = TestClient(app).post("/diagnose", json={"request": request, "mode": "provider"})
    assert response.status_code == 200
    assert "regression-secret-provider-token" not in response.text
    report = response.json()["report"]
    assert report["fault_category"] == "version_incompatible"
    assert report["execution_metadata"]["synthesis_source"] == "deterministic_fallback"
    assert not report["execution_metadata"]["provider_tool_call_used"]
    assert "provider_synthesis_failed_using_deterministic_report" in report["limitations"]


@pytest.mark.parametrize("guardrail", [True, False])
def test_invalid_provider_json_falls_back_with_provenance(monkeypatch, guardrail):
    class InvalidJSON(BrokenProvider):
        def invoke(self, prompt):
            return SimpleNamespace(content="not valid JSON", tool_calls=[])

    monkeypatch.setenv("OPENAI_STRUCTURED_OUTPUT", "0")
    report = agent_module.run_provider_agent(valid_request(), model=InvalidJSON(), guardrail=guardrail)
    assert report.execution_metadata["synthesis_source"] == "deterministic_fallback"
    assert report.execution_metadata["synthesis_attempts"] == 1


@pytest.mark.parametrize("mode", ["provider", "deterministic"])
def test_unsupported_replay_mode_rejected_before_provider_call(monkeypatch, mode):
    def unexpected_provider():
        pytest.fail("unsupported replay combination must be rejected before model initialization")

    monkeypatch.setattr(agent_module, "provider_chat_model", unexpected_provider)
    response = TestClient(app).post("/diagnose", json={
        "request": valid_request(), "mode": mode, "replay_requested": True,
    })
    assert response.status_code == 422
    assert "replay_requested requires mode=agent" in response.text


def test_successful_provider_synthesis_is_recorded_and_guarded(monkeypatch):
    class Provider(BrokenProvider):
        def bind_tools(self, tools):
            return SimpleNamespace(invoke=lambda _: SimpleNamespace(
                tool_calls=[{"name": "check_api_version", "args": {}}],
            ))

        def invoke(self, prompt):
            return DiagnosisResult(
                status="confirmed", fault_category=FaultCategory.VERSION_INCOMPATIBLE,
                root_cause="Invented cause with the right label",
            )

    monkeypatch.setenv("OPENAI_STRUCTURED_OUTPUT", "1")
    request = valid_request()
    request["headers"]["X-API-Version"] = "v1"
    report = agent_module.run_provider_agent(request, model=Provider())
    assert report.execution_metadata["synthesis_source"] == "provider"
    assert report.execution_metadata["provider_tool_call_used"]
    assert report.request_id == request["request_id"]
    assert report.root_cause != "Invented cause with the right label"
    assert report.evidence


def test_evaluation_counts_deterministic_fallback_separately(monkeypatch):
    run_mode = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_real_eval.py"))["_run_mode"]
    monkeypatch.setitem(run_mode.__globals__, "run_provider_agent", lambda *args, **kwargs: DiagnosisResult(
        status="confirmed", fault_category=FaultCategory.SERVER_ERROR,
        execution_metadata={"synthesis_source": "deterministic_fallback"},
    ))
    result = run_mode("agent", [{
        "id": "fallback", "request": valid_request(), "expected_status": "confirmed",
        "expected_category": "server_error",
    }], model=object(), usage_handler=SimpleNamespace(usage_metadata={}))
    assert result["category_accuracy"] == 1.0
    assert result["deterministic_fallback_count"] == 1
    assert result["deterministic_fallback_rate"] == 1.0
    assert result["accuracy_scope"] == "system_including_deterministic_fallback"
    assert result["cases"][0]["deterministic_fallback"]
