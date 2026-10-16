from __future__ import annotations

from fastapi.testclient import TestClient

from api_diagnosis.api import app
from api_diagnosis.evaluator import load_cases, run_benchmark


def test_diagnosis_http_boundary_returns_trace():
    client = TestClient(app)
    response = client.post(
        "/diagnose",
        json={
            "request": {
                "request_id": "http-001",
                "method": "POST",
                "path": "/weather/forecast",
                "headers": {"X-API-Version": "v2"},
                "body": {"city": "Singapore", "units": "metric"},
            }
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["report"]["fault_category"] == "missing_required_field"
    assert payload["trace"]["trace_id"]
    assert payload["trace"]["events"][0]["name"] == "diagnose"
    nodes = [event["node"] for event in payload["trace"]["agent_events"]]
    assert nodes == [
        "validate_input",
        "retrieve_docs",
        "generate_hypotheses",
        "select_tool",
        "execute_tool",
        "judge_sufficiency",
        "finalize_report",
    ]
    assert payload["report"]["hypotheses"]
    assert payload["thread_id"] == "http-001"


def test_http_replay_requires_token_and_can_be_rejected():
    client = TestClient(app)
    started = client.post(
        "/diagnose",
        json={
            "thread_id": "http-approval-001",
            "replay_requested": True,
            "request": {
                "request_id": "http-approval-001",
                "base_url": "http://127.0.0.1:8001",
                "method": "POST",
                "path": "/weather/forecast",
                "headers": {"X-API-Version": "v2"},
                "body": {"city": "Singapore", "units": "metric", "days": 3},
            },
        },
    )
    assert started.status_code == 200
    pending = started.json()["pending_approval"]
    assert pending["approval_token"]
    invalid = client.post(
        "/diagnose/resume",
        json={
            "thread_id": "http-approval-001",
            "approval_token": "x" * 24,
            "approved": False,
        },
    )
    assert invalid.status_code == 400
    resumed = client.post(
        "/diagnose/resume",
        json={
            "thread_id": "http-approval-001",
            "approval_token": pending["approval_token"],
            "approved": False,
        },
    )
    assert resumed.status_code == 200
    assert resumed.json()["report"]["needs_manual"] is True
    assert "replay_rejected_by_user" in resumed.json()["report"]["limitations"]


def test_offline_benchmark_has_all_requested_baselines():
    cases = load_cases()
    assert 40 <= len(cases) <= 60
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({str(case["request"]) for case in cases}) == len(cases)
    result = run_benchmark()
    assert set(result) == {
        "http_signal_heuristic",
        "retrieval_context_heuristic",
        "langgraph_oracle_wrapper",
        "deterministic_oracle_reference",
    }
    for name, metrics in result.items():
        assert 40 <= metrics["total"] <= 60
        assert metrics["unique_input_count"] == metrics["total"]
        assert 0.0 <= metrics["category_accuracy"] <= 1.0
        assert 0.0 <= metrics["category_macro_accuracy"] <= 1.0
        assert 0.0 <= metrics["tool_call_rate"] <= 1.0
        assert 0.0 <= metrics["tool_selection_accuracy"] <= 1.0
        assert 0.0 <= metrics["citation_coverage"] <= 1.0
        assert 0.0 <= metrics["oracle_derived_rate"] <= 1.0
        assert 0.0 <= metrics["oracle_fallback_rate"] <= 1.0
        assert len(metrics["cases"]) == metrics["total"]


def test_benchmark_provenance_exposes_oracle_wrapper_boundary():
    result = run_benchmark()
    assert result["http_signal_heuristic"]["tool_call_rate"] == 0.0
    assert result["retrieval_context_heuristic"]["tool_call_rate"] == 1.0
    assert result["langgraph_oracle_wrapper"]["oracle_derived_rate"] == 1.0
    assert result["langgraph_oracle_wrapper"]["tool_selection_accuracy"] == 1.0
    assert result["deterministic_oracle_reference"]["oracle_derived_rate"] == 1.0
    agent_sources = {row["prediction_source"] for row in result["langgraph_oracle_wrapper"]["cases"]}
    assert agent_sources <= {
        "langgraph_wrapper_over_deterministic_tool",
        "deterministic_oracle_fallback",
    }
    assert result["langgraph_oracle_wrapper"]["prediction_source_counts"]
