from __future__ import annotations

import api_diagnosis.agent as agent_module
from api_diagnosis.agent import (
    TOOLS,
    apply_deterministic_guardrail,
    build_agent_graph,
    pending_approval,
    resume_agent_state,
    run_agent,
    run_agent_state,
    run_provider_graph_state,
)
from api_diagnosis.models import DiagnosisResult, FaultCategory, ReplayResult


def test_langgraph_agent_returns_structured_report_and_uses_tools():
    report = run_agent(
        {
            "request_id": "agent-001",
            "method": "POST",
            "path": "/weather/forecast",
            "headers": {"X-API-Version": "v1"},
            "body": {"city": "Singapore", "units": "metric", "days": 3},
        }
    )
    assert report.status == "confirmed"
    assert report.fault_category == FaultCategory.VERSION_INCOMPATIBLE
    assert {tool.name for tool in TOOLS} == {
        "retrieve_contract",
        "validate_openapi_contract",
        "check_api_version",
        "inspect_observed_response",
        "replay_mock_request",
        "deterministic_diagnose",
    }
    assert report.hypotheses[0].validation_tool == "check_api_version"
    assert report.verification_steps[0].tool == "check_api_version"


def test_agent_graph_bounds_unsupported_retry_loop():
    graph = build_agent_graph()
    state = graph.invoke(
        {
            "request": {
                "request_id": "agent-loop-001",
                "method": "POST",
                "path": "/weather/forecast",
                "headers": {"X-API-Version": "v9"},
                "body": {"city": "Singapore", "units": "metric", "days": 3},
            },
            "trace": [],
        },
        config={"configurable": {"thread_id": "agent-loop-001"}},
    )
    assert state["report"]["status"] == "unsupported"
    assert state["iteration_count"] == 2
    assert state["tool_history"] == ["check_api_version", "validate_openapi_contract"]
    assert "revise_hypotheses" in [event["node"] for event in state["trace"]]
    assert "No matching changelog evidence" in state["query"]


def test_replay_pauses_for_approval_and_resumes_from_checkpoint(monkeypatch):
    class Replayer:
        def replay(self, request):
            assert request.approved
            return ReplayResult(status="success", status_code=200, request_count=1)

    monkeypatch.setattr(agent_module, "get_replayer", lambda: Replayer())
    thread_id = "approval-thread-001"
    state = run_agent_state(
        {
            "request_id": "approval-001",
            "base_url": "http://127.0.0.1:8001",
            "method": "POST",
            "path": "/weather/forecast",
            "headers": {"X-API-Version": "v2"},
            "body": {"city": "Singapore", "units": "metric", "days": 3},
        },
        thread_id=thread_id,
        replay_requested=True,
    )
    pending = pending_approval(state)
    assert pending and pending["type"] == "replay_approval"
    assert not state.get("report")

    resumed = resume_agent_state(
        thread_id=thread_id,
        approval_token=pending["approval_token"],
        approved=True,
    )
    assert resumed["report"]["replay_result"]["status"] == "success"
    assert resumed["approval_status"] == "approved"


def test_provider_guardrail_cannot_override_unknown_version_evidence():
    provider = DiagnosisResult(status="confirmed", fault_category=FaultCategory.VERSION_INCOMPATIBLE)
    deterministic = DiagnosisResult(status="unsupported", fault_category=FaultCategory.UNSUPPORTED)
    guarded = apply_deterministic_guardrail(provider, deterministic)
    assert guarded.status == "unsupported"
    assert guarded.fault_category == FaultCategory.UNSUPPORTED
    assert "deterministic_guardrail_corrected_provider_label" in guarded.limitations


def test_provider_planner_uses_bound_tool_call_and_keeps_guarded_tool_execution():
    class Response:
        tool_calls = [{"name": "check_api_version", "args": {}}]

    class Bound:
        def invoke(self, prompt):
            assert "Pending hypotheses" in prompt
            return Response()

    class Model:
        def bind_tools(self, tools):
            assert "check_api_version" in {item.name for item in tools}
            return Bound()

    state = run_provider_graph_state(
        {
            "request_id": "provider-planner-001",
            "method": "POST",
            "path": "/weather/forecast",
            "headers": {"X-API-Version": "v1"},
            "body": {"city": "Singapore", "units": "metric", "days": 3},
        },
        model=Model(),
    )
    select = next(event for event in state["trace"] if event["node"] == "select_tool")
    assert select["planner"] == "provider_tool_call"
    assert state["tool_history"] == ["check_api_version"]
    assert state["report"]["fault_category"] == "version_incompatible"
