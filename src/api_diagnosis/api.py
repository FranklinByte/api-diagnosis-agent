"""FastAPI boundary for the frozen diagnosis service."""

from __future__ import annotations

import os
import uuid
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent import (
    pending_approval,
    resume_agent_state,
    run_agent_state,
    run_provider_agent,
)
from .diagnosis import diagnose_v0
from .evaluator import run_benchmark
from .models import ApiRequest, DiagnosisResult
from .security import sanitize_payload
from .trace import TraceRecorder


class DiagnoseEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: ApiRequest
    mode: Literal["agent", "deterministic", "provider"] = "agent"
    thread_id: Optional[str] = Field(default=None, min_length=8, max_length=128)
    replay_requested: bool = False

    @model_validator(mode="after")
    def validate_replay_mode(self) -> "DiagnoseEnvelope":
        if self.replay_requested and self.mode != "agent":
            raise ValueError("replay_requested requires mode=agent for checkpointed human approval")
        return self


class ResumeEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(..., min_length=8, max_length=128)
    approval_token: str = Field(..., min_length=20, max_length=256)
    approved: bool


class DiagnoseResponse(BaseModel):
    report: DiagnosisResult
    trace: dict[str, Any]
    thread_id: str
    pending_approval: Optional[dict[str, Any]] = None


app = FastAPI(
    title="Third-Party API Diagnosis Agent",
    version="0.2.0",
    description="Deterministic OpenAPI diagnosis boundary with an offline Agent path.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api-diagnosis-agent"}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(payload: DiagnoseEnvelope) -> DiagnoseResponse:
    trace = TraceRecorder()
    event = trace.start("diagnose", request_id=payload.request.request_id, mode=payload.mode)
    agent_events: list[dict[str, Any]] = []
    thread_id = payload.thread_id or payload.request.request_id or uuid.uuid4().hex
    pending: Optional[dict[str, Any]] = None
    try:
        if payload.mode == "agent":
            state = run_agent_state(
                payload.request.model_dump(mode="json", by_alias=False),
                thread_id=thread_id,
                replay_requested=payload.replay_requested,
            )
            pending = pending_approval(state)
            report = (
                DiagnosisResult.model_validate(state["report"])
                if state.get("report")
                else _pending_report(payload.request.request_id)
            )
            agent_events = state.get("trace", [])
        elif payload.mode == "provider":
            if os.getenv("API_DIAGNOSIS_ENABLE_PROVIDER", "0").lower() not in {
                "1",
                "true",
                "yes",
            }:
                raise RuntimeError("provider mode is disabled")
            report = run_provider_agent(
                payload.request.model_dump(mode="json", by_alias=False)
            )
        else:
            report = diagnose_v0(
                sanitize_payload(
                    payload.request.model_dump(mode="json", by_alias=False)
                )
            )
    except RuntimeError as exc:
        trace.finish(event, status="error", error=type(exc).__name__)
        raise HTTPException(status_code=503, detail="requested diagnosis mode is unavailable") from exc
    except (ValueError, FileNotFoundError, TypeError) as exc:
        trace.finish(event, status="error", error=str(exc)[:512])
        raise HTTPException(status_code=400, detail="invalid diagnosis request") from exc
    trace.finish(event, status=report.status, fault_category=report.fault_category.value if report.fault_category else None)
    trace_value = trace.as_dict()
    trace_value["agent_events"] = agent_events
    return DiagnoseResponse(
        report=report,
        trace=trace_value,
        thread_id=thread_id,
        pending_approval=pending,
    )


@app.post("/diagnose/resume", response_model=DiagnoseResponse)
def resume_diagnosis(payload: ResumeEnvelope) -> DiagnoseResponse:
    """Resume one checkpointed Mock replay approval decision."""

    trace = TraceRecorder()
    event = trace.start("resume_diagnose", thread_id=payload.thread_id)
    try:
        state = resume_agent_state(
            thread_id=payload.thread_id,
            approval_token=payload.approval_token,
            approved=payload.approved,
        )
        report = DiagnosisResult.model_validate(state["report"])
    except (ValueError, TypeError) as exc:
        trace.finish(event, status="error", error=type(exc).__name__)
        raise HTTPException(status_code=400, detail="invalid or expired approval") from exc
    trace.finish(event, status=report.status)
    trace_value = trace.as_dict()
    trace_value["agent_events"] = state.get("trace", [])
    return DiagnoseResponse(
        report=report,
        trace=trace_value,
        thread_id=payload.thread_id,
    )


def _pending_report(request_id: str) -> DiagnosisResult:
    return DiagnosisResult(
        request_id=request_id,
        status="unconfirmed",
        root_cause="A non-idempotent Mock replay is waiting for human approval.",
        confidence="none",
        needs_manual=True,
        limitations=["pending_replay_approval"],
        revalidation_steps=["Resume the checkpoint with the approval token."],
    )


@app.post("/evaluate")
def evaluate() -> dict[str, Any]:
    """Run the offline benchmark; no external model call is made."""

    return run_benchmark()


__all__ = [
    "DiagnoseEnvelope",
    "DiagnoseResponse",
    "ResumeEnvelope",
    "app",
]
