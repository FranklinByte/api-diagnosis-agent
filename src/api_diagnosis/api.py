"""FastAPI boundary for the frozen diagnosis service."""

from __future__ import annotations

import os
import uuid
from typing import Any, Literal, Optional

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent import (
    pending_approval,
    resume_agent_state,
    run_agent_state,
    run_provider_agent,
)
from .diagnosis import diagnose_v0
from .evaluator import run_benchmark
from .general import run_general_diagnosis
from .llm import load_config, load_dotenv
from .models import ApiRequest, DiagnosisResult
from .security import sanitize_payload
from .spec_store import BUILTIN_SPEC_ID, MAX_SPEC_BYTES, SpecError, list_operations, list_specs, load_spec, save_spec
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


class SpecUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, max_length=MAX_SPEC_BYTES)
    name: Optional[str] = Field(default=None, max_length=80)


class GeneralDiagnoseEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_id: str = Field(default=BUILTIN_SPEC_ID, min_length=1, max_length=64)
    request: ApiRequest
    explain: bool = True


class GeneralDiagnoseResponse(BaseModel):
    report: DiagnosisResult
    label: Optional[str] = None
    spec: dict[str, Any]
    findings: list[dict[str, Any]]
    similar_paths: list[str]
    explanation: Optional[dict[str, Any]] = None


# Model credentials may be kept in a local, git-ignored ``.env`` file.
load_dotenv()
_STATIC_DIR = Path(__file__).resolve().parent / "static"

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


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html; charset=utf-8")


@app.get("/v1/status")
def status() -> dict[str, Any]:
    """Whether a real model is configured; the key itself is never returned."""

    config = load_config()
    return {"model_configured": config is not None, "model": config.model if config else None}


@app.get("/specs")
def specs() -> list[dict[str, Any]]:
    return list_specs()


@app.post("/specs", status_code=201)
def upload_spec(payload: SpecUpload) -> dict[str, Any]:
    """Register one OpenAPI 3.x document (JSON or YAML text)."""

    try:
        return save_spec(payload.content, payload.name)
    except SpecError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/specs/{spec_id}/operations")
def spec_operations(spec_id: str) -> list[dict[str, str]]:
    try:
        return list_operations(load_spec(spec_id)[1])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown spec_id") from exc


@app.post("/v1/diagnose", response_model=GeneralDiagnoseResponse)
def diagnose_general(payload: GeneralDiagnoseEnvelope) -> GeneralDiagnoseResponse:
    """Diagnose one request against any registered OpenAPI document.

    The fault label comes from deterministic contract checks.  When a model is
    configured it writes the explanation and a corrected request, and that
    corrected request is validated against the document before it is returned.
    """

    try:
        result = run_general_diagnosis(payload.spec_id, payload.request, with_explanation=payload.explain)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown spec_id") from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="invalid diagnosis request") from exc
    return GeneralDiagnoseResponse(**result)


@app.post("/evaluate")
def evaluate() -> dict[str, Any]:
    """Run the offline benchmark; no external model call is made."""

    return run_benchmark()


__all__ = [
    "GeneralDiagnoseEnvelope",
    "GeneralDiagnoseResponse",
    "SpecUpload",
    "DiagnoseEnvelope",
    "DiagnoseResponse",
    "ResumeEnvelope",
    "app",
]
