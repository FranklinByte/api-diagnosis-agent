"""Deterministic V0 diagnosis orchestration.

The function in this module is intentionally boring: it performs a fixed
sequence of contract, version, and schema checks and emits a typed report.  It
is the oracle used by the later Agent and evaluator, not an LLM replacement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from .models import (
    ApiRequest,
    ApiResponse,
    DiagnosisResult,
    Evidence,
    EvidenceType,
    FaultCategory,
    VerificationStep,
)
from .paths import project_root
from .validation import OpenAPIValidator
from .versioning import VersionChecker


PROJECT_ROOT = project_root()
DEFAULT_SPEC_PATH = PROJECT_ROOT / "specs" / "mock-api.json"


def _load_spec(spec: Any) -> Mapping[str, Any]:
    if isinstance(spec, Mapping):
        return spec
    path = Path(spec)
    return json.loads(path.read_text(encoding="utf-8"))


def _header(headers: Mapping[str, str], name: str) -> Optional[str]:
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return str(value)
    return None


def _base_evidence(request: ApiRequest, source: str, pointer: str, **details: Any) -> Evidence:
    return Evidence(
        source=source,
        source_type=EvidenceType.OPENAPI,
        api_name=request.api_name,
        endpoint=request.path,
        method=request.method,
        path=request.path,
        version=request.api_version,
        pointer=pointer,
        details=details,
    )


def _report(
    request: ApiRequest,
    *,
    status: str,
    category: Optional[FaultCategory],
    subtype: Optional[str],
    root_cause: str,
    signal: str,
    confidence: str,
    steps: list[VerificationStep],
    evidence: list[Evidence],
    repair: Optional[str] = None,
    needs_manual: bool = False,
    limitations: Optional[list[str]] = None,
) -> DiagnosisResult:
    return DiagnosisResult(
        request_id=request.request_id,
        status=status,  # type: ignore[arg-type]
        fault_category=category,
        fault_subtype=subtype,
        root_cause=root_cause,
        observed_signal=signal,
        confidence=confidence,  # type: ignore[arg-type]
        verification_steps=steps,
        evidence=evidence,
        repair_suggestion=repair,
        needs_manual=needs_manual,
        limitations=limitations or [],
    )


def _observed_response_fault(model: ApiRequest) -> Optional[tuple[FaultCategory, str, str, str]]:
    """Map explicit observed HTTP signals to stable V2 fault classes."""

    status = model.status_code
    message = (model.error_message or "").lower()
    if status in {408, 504} or "timeout" in message or "timed out" in message:
        return FaultCategory.TIMEOUT, "timeout", "The observed response exceeded the configured time budget.", f"HTTP status {status or 'unknown'} or timeout signal"
    if status in {401, 403}:
        return FaultCategory.AUTH_ERROR, "authentication_or_authorization", "The API rejected the request credentials or authorization.", f"HTTP status {status}"
    if status == 415:
        return FaultCategory.CONTENT_TYPE_ERROR, "unsupported_media_type", "The API rejected the request media type.", "HTTP status 415"
    if status == 429:
        return FaultCategory.RATE_LIMITED, "rate_limit", "The API rate limit was exceeded.", "HTTP status 429"
    if status is not None and 500 <= status <= 599:
        return FaultCategory.SERVER_ERROR, "server_error", "The upstream API returned a server-side error.", f"HTTP status {status}"
    return None


def diagnose_v0(
    request: ApiRequest | Mapping[str, Any],
    *,
    spec: Any = DEFAULT_SPEC_PATH,
    changelog_path: Optional[str | Path] = None,
    current_version: str = "v2",
) -> DiagnosisResult:
    """Run the fixed V0 checks and return a deterministic diagnosis report.

    Checks are ordered so endpoint/method/version mistakes are localized before
    schema validation.  A valid request is a confirmed *no-fault* result; an
    unsupported or ambiguous result is never presented as a root-cause claim.
    """

    model = request if isinstance(request, ApiRequest) else ApiRequest.model_validate(request)
    document = _load_spec(spec)
    # The contract's first server is the safe local default.  This keeps the
    # public fixture concise while still giving openapi-core a server URL to
    # resolve.  We do not invent required headers; a missing version header is
    # deliberately left for schema validation to report.
    updates: dict[str, Any] = {}
    if not model.base_url:
        servers = document.get("servers", [])
        if servers and isinstance(servers[0], Mapping):
            updates["base_url"] = servers[0].get("url")
    if model.body is not None and not model.content_type:
        updates["content_type"] = "application/json"
    if updates:
        model = model.model_copy(update=updates)
    paths = document.get("paths", {})
    path_item = paths.get(model.path)
    source = str(spec) if isinstance(spec, (str, Path)) else "openapi_spec"
    steps: list[VerificationStep] = []
    evidence: list[Evidence] = [
        Evidence(
            source="request",
            source_type=EvidenceType.INPUT,
            api_name=model.api_name,
            endpoint=model.path,
            method=model.method,
            path=model.path,
            version=model.api_version,
            details={"request_id": model.request_id},
        )
    ]

    if not isinstance(path_item, Mapping):
        ev = _base_evidence(model, source, "/paths", available_paths=list(paths))
        evidence.append(ev)
        steps.append(
            VerificationStep(
                action="locate_endpoint",
                tool="openapi_lookup",
                status="failed",
                summary=f"Path {model.path!r} is not declared by the OpenAPI contract.",
                evidence=[ev],
            )
        )
        return _report(
            model,
            status="confirmed",
            category=FaultCategory.PATH_ERROR,
            subtype="unknown_path",
            root_cause="The request path is absent from the OpenAPI paths object.",
            signal=f"No operation matches {model.path}",
            confidence="confirmed",
            steps=steps,
            evidence=evidence,
            repair="Use a path declared by the API contract.",
        )

    method_key = model.method.lower()
    operation = path_item.get(method_key)
    if not isinstance(operation, Mapping):
        allowed = sorted(
            str(key).upper()
            for key, value in path_item.items()
            if str(key).lower() in {"get", "post", "put", "patch", "delete", "head", "options"}
            and isinstance(value, Mapping)
        )
        ev = _base_evidence(model, source, f"/paths{model.path}", allowed_methods=allowed)
        evidence.append(ev)
        steps.append(
            VerificationStep(
                action="locate_operation",
                tool="openapi_lookup",
                status="failed",
                summary=f"Method {model.method} is not declared for {model.path}.",
                evidence=[ev],
            )
        )
        return _report(
            model,
            status="confirmed",
            category=FaultCategory.METHOD_ERROR,
            subtype="unsupported_method",
            root_cause="The HTTP method is not declared for the requested path.",
            signal=f"Allowed methods: {', '.join(allowed) or 'none'}",
            confidence="confirmed",
            steps=steps,
            evidence=evidence,
            repair=f"Use one of the declared methods: {', '.join(allowed) or 'none'}.",
        )

    steps.append(
        VerificationStep(
            action="locate_operation",
            tool="openapi_lookup",
            status="passed",
            summary=f"Located {model.method} {model.path} in the OpenAPI contract.",
            evidence=[_base_evidence(model, source, f"/paths{model.path}/{method_key}")],
        )
    )

    requested_version = model.api_version or _header(model.headers, "x-api-version")
    if requested_version:
        checker = VersionChecker(changelog_path) if changelog_path else VersionChecker()
        version_result = checker.check(
            requested_version=requested_version,
            current_version=current_version,
            # The OpenAPI title and changelog service identifier are allowed to
            # differ ("Weather Mock API" vs "weather-mock").  Unless the
            # caller supplied an explicit API name, match by endpoint/method.
            api_name=model.api_name,
            endpoint=model.path,
            method=model.method,
        )
        evidence.extend(version_result.evidence)
        version_status: Literal["passed", "failed"] = (
            "passed" if version_result.status == "compatible" else "failed"
        )
        steps.append(
            VerificationStep(
                action="check_version",
                tool="changelog_checker",
                status=version_status,
                summary=version_result.reason,
                evidence=version_result.evidence,
            )
        )
        if version_result.status == "incompatible":
            return _report(
                model,
                status="confirmed",
                category=FaultCategory.VERSION_INCOMPATIBLE,
                subtype="breaking_version_change",
                root_cause=version_result.reason,
                signal=f"Requested {requested_version}; current contract is {current_version}.",
                confidence="confirmed",
                steps=steps,
                evidence=evidence,
                repair=f"Migrate the request to API version {current_version}.",
            )
        if version_result.status == "unknown":
            return _report(
                model,
                status="unsupported",
                category=FaultCategory.UNSUPPORTED,
                subtype="version_evidence_missing",
                root_cause=version_result.reason,
                signal=f"Requested version {requested_version} cannot be proven compatible.",
                confidence="none",
                steps=steps,
                evidence=evidence,
                needs_manual=True,
                limitations=["The static changelog contains no compatibility evidence."],
            )

    observed_fault = _observed_response_fault(model)
    if observed_fault:
        category, subtype, root_cause, signal = observed_fault
        ev = Evidence(
            source="observed_response",
            source_type=EvidenceType.MOCK_RESPONSE,
            api_name=model.api_name,
            endpoint=model.path,
            method=model.method,
            path=model.path,
            version=model.api_version,
            excerpt=model.error_message,
            details={"status_code": model.status_code},
        )
        evidence.append(ev)
        steps.append(
            VerificationStep(
                action="classify_observed_response",
                tool="response_signal_classifier",
                status="passed",
                summary=signal,
                evidence=[ev],
            )
        )
        return _report(
            model,
            status="confirmed",
            category=category,
            subtype=subtype,
            root_cause=root_cause,
            signal=signal,
            confidence="confirmed",
            steps=steps,
            evidence=evidence,
            repair="Inspect the corresponding authentication, content-type, rate-limit, timeout, or upstream server condition.",
            needs_manual=category in {FaultCategory.AUTH_ERROR, FaultCategory.SERVER_ERROR},
        )

    if model.response_body is not None and model.status_code is not None and 200 <= model.status_code < 300:
        validator = OpenAPIValidator(spec, source=source)
        response_result = validator.validate_response(
            model,
            ApiResponse(
                status_code=model.status_code,
                headers=model.headers,
                body=model.response_body,
                content_type=model.content_type or "application/json",
            ),
        )
        evidence.extend(response_result.evidence)
        if not response_result.valid:
            issue = response_result.issues[0]
            steps.append(
                VerificationStep(
                    action="validate_response",
                    tool="openapi_core",
                    status="failed",
                    summary=issue.message,
                    evidence=response_result.evidence,
                )
            )
            return _report(
                model,
                status="confirmed",
                category=FaultCategory.RESPONSE_SCHEMA_CHANGE,
                subtype="response_contract_mismatch",
                root_cause="The observed successful response does not conform to the declared response schema.",
                signal=issue.message,
                confidence="confirmed",
                steps=steps,
                evidence=evidence,
                repair="Compare the observed payload with the current response schema and changelog.",
                needs_manual=True,
            )

    validator = OpenAPIValidator(spec, source=source)
    validation = validator.validate_request(model)
    evidence.extend(validation.evidence)
    if validation.valid:
        steps.append(
            VerificationStep(
                action="validate_request",
                tool="openapi_core",
                status="passed",
                summary="Request parameters and JSON body conform to the OpenAPI schema.",
                evidence=validation.evidence,
            )
        )
        return _report(
            model,
            status="confirmed",
            category=None,
            subtype="no_fault_detected",
            root_cause="No deterministic contract fault was found.",
            signal="Path, method, version, and request schema checks passed.",
            confidence="confirmed",
            steps=steps,
            evidence=evidence,
            limitations=["V0 does not infer undocumented server-side causes."],
        )

    categories = [issue.category for issue in validation.issues if issue.category]
    unique_categories = list(dict.fromkeys(categories))
    if len(unique_categories) != 1:
        return _report(
            model,
            status="ambiguous",
            category=FaultCategory.AMBIGUOUS,
            subtype="multiple_schema_failures",
            root_cause="The request violates multiple independent schema rules.",
            signal="; ".join(issue.message for issue in validation.issues),
            confidence="medium",
            steps=steps
            + [
                VerificationStep(
                    action="validate_request",
                    tool="openapi_core",
                    status="failed",
                    summary="Multiple schema issues were returned.",
                    evidence=validation.evidence,
                )
            ],
            evidence=evidence,
            repair="Correct every reported schema issue, then replay validation.",
            needs_manual=True,
        )
    category = unique_categories[0] if unique_categories else FaultCategory.UNSUPPORTED
    issue = validation.issues[0]
    repair = {
        FaultCategory.MISSING_REQUIRED_FIELD: "Add the required field identified by the JSON Pointer.",
        FaultCategory.FIELD_TYPE_ERROR: "Send the field using the JSON type declared in the schema.",
        FaultCategory.UNKNOWN_FIELD: "Remove fields not declared by the request schema.",
        FaultCategory.ENUM_INVALID: "Use one of the enum values declared by the schema.",
    }.get(category, "Correct the request according to the OpenAPI validation evidence.")
    steps.append(
        VerificationStep(
            action="validate_request",
            tool="openapi_core",
            status="failed",
            summary=issue.message,
            evidence=validation.evidence,
        )
    )
    return _report(
        model,
        status="confirmed" if category != FaultCategory.UNSUPPORTED else "unsupported",
        category=category,
        subtype=issue.keyword or category.value,
        root_cause=issue.message,
        signal=f"Schema violation at {issue.pointer or 'request boundary'}.",
        confidence="confirmed" if category != FaultCategory.UNSUPPORTED else "none",
        steps=steps,
        evidence=evidence,
        repair=repair,
    )


__all__ = ["diagnose_v0"]
