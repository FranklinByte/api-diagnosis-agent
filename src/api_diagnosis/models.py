"""Typed data contracts for the deterministic V0 diagnosis tools.

The models in this module deliberately contain no LLM-specific fields.  They
are the boundary between the HTTP/OpenAPI tools and the future orchestrator:
inputs are JSON-like values, evidence is explicit, and a diagnosis can state
that the available evidence is insufficient.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Model(BaseModel):
    """Common Pydantic configuration used by all public contracts."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class HttpMethod(str, Enum):
    """HTTP methods accepted by the V0 contracts."""

    GET = "GET"
    HEAD = "HEAD"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"


class FaultCategory(str, Enum):
    """Stable fault labels shared by fixtures and deterministic tools."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    FIELD_TYPE_ERROR = "field_type_error"
    UNKNOWN_FIELD = "unknown_field"
    ENUM_INVALID = "enum_invalid"
    METHOD_ERROR = "method_error"
    PATH_ERROR = "path_error"
    CONTENT_TYPE_ERROR = "content_type_error"
    AUTH_ERROR = "auth_error"
    VERSION_INCOMPATIBLE = "version_incompatible"
    RESPONSE_SCHEMA_CHANGE = "response_schema_change"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


class EvidenceType(str, Enum):
    """Where a diagnostic fact came from."""

    OPENAPI = "openapi"
    SCHEMA = "schema"
    CHANGELOG = "changelog"
    TOOL = "tool"
    MOCK_RESPONSE = "mock_response"
    INPUT = "input"
    TRACE = "trace"


class Evidence(_Model):
    """A citable fact used by a diagnosis.

    ``pointer`` should be a JSON Pointer or another stable location in the
    source.  The fields are optional because a tool error may only have a
    human-readable message, but a normal successful validation should fill in
    the endpoint, version, and pointer whenever they are available.
    """

    source: str = Field(..., min_length=1, max_length=512)
    source_type: EvidenceType = EvidenceType.TOOL
    title: Optional[str] = Field(default=None, max_length=512)
    api_name: Optional[str] = Field(default=None, max_length=128)
    endpoint: Optional[str] = Field(default=None, max_length=512)
    method: Optional[str] = Field(default=None, max_length=16)
    path: Optional[str] = Field(default=None, max_length=512)
    version: Optional[str] = Field(default=None, max_length=64)
    pointer: Optional[str] = Field(default=None, max_length=512)
    excerpt: Optional[str] = Field(default=None, max_length=4096)
    details: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("method")
    @classmethod
    def normalize_method(cls, value: Optional[str]) -> Optional[str]:
        return value.upper() if value else value


class ApiRequest(_Model):
    """An API request supplied to the validator or replay tool.

    Headers are expected to be already sanitized at the public boundary.  The
    replay implementation applies a second defensive sanitization pass before
    any network call.
    """

    request_id: str = Field(default="", max_length=128)
    api_name: Optional[str] = Field(default=None, max_length=128)
    base_url: Optional[str] = Field(default=None, max_length=2048)
    method: str = Field(..., min_length=1, max_length=16)
    path: str = Field(..., min_length=1, max_length=2048)
    headers: Dict[str, str] = Field(
        default_factory=dict,
        alias="sanitized_headers",
    )
    query_params: Dict[str, Any] = Field(default_factory=dict)
    body: Any = Field(default=None, alias="request_body")
    content_type: Optional[str] = Field(default=None, max_length=256)
    api_version: Optional[str] = Field(default=None, max_length=64)
    status_code: Optional[int] = Field(default=None, ge=100, le=599)
    response_body: Any = None
    error_message: Optional[str] = Field(default=None, max_length=4096)

    @field_validator("method")
    @classmethod
    def normalize_method(cls, value: str) -> str:
        return value.upper()


class ApiResponse(_Model):
    """Observed HTTP response passed to response-schema validation."""

    status_code: int = Field(..., ge=100, le=599)
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Any = None
    content_type: Optional[str] = Field(default=None, max_length=256)
    trace_id: Optional[str] = Field(default=None, max_length=128)


class ValidationIssue(_Model):
    """One normalized openapi-core validation error."""

    code: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4096)
    location: Optional[str] = Field(default=None, max_length=64)
    pointer: Optional[str] = Field(default=None, max_length=512)
    keyword: Optional[str] = Field(default=None, max_length=128)
    category: Optional[FaultCategory] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(_Model):
    """Deterministic result returned by request/response validation."""

    valid: bool
    direction: Literal["request", "response"]
    issues: List[ValidationIssue] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    operation: Optional[str] = Field(default=None, max_length=512)
    path: Optional[str] = Field(default=None, max_length=2048)
    method: Optional[str] = Field(default=None, max_length=16)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Backward-compatible readable alias for ``issues``."""

        return self.issues


class VerificationStep(_Model):
    """A deterministic action recorded in a diagnosis report."""

    action: str = Field(..., min_length=1, max_length=128)
    tool: str = Field(..., min_length=1, max_length=128)
    status: Literal["passed", "failed", "blocked", "skipped", "unknown"]
    summary: str = Field(..., min_length=1, max_length=4096)
    evidence: List[Evidence] = Field(default_factory=list)


class DiagnosisHypothesis(_Model):
    """One bounded root-cause hypothesis tracked by the Agent loop."""

    hypothesis_id: str = Field(..., min_length=1, max_length=64)
    category: Optional[FaultCategory] = None
    reason: str = Field(..., min_length=1, max_length=2048)
    validation_tool: str = Field(..., min_length=1, max_length=128)
    status: Literal["pending", "supported", "rejected", "unknown"] = "pending"
    evidence_refs: List[str] = Field(default_factory=list)


class ReplayPolicy(_Model):
    """Safety limits for the controlled mock request replayer."""

    allowed_base_urls: List[str] = Field(
        default_factory=list,
        alias="allowed_urls",
    )
    allowed_methods: Set[str] = Field(default_factory=lambda: {"GET", "HEAD"})
    timeout_seconds: float = Field(default=3.0, gt=0, le=60)
    max_requests: int = Field(default=2, ge=1, le=100)
    max_response_bytes: int = Field(default=1_048_576, ge=1, le=20_000_000)
    max_request_bytes: int = Field(default=1_048_576, ge=1, le=20_000_000)
    require_approval_for_non_idempotent: bool = True
    allow_redirects: bool = False
    forward_sensitive_headers: bool = False

    @field_validator("allowed_methods")
    @classmethod
    def normalize_methods(cls, value: Set[str]) -> Set[str]:
        return {method.upper() for method in value}


class ReplayRequest(_Model):
    """Request envelope consumed by :class:`MockRequestReplayer`."""

    url: str = Field(..., min_length=1, max_length=4096)
    method: str = Field(..., min_length=1, max_length=16)
    headers: Dict[str, str] = Field(default_factory=dict)
    query_params: Dict[str, Any] = Field(default_factory=dict)
    body: Any = None
    approved: bool = False
    timeout_seconds: Optional[float] = Field(default=None, gt=0, le=60)

    @field_validator("method")
    @classmethod
    def normalize_method(cls, value: str) -> str:
        return value.upper()


class ReplayResult(_Model):
    """Bounded, secret-safe result of a replay attempt."""

    status: Literal[
        "success",
        "http_error",
        "blocked",
        "timeout",
        "network_error",
        "response_too_large",
        "invalid_request",
    ]
    status_code: Optional[int] = Field(default=None, ge=100, le=599)
    response_headers: Dict[str, str] = Field(default_factory=dict)
    response_body: Any = None
    response_size_bytes: int = Field(default=0, ge=0)
    elapsed_ms: Optional[float] = Field(default=None, ge=0)
    trace_id: Optional[str] = Field(default=None, max_length=128)
    error: Optional[str] = Field(default=None, max_length=4096)
    request_count: int = Field(default=0, ge=0)
    sanitized_request_headers: Dict[str, str] = Field(default_factory=dict)


class VersionChange(_Model):
    """One normalized changelog entry relevant to a version comparison."""

    from_version: Optional[str] = Field(default=None, max_length=64)
    to_version: Optional[str] = Field(default=None, max_length=64)
    endpoint: Optional[str] = Field(default=None, max_length=512)
    method: Optional[str] = Field(default=None, max_length=16)
    breaking: bool = False
    summary: Optional[str] = Field(default=None, max_length=4096)
    affected_fields: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("method")
    @classmethod
    def normalize_method(cls, value: Optional[str]) -> Optional[str]:
        return value.upper() if value else value


class VersionCheckResult(_Model):
    """Result of comparing a requested API version with the mock contract."""

    status: Literal["compatible", "incompatible", "unknown"]
    compatible: Optional[bool] = None
    requested_version: str = Field(..., min_length=1, max_length=64)
    current_version: Optional[str] = Field(default=None, max_length=64)
    api_name: Optional[str] = Field(default=None, max_length=128)
    endpoint: Optional[str] = Field(default=None, max_length=512)
    method: Optional[str] = Field(default=None, max_length=16)
    changes: List[VersionChange] = Field(default_factory=list)
    affected_fields: List[str] = Field(default_factory=list)
    reason: str = Field(..., min_length=1, max_length=4096)
    evidence: List[Evidence] = Field(default_factory=list)


class DiagnosisResult(_Model):
    """Structured deterministic diagnosis report.

    ``status`` deliberately distinguishes a confirmed finding from an
    unsupported or ambiguous input.  A consumer must not treat an
    ``unknown``/``ambiguous`` report as a confirmed root cause.
    """

    request_id: str = Field(default="", max_length=128)
    status: Literal["confirmed", "unconfirmed", "unsupported", "ambiguous", "error"]
    fault_category: Optional[FaultCategory] = None
    fault_subtype: Optional[str] = Field(default=None, max_length=128)
    root_cause: Optional[str] = Field(default=None, max_length=4096)
    observed_signal: Optional[str] = Field(default=None, max_length=4096)
    confidence: Literal["confirmed", "high", "medium", "low", "none"] = "none"
    hypotheses: List[DiagnosisHypothesis] = Field(default_factory=list)
    verification_steps: List[VerificationStep] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    repair_suggestion: Optional[str] = Field(default=None, max_length=4096)
    corrected_request: Optional[ApiRequest] = None
    replay_result: Optional[ReplayResult] = None
    needs_manual: bool = False
    limitations: List[str] = Field(default_factory=list)
    revalidation_steps: List[str] = Field(default_factory=list)
    execution_metadata: Dict[str, Any] = Field(default_factory=dict)


# Names used by callers that prefer the more explicit terminology in the
# project context.  Aliases keep the data contract small without forcing the
# future orchestrator to duplicate models.
DiagnosisInput = ApiRequest
RequestInput = ApiRequest
ResponseInput = ApiResponse
DiagnosisReport = DiagnosisResult


__all__ = [
    "ApiRequest",
    "ApiResponse",
    "DiagnosisInput",
    "DiagnosisHypothesis",
    "DiagnosisReport",
    "DiagnosisResult",
    "Evidence",
    "EvidenceType",
    "FaultCategory",
    "HttpMethod",
    "ReplayPolicy",
    "ReplayRequest",
    "ReplayResult",
    "RequestInput",
    "ResponseInput",
    "ValidationIssue",
    "ValidationResult",
    "VerificationStep",
    "VersionChange",
    "VersionCheckResult",
]
