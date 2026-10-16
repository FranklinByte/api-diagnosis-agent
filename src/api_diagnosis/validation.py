"""openapi-core adapter used by the V0 deterministic validation tools.

``openapi-core`` works with small protocol objects rather than requiring a
specific web framework.  This module converts the project's Pydantic request
and response envelopes to those protocol objects and normalizes library
exceptions into stable :class:`ValidationResult` values.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import parse_qsl, urlsplit

from .models import (
    ApiRequest,
    ApiResponse,
    Evidence,
    EvidenceType,
    FaultCategory,
    ValidationIssue,
    ValidationResult,
)
from .security import sanitize_value


class OpenAPICoreUnavailable(RuntimeError):
    """Raised when validation is requested without the openapi-core package."""


def _import_openapi():
    try:
        from openapi_core import OpenAPI
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OpenAPICoreUnavailable(
            "openapi-core is required for OpenAPI request/response validation"
        ) from exc
    return OpenAPI


class _CaseInsensitiveMap(Mapping[str, Any]):
    """Small case-insensitive mapping accepted by openapi-core headers."""

    def __init__(self, values: Optional[Mapping[str, Any]] = None):
        self._values = {
            str(key).lower(): value for key, value in (values or {}).items()
        }

    def __getitem__(self, key: str) -> Any:
        return self._values[str(key).lower()]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(str(key).lower(), default)


class _MultiValueMap(Mapping[str, Any]):
    """Mapping with ``getlist`` semantics used by query deserializers."""

    def __init__(self, values: Optional[Mapping[str, Any]] = None):
        self._values: dict[str, list[Any]] = {}
        for key, value in (values or {}).items():
            self._values[str(key)] = (
                list(value) if isinstance(value, (list, tuple)) else [value]
            )

    def __getitem__(self, key: str) -> Any:
        values = self._values[str(key)]
        return values[0] if values else ""

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def getlist(self, key: str) -> list[Any]:
        return list(self._values.get(str(key), []))


def _json_bytes(value: Any) -> Optional[bytes]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _safe_value(value: Any, limit: int = 2048) -> Any:
    """Make exception details bounded, serializable, and secret-safe."""

    return sanitize_value(
        value,
        max_depth=8,
        max_items=50,
        max_string_length=limit,
    )


def _pointer(path: Any) -> Optional[str]:
    if path is None:
        return None
    if isinstance(path, str):
        if not path:
            return None
        return path if path.startswith("/") else "/" + path
    try:
        parts = list(path)
    except TypeError:
        parts = [path]
    escaped = []
    for part in parts:
        text = str(part).replace("~", "~0").replace("/", "~1")
        escaped.append(text)
    return "/" + "/".join(escaped) if escaped else None


def _schema_error_details(error: BaseException) -> list[Any]:
    details = getattr(error, "details", None)
    values: list[Any] = []
    if isinstance(details, Mapping):
        nested = details.get("schema_errors")
        if isinstance(nested, Iterable) and not isinstance(nested, (str, bytes)):
            values.extend(nested)
    nested_attr = getattr(error, "schema_errors", None)
    if isinstance(nested_attr, Iterable) and not isinstance(nested_attr, (str, bytes)):
        values.extend(nested_attr)
    cause = getattr(error, "__cause__", None)
    if cause is not None and cause is not error:
        values.extend(_schema_error_details(cause))
    return values


def _classify_error(
    error: BaseException,
    message: str,
    *,
    keyword: Optional[str] = None,
) -> FaultCategory:
    text = f"{type(error).__name__} {message}".lower()
    keyword_text = (keyword or "").lower()
    if keyword_text in {"required", "dependentrequired"} or "required property" in text:
        return FaultCategory.MISSING_REQUIRED_FIELD
    if keyword_text in {"enum", "const", "oneof", "anyof"} or "is not one of" in text:
        return FaultCategory.ENUM_INVALID
    if keyword_text in {"additionalproperties", "unevaluatedproperties"} or "additional propert" in text:
        return FaultCategory.UNKNOWN_FIELD
    if keyword_text in {"type", "format"} or "is not of type" in text:
        return FaultCategory.FIELD_TYPE_ERROR
    if "method" in text and any(word in text for word in ("not", "allow", "found")):
        return FaultCategory.METHOD_ERROR
    if "path" in text or "operation" in text or "endpoint" in text:
        return FaultCategory.PATH_ERROR
    if "media" in text or "content type" in text or "content_type" in text:
        return FaultCategory.CONTENT_TYPE_ERROR
    if "missing" in text and any(
        word in text for word in ("required", "parameter", "body", "field")
    ):
        return FaultCategory.MISSING_REQUIRED_FIELD
    if "enum" in text or "oneof" in text or "const" in text:
        return FaultCategory.ENUM_INVALID
    if any(word in text for word in ("additional", "unknown", "extra")):
        return FaultCategory.UNKNOWN_FIELD
    if any(
        word in text
        for word in ("type", "schema", "validation", "invalidrequestbody")
    ):
        return FaultCategory.FIELD_TYPE_ERROR
    return FaultCategory.UNSUPPORTED


def _normalize_error(error: BaseException) -> ValidationIssue:
    message = str(error).strip() or type(error).__name__
    details = getattr(error, "details", None)
    safe_details = _safe_value(details if isinstance(details, Mapping) else {})
    pointer = None
    keyword = None
    classification_message = message
    schema_errors = _schema_error_details(error)
    if schema_errors:
        first = schema_errors[0]
        if isinstance(first, Mapping):
            first_path = first.get("path")
            keyword_value = first.get("validator") or first.get("keyword")
            first_message = str(first.get("message") or "")
        else:
            first_path = getattr(first, "path", None)
            keyword_value = getattr(first, "validator", None)
            first_message = str(getattr(first, "message", "") or "")
        # openapi-core 0.23 exposes normalized schema errors as dictionaries;
        # older releases expose jsonschema.ValidationError objects. Support both
        # and recover the required field name when the path is empty.
        if first_path in (None, [], ()) and first_message:
            required_match = re.match(r"^['\"]([^'\"]+)['\"] is a required property", first_message)
            if required_match:
                first_path = [required_match.group(1)]
        pointer = _pointer(first_path)
        keyword = str(keyword_value) if keyword_value is not None else None
        # openapi-core may combine several jsonschema failures into one
        # exception message.  Classify from the first normalized failure so a
        # required-field failure cannot mask an earlier unknown-field failure.
        classification_message = first_message or message
        if (
            pointer is None
            and "additional propert" in classification_message.lower()
        ):
            unexpected_match = re.search(
                r"['\"]([^'\"]+)['\"]\s+was unexpected",
                classification_message,
            )
            if unexpected_match:
                pointer = _pointer([unexpected_match.group(1)])
        if not message:
            message = first_message or type(error).__name__
        if isinstance(safe_details, dict):
            safe_details.setdefault("schema_errors", [])
            safe_details["schema_errors"] = [
                _safe_value(
                    {
                        "message": item.get("message", str(item)) if isinstance(item, Mapping) else getattr(item, "message", str(item)),
                        "path": _pointer(item.get("path") if isinstance(item, Mapping) else getattr(item, "path", None)),
                        "keyword": item.get("validator") or item.get("keyword") if isinstance(item, Mapping) else getattr(item, "validator", None),
                    }
                )
                for item in schema_errors[:20]
            ]
    category = _classify_error(error, classification_message, keyword=keyword)
    return ValidationIssue(
        code=type(error).__name__,
        message=message[:4096],
        pointer=pointer,
        keyword=keyword,
        category=category,
        details=safe_details if isinstance(safe_details, dict) else {},
    )


def _schema_item_issue(base: ValidationIssue, item: Any) -> Optional[ValidationIssue]:
    """Expand one normalized jsonschema child error when a library bundles many."""

    if isinstance(item, Mapping):
        message = str(item.get("message") or "").strip()
        raw_path = item.get("path")
        keyword = item.get("validator") or item.get("keyword")
    else:
        message = str(getattr(item, "message", "") or "").strip()
        raw_path = getattr(item, "path", None)
        keyword = getattr(item, "validator", None)
    if not message:
        return None
    pointer = _pointer(raw_path)
    if pointer is None and "additional propert" in message.lower():
        match = re.search(r"['\"]([^'\"]+)['\"]\s+was unexpected", message)
        if match:
            pointer = _pointer([match.group(1)])
    if pointer is None:
        match = re.match(r"^['\"]([^'\"]+)['\"] is a required property", message)
        if match:
            pointer = _pointer([match.group(1)])
    keyword_text = str(keyword) if keyword is not None else None
    category = _classify_error(RuntimeError(base.code), message, keyword=keyword_text)
    return ValidationIssue(
        code=base.code,
        message=message,
        pointer=pointer,
        keyword=keyword_text,
        category=category,
        details={"expanded_from": base.code},
    )


@dataclass
class OpenAPIRequestAdapter:
    """Implementation of the openapi-core Request protocol."""

    host_url: str
    path: str
    method: str
    parameters: Any
    body: Optional[bytes]
    content_type: str
    path_pattern: Optional[str] = None

    @property
    def full_url_pattern(self) -> str:
        return f"{self.host_url.rstrip('/')}/{self.path.lstrip('/')}"


@dataclass
class OpenAPIResponseAdapter:
    """Implementation of the openapi-core Response protocol."""

    status_code: int
    headers: Mapping[str, Any]
    data: Optional[bytes]
    content_type: str


def _request_adapter(request: ApiRequest) -> OpenAPIRequestAdapter:
    raw_path = request.path
    parsed = urlsplit(raw_path)
    if parsed.scheme and parsed.netloc:
        path = parsed.path or "/"
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        query: dict[str, list[str]] = {}
        for key, value in query_pairs:
            query.setdefault(key, []).append(value)
        if request.query_params:
            query.update(request.query_params)
        host_url = f"{parsed.scheme}://{parsed.netloc}"
    else:
        path = raw_path.split("?", 1)[0] or "/"
        query = request.query_params
        host_url = (request.base_url or "http://mock.local").rstrip("/")
    headers = _CaseInsensitiveMap(request.headers)
    content_type = (request.content_type or headers.get("content-type", "") or "").lower()
    try:
        from openapi_core.datatypes import RequestParameters
    except ImportError as exc:  # pragma: no cover - guarded by _import_openapi
        raise OpenAPICoreUnavailable("openapi-core datatypes are unavailable") from exc
    params = RequestParameters(
        query=_MultiValueMap(query),
        header=headers,
        cookie=_MultiValueMap(),
        path={},
    )
    return OpenAPIRequestAdapter(
        host_url=host_url,
        path=path,
        method=request.method.lower(),
        parameters=params,
        body=_json_bytes(request.body),
        content_type=content_type,
        path_pattern=path,
    )


def _response_adapter(response: ApiResponse) -> OpenAPIResponseAdapter:
    headers = _CaseInsensitiveMap(response.headers)
    content_type = (response.content_type or headers.get("content-type", "") or "").lower()
    return OpenAPIResponseAdapter(
        status_code=response.status_code,
        headers=headers,
        data=_json_bytes(response.body),
        content_type=content_type,
    )


def _as_request(value: Union[ApiRequest, Mapping[str, Any]]) -> ApiRequest:
    if isinstance(value, ApiRequest):
        return value
    return ApiRequest.model_validate(value)


def _as_response(value: Union[ApiResponse, Mapping[str, Any]]) -> ApiResponse:
    if isinstance(value, ApiResponse):
        return value
    return ApiResponse.model_validate(value)


def _load_openapi(spec: Any):
    OpenAPI = _import_openapi()
    if hasattr(spec, "validate_request") and hasattr(spec, "validate_response"):
        return spec
    if isinstance(spec, Mapping):
        return OpenAPI.from_dict(dict(spec))
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(f"OpenAPI specification not found: {path}")
        # ``from_path`` is available in current openapi-core; fall back to the
        # older file-path spelling for the 0.19 release line.
        if hasattr(OpenAPI, "from_path"):
            return OpenAPI.from_path(path)
        return OpenAPI.from_file_path(str(path))
    raise TypeError("spec must be an OpenAPI object, mapping, or file path")


def _iter_errors(openapi: Any, request: Any, response: Any = None) -> list[BaseException]:
    try:
        if response is None:
            iterator = getattr(openapi, "iter_request_errors", None)
            if callable(iterator):
                return list(iterator(request))
            openapi.validate_request(request)
            return []
        iterator = getattr(openapi, "iter_response_errors", None)
        if callable(iterator):
            return list(iterator(request, response))
        openapi.validate_response(request, response)
        return []
    except BaseException as exc:  # normalize library-specific validation errors
        return [exc]


def _validation_result(
    *,
    request: ApiRequest,
    direction: str,
    errors: list[BaseException],
    source: str,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    for error in errors:
        normalized = _normalize_error(error)
        schema_errors = normalized.details.get("schema_errors", [])
        expanded_issues: list[ValidationIssue] = []
        if isinstance(schema_errors, list):
            seen: set[tuple[str, Optional[str], Optional[FaultCategory]]] = set()
            for item in schema_errors:
                expanded = _schema_item_issue(normalized, item)
                if expanded is None:
                    continue
                key = (expanded.message, expanded.pointer, expanded.category)
                if key not in seen:
                    expanded_issues.append(expanded)
                    seen.add(key)
        # Bundled openapi-core schema failures are more useful as individual
        # stable issues; retain the parent only when no child was recoverable.
        issues.extend(expanded_issues or [normalized])
    evidence: list[Evidence] = [
        Evidence(
            source=source,
            source_type=EvidenceType.OPENAPI,
            api_name=request.api_name,
            endpoint=request.path,
            method=request.method,
            path=request.path,
            version=request.api_version,
            pointer=f"/paths{request.path}",
            details={"valid": not issues},
        )
    ]
    for issue in issues:
        evidence.append(
            Evidence(
                source=source,
                source_type=EvidenceType.SCHEMA,
                api_name=request.api_name,
                endpoint=request.path,
                method=request.method,
                path=request.path,
                version=request.api_version,
                pointer=issue.pointer,
                excerpt=issue.message,
                details={"code": issue.code, "category": issue.category.value if issue.category else None},
            )
        )
    return ValidationResult(
        valid=not issues,
        direction=direction,  # type: ignore[arg-type]
        issues=issues,
        evidence=evidence,
        operation=request.path,
        path=request.path,
        method=request.method,
    )


class OpenAPIValidator:
    """Reusable adapter around one loaded OpenAPI document."""

    def __init__(self, spec: Any, *, source: Optional[str] = None):
        self.source = source or (str(spec) if isinstance(spec, (str, Path)) else "openapi_spec")
        self.openapi = _load_openapi(spec)

    def validate_request(
        self, request: Union[ApiRequest, Mapping[str, Any]]
    ) -> ValidationResult:
        model = _as_request(request)
        adapter = _request_adapter(model)
        errors = _iter_errors(self.openapi, adapter)
        return _validation_result(
            request=model,
            direction="request",
            errors=errors,
            source=self.source,
        )

    def validate_response(
        self,
        request: Union[ApiRequest, Mapping[str, Any]],
        response: Union[ApiResponse, Mapping[str, Any]],
    ) -> ValidationResult:
        request_model = _as_request(request)
        response_model = _as_response(response)
        request_adapter = _request_adapter(request_model)
        response_adapter = _response_adapter(response_model)
        errors = _iter_errors(self.openapi, request_adapter, response_adapter)
        return _validation_result(
            request=request_model,
            direction="response",
            errors=errors,
            source=self.source,
        )


def validate_request(
    request: Union[ApiRequest, Mapping[str, Any]],
    spec: Any,
    *,
    source: Optional[str] = None,
) -> ValidationResult:
    """Validate a request using openapi-core and return normalized evidence."""

    return OpenAPIValidator(spec, source=source).validate_request(request)


def validate_response(
    request: Union[ApiRequest, Mapping[str, Any]],
    response: Union[ApiResponse, Mapping[str, Any]],
    spec: Any,
    *,
    source: Optional[str] = None,
) -> ValidationResult:
    """Validate a response using the operation selected by its request."""

    return OpenAPIValidator(spec, source=source).validate_response(request, response)


__all__ = [
    "OpenAPICoreUnavailable",
    "OpenAPIRequestAdapter",
    "OpenAPIResponseAdapter",
    "OpenAPIValidator",
    "validate_request",
    "validate_response",
]
