from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from openapi_spec_validator import validate

from api_diagnosis.diagnosis import diagnose_v0
from api_diagnosis.models import ApiRequest, ApiResponse, FaultCategory, ReplayPolicy, ReplayRequest
from api_diagnosis.mock_api import app
from api_diagnosis.replay import MockRequestReplayer
from api_diagnosis.validation import OpenAPIValidator
from api_diagnosis.versioning import VersionChecker


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "specs" / "mock-api.json"


def request(body: dict, **overrides):
    payload = {
        "request_id": "case-001",
        "base_url": "http://127.0.0.1:8001",
        "method": "POST",
        "path": "/weather/forecast",
        "headers": {"X-API-Version": "v2"},
        "content_type": "application/json",
        "body": body,
    }
    payload.update(overrides)
    return payload


def test_mock_api_success_and_deterministic_errors():
    client = TestClient(app)
    body = {"city": "Singapore", "units": "metric", "days": 3}
    first = client.post("/weather/forecast", headers={"X-API-Version": "v2"}, json=body)
    second = client.post("/weather/forecast", headers={"X-API-Version": "v2"}, json=body)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert client.post("/weather/forecast", headers={"X-API-Version": "v2"}, json={"city": "Singapore"}).status_code == 422
    assert client.post("/weather/forecast", headers={"X-API-Version": "v2"}, json={**body, "dayz": 3}).status_code == 422
    stale = client.post("/weather/forecast", headers={"X-API-Version": "v1"}, json=body)
    assert stale.status_code == 400
    assert stale.json()["detail"]["changelog_id"] == "weather-v2-001"
    assert client.get("/weather/forecast").status_code == 405
    assert client.post("/weather/unknown", headers={"X-API-Version": "v2"}, json=body).status_code == 404


def test_static_spec_is_valid():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert validate(spec) is None


def test_openapi_core_normalizes_v0_schema_faults():
    validator = OpenAPIValidator(SPEC_PATH)
    cases = [
        ({"city": "Singapore", "units": "metric"}, FaultCategory.MISSING_REQUIRED_FIELD, "/days"),
        ({"city": "Singapore", "units": "metric", "days": "3"}, FaultCategory.FIELD_TYPE_ERROR, "/days"),
        ({"city": "Singapore", "units": "bad", "days": 3}, FaultCategory.ENUM_INVALID, "/units"),
        ({"city": "Singapore", "units": "metric", "days": 3, "dayz": 1}, FaultCategory.UNKNOWN_FIELD, "/dayz"),
    ]
    for body, category, pointer in cases:
        result = validator.validate_request(ApiRequest(**request(body)))
        assert not result.valid
        assert result.issues[0].category == category
        assert result.issues[0].pointer == pointer

    valid = validator.validate_request(ApiRequest(**request({"city": "Singapore", "units": "metric", "days": 3})))
    assert valid.valid
    response = ApiResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        body={"api_version": "v2", "city": "Singapore", "units": "metric", "days": 3, "forecast": [{"day": 1, "temperature": 24.0, "condition": "clear"}]},
    )
    assert validator.validate_response(ApiRequest(**request({"city": "Singapore", "units": "metric", "days": 3})), response).valid


def test_diagnosis_localizes_faults_without_false_positive():
    cases = [
        (request({"city": "Singapore", "units": "metric", "days": 3}), None),
        (request({"city": "Singapore", "units": "metric"}), FaultCategory.MISSING_REQUIRED_FIELD),
        (request({"city": "Singapore", "units": "bad", "days": 3}), FaultCategory.ENUM_INVALID),
        (request({"city": "Singapore", "units": "metric", "days": 3, "dayz": 1}), FaultCategory.UNKNOWN_FIELD),
        (request({}, method="GET"), FaultCategory.METHOD_ERROR),
        (request({}, path="/weather/unknown"), FaultCategory.PATH_ERROR),
        (request({"city": "Singapore", "units": "metric", "days": 3}, headers={"X-API-Version": "v1"}), FaultCategory.VERSION_INCOMPATIBLE),
    ]
    for payload, expected in cases:
        report = diagnose_v0(payload, spec=SPEC_PATH)
        assert report.status == "confirmed"
        assert report.fault_category == expected


def test_changelog_checker_requires_explicit_evidence():
    checker = VersionChecker(ROOT / "data" / "changelog.json")
    incompatible = checker.check("v1", current_version="v2", endpoint="/weather/forecast", method="POST")
    assert incompatible.status == "incompatible"
    assert incompatible.changes
    assert checker.check("v2", current_version="v2").status == "compatible"
    assert checker.check("v9", current_version="v2").status == "unknown"


def test_replay_is_bounded_and_redacts_sensitive_headers():
    class Response:
        status = 200
        headers = {"set-cookie": "secret", "x-trace": "abc", "content-type": "application/json"}
        payload = b'{"ok": true}'

        def read(self, limit=-1):
            return self.payload if limit < 0 else self.payload[:limit]

    class Opener:
        def open(self, incoming, timeout):
            assert incoming.headers.get("X-api-key") is None
            return Response()

    policy = ReplayPolicy(allowed_urls=["http://127.0.0.1:8000"], allowed_methods={"POST"})
    replayer = MockRequestReplayer(policy, opener=Opener())
    result = replayer.replay(
        ReplayRequest(
            url="http://127.0.0.1:8000/weather/forecast",
            method="POST",
            headers={"X-API-Key": "secret", "Content-Type": "application/json"},
            body={"days": 3},
            approved=True,
        )
    )
    assert result.status == "success"
    assert result.response_body == {"ok": True}
    assert result.sanitized_request_headers["X-API-Key"] == "[REDACTED]"
    assert result.response_headers["set-cookie"] == "[REDACTED]"
    blocked = replayer.replay(ReplayRequest(url="https://evil.example", method="POST", approved=True))
    assert blocked.status == "blocked"
