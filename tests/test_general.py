"""Any-document diagnosis path: spec store, path matching, tool-checked fixes."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from api_diagnosis import general, spec_store
from api_diagnosis.api import app
from api_diagnosis.llm import LLMConfig
from api_diagnosis.models import ApiRequest, FaultCategory

SHOP = "example-shop-orders"
HEADERS = {"Authorization": "Bearer abc", "Idempotency-Key": "k-12345678"}
GOOD_BODY = {
    "items": [{"sku": "A1", "quantity": 2}],
    "currency": "CNY",
    "address": {"receiver": "张三", "phone": "13800000000", "province": "浙江", "city": "杭州", "detail": "示例路1号"},
}
CONFIG = LLMConfig(base_url="https://model.invalid/v1", api_key="test-key", model="test-model", timeout_seconds=5)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("API_DIAGNOSIS_SPEC_DIR", str(tmp_path / "specs"))
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _diagnose(**request):
    return general.run_general_diagnosis(SHOP, ApiRequest(**request), with_explanation=False)


@pytest.mark.parametrize(
    ("request_fields", "category"),
    [
        (dict(method="POST", path="/orders", headers=HEADERS, body={**GOOD_BODY, "items": [{"sku": "A1", "quantity": "2"}]}), FaultCategory.FIELD_TYPE_ERROR),
        (dict(method="POST", path="/orders", headers={"Authorization": "Bearer abc"}, body=GOOD_BODY), FaultCategory.MISSING_REQUIRED_FIELD),
        (dict(method="POST", path="/orders/ORD-20261113/cancel", headers=HEADERS, body={"reason": "other", "refund": True}), FaultCategory.UNKNOWN_FIELD),
        (dict(method="GET", path="/orders?status=done", headers=HEADERS), FaultCategory.ENUM_INVALID),
        (dict(method="GET", path="/orders/12345", headers=HEADERS), FaultCategory.CONSTRAINT_VIOLATION),
        (dict(method="GET", path="/order/ORD-20261113", headers=HEADERS), FaultCategory.PATH_ERROR),
        (dict(method="DELETE", path="/orders/ORD-20261113", headers=HEADERS), FaultCategory.METHOD_ERROR),
        (dict(method="GET", path="/orders", headers={}), FaultCategory.AUTH_ERROR),
        (dict(method="GET", path="/orders", headers=HEADERS, status_code=429), FaultCategory.RATE_LIMITED),
    ],
)
def test_faults_are_labelled_on_a_second_unrelated_document(request_fields, category):
    result = _diagnose(**request_fields)
    assert result["report"].status == "confirmed"
    assert result["report"].fault_category == category


def test_full_url_server_prefix_and_templated_path_are_matched():
    for path in (
        "https://api.shop.example.com/v1/orders/ORD-20261113",
        "/v1/orders/ORD-20261113",
        "https://sandbox.other-host.test/v1/orders/ORD-20261113/",
    ):
        result = _diagnose(method="GET", path=path, headers=HEADERS)
        assert result["spec"]["matched_path"] == "/orders/{order_id}"
        assert result["report"].fault_category is None


def test_every_violation_is_listed_not_only_the_first():
    body = {**GOOD_BODY, "currency": "RMB", "address": {"receiver": "张三", "phone": "13800000000"}}
    result = _diagnose(method="POST", path="/orders", headers=HEADERS, body=body, status_code=400)
    assert result["report"].fault_category == FaultCategory.AMBIGUOUS
    assert result["report"].status == "confirmed"
    assert [item["category"] for item in result["findings"]] == ["enum_invalid"] + ["missing_required_field"] * 3


def test_valid_request_that_still_failed_is_not_reported_as_no_fault():
    result = _diagnose(
        method="POST", path="/orders", headers=HEADERS, body=GOOD_BODY, status_code=409, response_body={"error": "OUT_OF_STOCK"}
    )
    assert result["report"].status == "unconfirmed"
    assert result["report"].needs_manual is True
    assert result["findings"] == []


def test_redacted_credential_counts_as_present():
    result = _diagnose(method="GET", path="/orders", headers={"Authorization": "[REDACTED]"})
    assert result["report"].fault_category is None


def test_builtin_contract_keeps_its_version_check():
    request = ApiRequest(method="POST", path="/weather/forecast", headers={"X-API-Version": "v1"}, body={"city": "Singapore"})
    result = general.run_general_diagnosis(spec_store.BUILTIN_SPEC_ID, request, with_explanation=False)
    assert result["report"].fault_category == FaultCategory.VERSION_INCOMPATIBLE


def _model_transport(answers, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        content = json.dumps(answers[min(len(seen), len(answers)) - 1], ensure_ascii=False)
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": usage})

    return httpx.MockTransport(handler)


def _explain(request, answers, seen):
    _, document = spec_store.load_spec(SHOP)
    report, context = general.diagnose_with_spec(request, document)
    return general.explain(request, document, report, context, config=CONFIG, transport=_model_transport(answers, seen))


def _answer(currency, headers):
    return {
        "summary": "币种不对",
        "cause": "currency 不在允许值内",
        "fix_steps": ["改币种"],
        "guesses": [],
        "corrected_request": {"method": "POST", "path": "/orders", "headers": headers, "body": {**GOOD_BODY, "currency": currency}},
    }


def test_model_fix_is_revalidated_and_errors_are_sent_back_once():
    request = ApiRequest(
        method="POST",
        path="/orders",
        headers={"Authorization": "Bearer secret-token", "Idempotency-Key": "k-12345678"},
        body={**GOOD_BODY, "currency": "RMB"},
        status_code=400,
    )
    redacted = {"Authorization": "[REDACTED]", "Idempotency-Key": "k-12345678"}
    seen: list[dict] = []
    result = _explain(request, [_answer("EUR", redacted), _answer("CNY", redacted)], seen)
    assert result["source"] == "model" and result["rounds"] == 2
    assert result["corrected_request_verified"] is True
    assert result["usage"]["total_tokens"] == 30
    assert "EUR" in seen[1]["messages"][-1]["content"]
    # The credential never reaches the model.
    assert "secret-token" not in json.dumps(seen)


def test_unfixable_model_answer_is_flagged_not_trusted():
    request = ApiRequest(method="POST", path="/orders", headers=HEADERS, body={**GOOD_BODY, "currency": "RMB"})
    seen: list[dict] = []
    result = _explain(request, [_answer("EUR", HEADERS)], seen)
    assert len(seen) == general.MAX_EXPLAIN_ROUNDS
    assert result["corrected_request_verified"] is False
    assert result["corrected_request_issues"][0]["category"] == "enum_invalid"


def test_model_failure_falls_back_to_template_and_keeps_tool_label():
    request = ApiRequest(method="GET", path="/orders/12345", headers=HEADERS)
    _, document = spec_store.load_spec(SHOP)
    report, context = general.diagnose_with_spec(request, document)
    transport = httpx.MockTransport(lambda _: httpx.Response(500, json={}))
    result = general.explain(request, document, report, context, config=CONFIG, transport=transport)
    assert result["source"] == "template" and "500" in result["fallback_reason"]
    assert report.fault_category == FaultCategory.CONSTRAINT_VIOLATION


def test_upload_list_and_diagnose_through_http():
    client = TestClient(app)
    operation = {
        "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
        "responses": {"200": {"description": "ok"}},
    }
    document = {"openapi": "3.0.3", "info": {"title": "Tickets", "version": "2"}, "paths": {"/tickets/{id}": {"get": operation}}}
    created = client.post("/specs", json={"content": json.dumps(document), "name": "工单系统"})
    assert created.status_code == 201
    spec_id = created.json()["spec_id"]
    assert any(item["spec_id"] == spec_id and item["kind"] == "uploaded" for item in client.get("/specs").json())
    assert client.get(f"/specs/{spec_id}/operations").json() == [{"method": "GET", "path": "/tickets/{id}", "summary": ""}]
    response = client.post("/v1/diagnose", json={"spec_id": spec_id, "request": {"method": "GET", "path": "/tickets/abc"}})
    assert response.status_code == 200
    body = response.json()
    assert body["report"]["fault_category"] == "field_type_error"
    assert body["explanation"]["source"] == "template"
    assert client.get("/").status_code == 200
    assert client.get("/v1/status").json() == {"model_configured": False, "model": None}


@pytest.mark.parametrize(
    "content",
    [
        "   ",
        "not: [valid",
        json.dumps({"swagger": "2.0", "paths": {"/a": {}}}),
        json.dumps({"openapi": "3.0.3", "info": {"title": "x", "version": "1"}, "paths": {}}),
        json.dumps({"openapi": "3.0.3", "paths": {"/a": {"get": {}}}}),
    ],
)
def test_unusable_documents_are_rejected(content):
    client = TestClient(app)
    assert client.post("/specs", json={"content": content}).status_code == 422
    assert [item["kind"] for item in client.get("/specs").json()] == ["builtin", "example"]


def test_unknown_or_crafted_spec_id_is_not_found():
    client = TestClient(app)
    for spec_id in ("deadbeefdeadbeef", "../../secret", "example-missing"):
        response = client.post("/v1/diagnose", json={"spec_id": spec_id, "request": {"method": "GET", "path": "/a"}})
        assert response.status_code == 404
