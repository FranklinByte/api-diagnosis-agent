"""Evaluate the any-document path on the bundled order-service example.

For every faulty request this records:
- whether the deterministic tools produced the expected fault label;
- when a model is configured, whether the corrected request it wrote passes
  validation against the document, in how many rounds, with how many tokens.

The cases are hand-written mutations of one valid request.  They measure this
pipeline on one example document; they are not a claim about arbitrary APIs.
No key, prompt or request body is written to the output file.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api_diagnosis.general import run_general_diagnosis  # noqa: E402
from api_diagnosis.llm import load_config, load_dotenv  # noqa: E402
from api_diagnosis.models import ApiRequest  # noqa: E402

SPEC_ID = "example-shop-orders"
H = {"Authorization": "Bearer abc", "Idempotency-Key": "k-12345678"}
ADDRESS = {"receiver": "张三", "phone": "13800000000", "province": "浙江", "city": "杭州", "detail": "示例路1号"}
BODY = {"items": [{"sku": "A1", "quantity": 2}], "currency": "CNY", "address": ADDRESS}


def _body(**changes):
    return {**BODY, **changes}


def _without(mapping, key):
    return {name: value for name, value in mapping.items() if name != key}


# (case id, request fields, expected fault label, whether a fix can be written from the document alone)
CASES = [
    ("missing-currency", dict(method="POST", path="/orders", headers=H, body=_without(BODY, "currency"), status_code=400), "missing_required_field", True),
    ("missing-items", dict(method="POST", path="/orders", headers=H, body=_without(BODY, "items"), status_code=400), "missing_required_field", True),
    ("missing-address-city", dict(method="POST", path="/orders", headers=H, body=_body(address=_without(ADDRESS, "city")), status_code=400), "missing_required_field", True),
    ("missing-idempotency-header", dict(method="POST", path="/orders", headers={"Authorization": "Bearer abc"}, body=BODY, status_code=400), "missing_required_field", True),
    ("missing-cancel-reason", dict(method="POST", path="/orders/ORD-20261113/cancel", headers=H, body={"note": "不想要了"}, status_code=400), "missing_required_field", True),
    ("quantity-as-string", dict(method="POST", path="/orders", headers=H, body=_body(items=[{"sku": "A1", "quantity": "2"}]), status_code=400), "field_type_error", True),
    ("items-as-object", dict(method="POST", path="/orders", headers=H, body=_body(items={"sku": "A1", "quantity": 2}), status_code=400), "field_type_error", True),
    ("page-as-word", dict(method="GET", path="/orders?page=two", headers=H, status_code=400), "field_type_error", True),
    ("currency-rmb", dict(method="POST", path="/orders", headers=H, body=_body(currency="RMB"), status_code=400), "enum_invalid", True),
    ("status-done", dict(method="GET", path="/orders?status=done", headers=H, status_code=400), "enum_invalid", True),
    ("cancel-reason-free-text", dict(method="POST", path="/orders/ORD-20261113/cancel", headers=H, body={"reason": "不想要了"}, status_code=400), "enum_invalid", True),
    ("extra-top-level-field", dict(method="POST", path="/orders", headers=H, body=_body(pay_method="alipay"), status_code=400), "unknown_field", True),
    ("extra-item-field", dict(method="POST", path="/orders", headers=H, body=_body(items=[{"sku": "A1", "quantity": 2, "price": 9.9}]), status_code=400), "unknown_field", True),
    ("extra-cancel-field", dict(method="POST", path="/orders/ORD-20261113/cancel", headers=H, body={"reason": "other", "refund": True}, status_code=400), "unknown_field", True),
    ("quantity-zero", dict(method="POST", path="/orders", headers=H, body=_body(items=[{"sku": "A1", "quantity": 0}]), status_code=400), "constraint_violation", True),
    ("page-size-too-large", dict(method="GET", path="/orders?page_size=500", headers=H, status_code=400), "constraint_violation", True),
    ("phone-wrong-format", dict(method="POST", path="/orders", headers=H, body=_body(address={**ADDRESS, "phone": "0754-1234567"}), status_code=400), "constraint_violation", False),
    ("order-id-wrong-format", dict(method="GET", path="/orders/12345", headers=H, status_code=404), "constraint_violation", False),
    ("empty-items", dict(method="POST", path="/orders", headers=H, body=_body(items=[]), status_code=400), "constraint_violation", False),
    ("path-singular", dict(method="GET", path="/order/ORD-20261113", headers=H, status_code=404), "path_error", True),
    ("path-typo-with-prefix", dict(method="POST", path="https://api.shop.example.com/v1/order", headers=H, body=BODY, status_code=404), "path_error", True),
    ("delete-not-declared", dict(method="DELETE", path="/orders/ORD-20261113", headers=H, status_code=405), "method_error", True),
    ("put-instead-of-post", dict(method="PUT", path="/orders", headers=H, body=BODY, status_code=405), "method_error", True),
    ("no-credential", dict(method="GET", path="/orders", headers={}, status_code=401), "auth_error", False),
    ("several-violations", dict(method="POST", path="/orders", headers=H, body=_body(currency="RMB", address={"receiver": "张三", "phone": "13800000000"}), status_code=400), "ambiguous", True),
    ("valid-but-out-of-stock", dict(method="POST", path="/orders", headers=H, body=BODY, status_code=409, response_body={"error": "OUT_OF_STOCK", "sku": "A1"}), None, False),
]


def main() -> int:
    load_dotenv()
    config = load_config()
    rows = []
    for case_id, fields, expected, fixable in CASES:
        started = time.perf_counter()
        result = run_general_diagnosis(SPEC_ID, ApiRequest(**fields), with_explanation=config is not None)
        elapsed = time.perf_counter() - started
        report, explanation = result["report"], result["explanation"] or {}
        observed = report.fault_category.value if report.fault_category else None
        row = {
            "id": case_id,
            "expected_label": expected,
            "observed_label": observed,
            "label_correct": observed == expected,
            "status": report.status,
            "finding_count": len(result["findings"]),
            "fix_expected": fixable,
            "explanation_source": explanation.get("source"),
            "fallback_reason": explanation.get("fallback_reason"),
            "fix_offered": explanation.get("corrected_request") is not None,
            "fix_verified": explanation.get("corrected_request_verified"),
            "rounds": explanation.get("rounds"),
            "total_tokens": (explanation.get("usage") or {}).get("total_tokens"),
            "latency_seconds": round(elapsed, 2),
        }
        rows.append(row)
        print(f"{case_id}: label={'ok' if row['label_correct'] else 'WRONG ' + str(observed)} source={row['explanation_source']} fix_verified={row['fix_verified']} rounds={row['rounds']} {row['latency_seconds']}s", flush=True)

    modelled = [row for row in rows if row["explanation_source"] == "model"]
    offered = [row for row in modelled if row["fix_offered"]]
    tokens = [row["total_tokens"] for row in modelled if row["total_tokens"]]
    latencies = sorted(row["latency_seconds"] for row in modelled)
    summary = {
        "dataset": "shop-orders hand-written mutations",
        "spec_id": SPEC_ID,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.model if config else None,
        "case_count": len(rows),
        "label_correct": sum(row["label_correct"] for row in rows),
        "model_explanations": len(modelled),
        "template_fallbacks": sum(row["explanation_source"] == "template" for row in rows),
        "fixes_offered": len(offered),
        "fixes_verified": sum(row["fix_verified"] is True for row in offered),
        "fixes_verified_first_round": sum(row["fix_verified"] is True and row["rounds"] == 1 for row in offered),
        "fixes_failed_validation": sum(row["fix_verified"] is False for row in offered),
        "total_tokens": sum(tokens),
        "mean_tokens_per_case": round(statistics.mean(tokens)) if tokens else None,
        "mean_latency_seconds": round(statistics.mean(latencies), 2) if latencies else None,
        "p50_latency_seconds": statistics.median(latencies) if latencies else None,
        "max_latency_seconds": latencies[-1] if latencies else None,
    }
    output = Path(os.getenv("GENERAL_EVAL_OUTPUT", ROOT / "eval-results" / "general-2026-11-13" / "shop-orders.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["label_correct"] == summary["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
