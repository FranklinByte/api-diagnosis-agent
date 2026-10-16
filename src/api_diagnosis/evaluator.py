"""Offline benchmark and explicitly bounded baseline comparison.

The dataset contains authored request/response observations.  The benchmark
does not claim that a deterministic wrapper is an independent Agent, so every
prediction records its tools, citations, loop count, and whether it is derived
from the deterministic oracle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .diagnosis import diagnose_v0
from .paths import project_root


ROOT = project_root()
DEFAULT_DATASET = ROOT / "data" / "ground_truth.json"


@dataclass(frozen=True)
class Prediction:
    """A prediction plus provenance needed to audit the benchmark claim."""

    status: str
    category: Optional[str]
    reason: str = ""
    tools_used: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    rounds: int = 0
    oracle_derived: bool = False
    source: str = "unspecified"


def _fingerprint(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def load_cases(path: str | Path = DEFAULT_DATASET) -> list[Mapping[str, Any]]:
    """Load authored cases and reject the old repetition multiplier format."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = value.get("cases") if isinstance(value, Mapping) else None
    if not isinstance(cases, list):
        raise ValueError("ground truth must contain a cases array")
    if isinstance(value, Mapping) and "repeat_cases" in value:
        raise ValueError("repeat_cases is forbidden; author independent cases instead")
    contracts = value.get("label_contracts") if isinstance(value, Mapping) else None
    if not isinstance(contracts, Mapping):
        raise ValueError("ground truth must contain label_contracts")

    loaded: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    seen_requests: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("each ground-truth case must be an object")
        case_id = case.get("id")
        request = case.get("request")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("each ground-truth case needs a non-empty id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate ground-truth id: {case_id}")
        if not isinstance(request, Mapping):
            raise ValueError(f"case {case_id} needs a request object")
        request_key = _fingerprint(request)
        if request_key in seen_requests:
            raise ValueError(f"duplicate request input: {case_id}")
        seen_ids.add(case_id)
        seen_requests.add(request_key)
        label = "no_fault" if case.get("expected_category") is None else str(case.get("expected_category"))
        contract = contracts.get(label)
        if not isinstance(contract, Mapping):
            raise ValueError(f"case {case_id} has no label contract for {label}")
        expected_tool = contract.get("expected_primary_tool")
        if not isinstance(expected_tool, str) or not expected_tool:
            raise ValueError(f"label contract {label} needs expected_primary_tool")
        enriched = dict(case)
        enriched["expected_primary_tool"] = expected_tool
        enriched["expected_evidence_source"] = contract.get("evidence_source")
        enriched["expected_needs_manual"] = bool(contract.get("needs_manual", False))
        enriched["repair_principle"] = contract.get("repair_principle")
        loaded.append(enriched)
    return loaded


def _body(request: Mapping[str, Any]) -> Mapping[str, Any]:
    value = request.get("body", request.get("request_body", {}))
    return value if isinstance(value, Mapping) else {}


def _header(request: Mapping[str, Any], name: str) -> Optional[str]:
    headers = request.get("headers", request.get("sanitized_headers", {}))
    if not isinstance(headers, Mapping):
        return None
    for key, value in headers.items():
        if str(key).lower() == name.lower():
            return str(value)
    return None


def heuristic_baseline(request: Mapping[str, Any], *, context: str = "") -> Prediction:
    """Contract-rule heuristic used only as an explicitly named proxy.

    It is intentionally not called RAG: it inspects the fixture's request
    shape directly and does not execute OpenAPI validation.
    """

    del context
    path = str(request.get("path", ""))
    method = str(request.get("method", "")).upper()
    body = _body(request)
    version = _header(request, "x-api-version")
    status_code = request.get("status_code")
    error_message = str(request.get("error_message", "")).lower()
    response_body = request.get("response_body")
    if status_code in {408, 504} or "timeout" in error_message:
        return Prediction("confirmed", "timeout", "observed timeout", source="contract_rule_heuristic")
    if status_code in {401, 403}:
        return Prediction("confirmed", "auth_error", "observed auth rejection", source="contract_rule_heuristic")
    if status_code == 415:
        return Prediction("confirmed", "content_type_error", "observed media type rejection", source="contract_rule_heuristic")
    if status_code == 429:
        return Prediction("confirmed", "rate_limited", "observed rate limit", source="contract_rule_heuristic")
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return Prediction("confirmed", "server_error", "observed server error", source="contract_rule_heuristic")
    if status_code == 200 and response_body is not None and isinstance(response_body, Mapping):
        forecast = response_body.get("forecast")
        if (
            not isinstance(forecast, list)
            or not forecast
            or not isinstance(forecast[0], Mapping)
            or not isinstance(forecast[0].get("temperature"), (int, float))
            or forecast[0].get("condition") != "clear"
        ):
            return Prediction("confirmed", "response_schema_change", "observed response schema mismatch", source="contract_rule_heuristic")
    if path != "/weather/forecast":
        return Prediction("confirmed", "path_error", "path mismatch", source="contract_rule_heuristic")
    if method != "POST":
        return Prediction("confirmed", "method_error", "method mismatch", source="contract_rule_heuristic")
    if version and version != "v2":
        return Prediction(
            "confirmed" if version == "v1" else "unsupported",
            "version_incompatible" if version == "v1" else "unsupported",
            "version mismatch",
            source="contract_rule_heuristic",
        )
    required = {"city", "units", "days"}
    missing = required - set(body)
    unknown = set(body) - required
    schema_categories: list[str] = []
    if missing or version is None:
        schema_categories.append("missing_required_field")
    if unknown:
        schema_categories.append("unknown_field")
    if "days" in body and not isinstance(body["days"], int):
        schema_categories.append("field_type_error")
    if "units" in body and (
        not isinstance(body["units"], str) or body["units"] not in {"metric", "imperial"}
    ):
        schema_categories.append("enum_invalid")
    if len(set(schema_categories)) > 1:
        return Prediction("ambiguous", "ambiguous", "multiple heuristic schema signals", source="contract_rule_heuristic")
    if schema_categories:
        return Prediction("confirmed", schema_categories[0], "schema signal", source="contract_rule_heuristic")
    return Prediction("confirmed", None, "no heuristic fault", source="contract_rule_heuristic")


def _http_signal_prediction(request: Mapping[str, Any]) -> Prediction:
    """No-tool proxy based only on conspicuous HTTP/text signals."""

    path = str(request.get("path", ""))
    method = str(request.get("method", "")).upper()
    status_code = request.get("status_code")
    message = str(request.get("error_message", "")).lower()
    version = _header(request, "x-api-version")
    if path != "/weather/forecast":
        return Prediction("confirmed", "path_error", "obvious endpoint mismatch")
    if method != "POST":
        return Prediction("confirmed", "method_error", "obvious method mismatch")
    if status_code in {408, 504} or "timeout" in message:
        return Prediction("confirmed", "timeout", "timeout text/status")
    if status_code in {401, 403}:
        return Prediction("confirmed", "auth_error", "auth status")
    if status_code == 415:
        return Prediction("confirmed", "content_type_error", "media status")
    if status_code == 429:
        return Prediction("confirmed", "rate_limited", "rate-limit status")
    if isinstance(status_code, int) and status_code >= 500:
        return Prediction("confirmed", "server_error", "server status")
    if version and version != "v2":
        return Prediction(
            "confirmed" if version == "v1" else "unsupported",
            "version_incompatible" if version == "v1" else "unsupported",
            "version text",
        )
    return Prediction("confirmed", None, "no contract tools available")


def http_signal_heuristic_baseline(request: Mapping[str, Any]) -> Prediction:
    return replace(_http_signal_prediction(request), source="http_signal_heuristic")


def no_tool_signal_baseline(request: Mapping[str, Any]) -> Prediction:
    """Public descriptive name for the no-tool baseline."""

    return http_signal_heuristic_baseline(request)


def contract_rule_heuristic_baseline(request: Mapping[str, Any]) -> Prediction:
    return heuristic_baseline(request)


def retrieval_context_heuristic_baseline(request: Mapping[str, Any]) -> Prediction:
    """Retrieve evidence, then apply the transparent contract-rule heuristic.

    This measures a retrieval-context wrapper, not a pure RAG model and not an
    OpenAPI validator.  The metadata makes that limitation visible in results.
    """

    from .retrieval import build_retriever

    query = json.dumps(request, ensure_ascii=False, sort_keys=True)
    documents = build_retriever(default_k=4).search(query, k=4)
    citations = tuple(
        str(document.metadata.get("pointer") or document.metadata.get("source"))
        for document in documents
    )
    prediction = contract_rule_heuristic_baseline(request)
    return replace(
        prediction,
        tools_used=("retrieve_contract",),
        citations=citations,
        rounds=1,
        source="retrieval_context_plus_contract_heuristic",
    )


# Backward-compatible import name; benchmark output uses the honest name above.
llm_like_baseline = no_tool_signal_baseline
rag_like_baseline = retrieval_context_heuristic_baseline


def _report_prediction(
    report: Any,
    *,
    tools_used: tuple[str, ...],
    rounds: int,
    source: str,
    oracle_derived: bool,
) -> Prediction:
    category = getattr(report, "fault_category", None)
    category = getattr(category, "value", category)
    citations = tuple(
        str(getattr(evidence, "pointer", ""))
        for evidence in getattr(report, "evidence", [])
        if getattr(evidence, "pointer", None)
    )
    return Prediction(
        status=str(getattr(report, "status", "unsupported")),
        category=category,
        reason=str(getattr(report, "root_cause", "")),
        tools_used=tools_used,
        citations=citations,
        rounds=rounds,
        oracle_derived=oracle_derived,
        source=source,
    )


def _agent_predict(request: Mapping[str, Any]) -> Prediction:
    try:
        from .agent import run_agent_state
        from .models import DiagnosisResult

        state = run_agent_state(request)
        report = DiagnosisResult.model_validate(state["report"])
        events = state.get("trace", [])
        tools = tuple(dict.fromkeys(str(event["tool"]) for event in events if event.get("tool")))
        raw_rounds = state.get("iteration_count")
        if not isinstance(raw_rounds, int):
            raw_rounds = sum(
                1 for event in events if event.get("node") == "execute_tool"
            )
        rounds = raw_rounds
        rounds = max(1, rounds) if report.status else rounds
        return _report_prediction(
            report,
            tools_used=tools,
            rounds=rounds,
            source="langgraph_wrapper_over_deterministic_tool",
            oracle_derived=True,
        )
    except (ImportError, RuntimeError, ValueError, TypeError, KeyError):
        # This fallback is explicit in the result metadata; it is not presented
        # as an independent Agent measurement.
        return _oracle_predict(request, source="deterministic_oracle_fallback")


def _oracle_predict(request: Mapping[str, Any], *, source: str = "deterministic_oracle_reference") -> Prediction:
    report = diagnose_v0(request)
    tools = ("deterministic_diagnose",)
    return _report_prediction(
        report,
        tools_used=tools,
        rounds=1,
        source=source,
        oracle_derived=True,
    )


def deterministic_oracle_reference(request: Mapping[str, Any]) -> Prediction:
    """Public reference predictor; never interpret it as an Agent result."""

    return _oracle_predict(request)


def predictor_for(name: str) -> Callable[[Mapping[str, Any]], Prediction]:
    normalized = name.lower()
    if normalized in {"http_signal_heuristic", "no_tool_signal", "llm", "llm_baseline"}:
        return no_tool_signal_baseline
    if normalized in {"retrieval_context_heuristic", "rag", "rag_baseline"}:
        return retrieval_context_heuristic_baseline
    if normalized in {"langgraph_oracle_wrapper", "agent", "agent_baseline"}:
        return _agent_predict
    if normalized in {"deterministic_oracle_reference", "oracle"}:
        return deterministic_oracle_reference
    raise ValueError(f"unknown baseline: {name}")


def _macro_accuracy(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        label = row.get(field)
        key = "__none__" if label is None else str(label)
        grouped.setdefault(key, []).append(bool(row["category_correct"] if field == "expected_category" else row["status_correct"]))
    if not grouped:
        return 0.0
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def evaluate(
    cases: Iterable[Mapping[str, Any]],
    predictor: Callable[[Mapping[str, Any]], Prediction],
    *,
    baseline: str = "custom",
) -> dict[str, Any]:
    case_list = list(cases)
    rows: list[dict[str, Any]] = []
    for case in case_list:
        request = case.get("request", {})
        prediction = predictor(request)
        expected_status = case.get("expected_status")
        expected_category = case.get("expected_category")
        expected_tool = case.get("expected_primary_tool")
        rows.append(
            {
                "id": case.get("id"),
                "expected_status": expected_status,
                "predicted_status": prediction.status,
                "expected_category": expected_category,
                "predicted_category": prediction.category,
                "status_correct": prediction.status == expected_status,
                "category_correct": prediction.category == expected_category,
                "expected_primary_tool": expected_tool,
                "tool_selection_correct": expected_tool in prediction.tools_used,
                "reason": prediction.reason,
                "tools_used": list(prediction.tools_used),
                "citations": list(prediction.citations),
                "rounds": prediction.rounds,
                "oracle_derived": prediction.oracle_derived,
                "prediction_source": prediction.source,
            }
        )
    total = len(rows)
    source_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    for row in rows:
        source = str(row["prediction_source"])
        source_counts[source] = source_counts.get(source, 0) + 1
        for tool in row["tools_used"]:
            tool_counts[str(tool)] = tool_counts.get(str(tool), 0) + 1
    return {
        "baseline": baseline,
        "total": total,
        "unique_input_count": len({_fingerprint(case.get("request", {})) for case in case_list}),
        "status_accuracy": sum(row["status_correct"] for row in rows) / total if total else 0.0,
        "category_accuracy": sum(row["category_correct"] for row in rows) / total if total else 0.0,
        "category_macro_accuracy": _macro_accuracy(rows, "expected_category"),
        "correct": sum(row["category_correct"] for row in rows),
        "tool_call_rate": sum(bool(row["tools_used"]) for row in rows) / total if total else 0.0,
        "tool_selection_accuracy": sum(bool(row["tool_selection_correct"]) for row in rows) / total if total else 0.0,
        "citation_coverage": sum(bool(row["citations"]) for row in rows) / total if total else 0.0,
        "mean_rounds": sum(float(row["rounds"]) for row in rows) / total if total else 0.0,
        "oracle_derived_rate": sum(bool(row["oracle_derived"]) for row in rows) / total if total else 0.0,
        "oracle_fallback_rate": source_counts.get("deterministic_oracle_fallback", 0) / total if total else 0.0,
        "prediction_source_counts": source_counts,
        "tool_usage_counts": tool_counts,
        "cases": rows,
    }


def run_benchmark(path: str | Path = DEFAULT_DATASET) -> dict[str, Any]:
    cases = load_cases(path)
    return {
        name: evaluate(cases, predictor_for(name), baseline=name)
        for name in (
            "http_signal_heuristic",
            "retrieval_context_heuristic",
            "langgraph_oracle_wrapper",
            "deterministic_oracle_reference",
        )
    }


__all__ = [
    "DEFAULT_DATASET",
    "Prediction",
    "contract_rule_heuristic_baseline",
    "deterministic_oracle_reference",
    "evaluate",
    "heuristic_baseline",
    "http_signal_heuristic_baseline",
    "llm_like_baseline",
    "load_cases",
    "no_tool_signal_baseline",
    "predictor_for",
    "rag_like_baseline",
    "retrieval_context_heuristic_baseline",
    "run_benchmark",
]
