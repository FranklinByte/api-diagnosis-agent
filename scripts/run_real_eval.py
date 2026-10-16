"""Evaluate true direct, RAG, and tool-using provider paths.

Credentials are read only from environment variables. Result files contain
labels, provenance, timings, and token counts when the provider reports usage;
they never serialize prompts, request bodies, or credentials.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from langchain_core.callbacks import UsageMetadataCallbackHandler

from api_diagnosis.agent import (
    provider_chat_model,
    run_provider_agent,
    run_provider_baseline,
)
from api_diagnosis.evaluator import DEFAULT_DATASET, load_cases


def _stratified_cases(cases: list[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    if limit <= 0 or limit >= len(cases):
        return cases
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[str(case.get("expected_category"))].append(case)
    selected: list[Mapping[str, Any]] = []
    while len(selected) < limit and any(groups.values()):
        for label in sorted(groups):
            if groups[label] and len(selected) < limit:
                selected.append(groups[label].pop(0))
    return selected


def _usage_totals(value: Mapping[str, Any]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in value.values():
        if not isinstance(usage, Mapping):
            continue
        for key in totals:
            raw = usage.get(key, 0)
            if isinstance(raw, int):
                totals[key] += raw
    return totals


def _usage_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    return {key: max(0, after.get(key, 0) - before.get(key, 0)) for key in after}


def _run_mode(
    mode: str,
    cases: list[Mapping[str, Any]],
    *,
    model: Any,
    usage_handler: UsageMetadataCallbackHandler,
) -> dict[str, Any]:
    rows: list[dict[str, object]] = []
    guardrail = os.getenv("REAL_EVAL_GUARDRAIL", "0").lower() in {"1", "true", "yes"}
    for index, case in enumerate(cases, start=1):
        before = _usage_totals(usage_handler.usage_metadata)
        started = time.perf_counter()
        try:
            if mode == "agent":
                report = run_provider_agent(case.get("request", {}), model=model, guardrail=guardrail)
            else:
                report = run_provider_baseline(case.get("request", {}), mode=mode, model=model)
            predicted_status = report.status
            predicted_category = report.fault_category.value if report.fault_category else None
            execution_metadata = report.execution_metadata
            error_type = None
        except Exception as exc:  # keep the batch running without serializing error text
            predicted_status, predicted_category = "error", None
            execution_metadata = {}
            error_type = type(exc).__name__
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        usage = _usage_delta(before, _usage_totals(usage_handler.usage_metadata))
        rows.append(
            {
                "id": case.get("id"),
                "expected_status": case.get("expected_status"),
                "predicted_status": predicted_status,
                "expected_category": case.get("expected_category"),
                "predicted_category": predicted_category,
                "status_correct": predicted_status == case.get("expected_status"),
                "category_correct": predicted_category == case.get("expected_category"),
                "expected_primary_tool": case.get("expected_primary_tool"),
                "tool_selection_correct": (
                    case.get("expected_primary_tool")
                    in execution_metadata.get("tool_history", [])
                    if mode == "agent"
                    else False
                ),
                "elapsed_ms": elapsed_ms,
                "usage": usage,
                "execution_metadata": execution_metadata,
                "deterministic_fallback": execution_metadata.get("synthesis_source") == "deterministic_fallback",
                "error_type": error_type,
            }
        )
        print(f"{mode} {index}/{len(cases)} {case.get('id')} {elapsed_ms}ms", flush=True)
    total = len(rows)
    latencies = sorted(float(row["elapsed_ms"]) for row in rows)
    token_totals = {
        key: sum(int(row["usage"][key]) for row in rows)  # type: ignore[index]
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    return {
        "mode": mode,
        "guardrail": guardrail if mode == "agent" else False,
        "total": total,
        "status_accuracy": sum(bool(row["status_correct"]) for row in rows) / total if total else 0.0,
        "category_accuracy": sum(bool(row["category_correct"]) for row in rows) / total if total else 0.0,
        "accuracy_scope": "system_including_deterministic_fallback" if mode == "agent" else "provider_baseline",
        "deterministic_fallback_count": sum(bool(row["deterministic_fallback"]) for row in rows),
        "deterministic_fallback_rate": (
            sum(bool(row["deterministic_fallback"]) for row in rows) / total if total else 0.0
        ),
        "mean_latency_ms": round(sum(latencies) / total, 2) if total else 0.0,
        "p50_latency_ms": latencies[int((total - 1) * 0.50)] if total else 0.0,
        "p95_latency_ms": latencies[int((total - 1) * 0.95)] if total else 0.0,
        "token_usage": token_totals,
        "tool_selection_accuracy": (
            sum(bool(row["tool_selection_correct"]) for row in rows) / total
            if mode == "agent" and total
            else None
        ),
        "provider_tool_call_rate": (
            sum(
                bool(row["execution_metadata"].get("provider_tool_call_used"))  # type: ignore[union-attr]
                for row in rows
            )
            / total
            if mode == "agent" and total
            else None
        ),
        "errors": sum(row["error_type"] is not None for row in rows),
        "cases": rows,
    }


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    dataset_path = Path(os.getenv("REAL_EVAL_DATASET", str(DEFAULT_DATASET)))
    cases = list(load_cases(dataset_path))
    cases = _stratified_cases(cases, int(os.getenv("REAL_EVAL_LIMIT", "0")))
    modes = [
        mode.strip().casefold()
        for mode in os.getenv("REAL_EVAL_MODES", "direct,rag,agent").split(",")
        if mode.strip()
    ]
    invalid = set(modes) - {"direct", "rag", "agent"}
    if invalid:
        raise SystemExit(f"unsupported REAL_EVAL_MODES: {sorted(invalid)}")

    usage_handler = UsageMetadataCallbackHandler()
    model = provider_chat_model().with_config(callbacks=[usage_handler])
    runs = {
        mode: _run_mode(mode, cases, model=model, usage_handler=usage_handler)
        for mode in modes
    }
    result = {
        "provider": {
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "model": os.getenv("OPENAI_MODEL"),
            "structured_output_mode": os.getenv("OPENAI_STRUCTURED_OUTPUT", "1"),
        },
        "dataset": dataset_path.as_posix(),
        "dataset_case_count": len(load_cases(dataset_path)),
        "evaluated_case_count": len(cases),
        "case_ids": [case.get("id") for case in cases],
        "runs": runs,
    }
    output_path = Path(os.getenv("REAL_EVAL_OUTPUT", "eval-results/real-evaluation-v2.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        mode: {key: value for key, value in run.items() if key not in {"cases"}}
        for mode, run in runs.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
